# run_deit_cifar100_pipeline.py
import argparse, csv, json, math, os, time
from dataclasses import dataclass, asdict
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets as tv_datasets
from torchvision import transforms

import timm
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data.mixup import Mixup
from timm.scheduler.cosine_lr import CosineLRScheduler
from fvcore.nn import FlopCountAnalysis
from tqdm import tqdm, trange

from vit_pruning_spec import (
    PruneSpec,
    make_savit_taylor_spec,
    make_xpruner_mask_spec,
    apply_prune_spec_inplace,
)

# ---------------------------
# Metrics row (what you need like the paper)
# ---------------------------
@dataclass
class Row:
    run_name: str
    stage: str                 # dense | make_spec | finetune_pruned | scratch_pruned
    prune_method: str          # none | savit_taylor | xpruner_masks
    spec_path: str

    model_name: str
    dataset: str
    num_classes: int
    image_size: int
    batch_size: int
    device: str
    precision: str

    params: int
    flops_per_forward: int

    latency_ms_bs1: float
    latency_ms_batch: float
    throughput_img_s: float

    top1: float
    top5: float

    epochs: int
    steps: int
    total_train_compute_flops: float

    best_epoch: int
    best_top1: float
    best_top5: float
    stopped_early: bool

    scratchB_steps_rule: str


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

@torch.no_grad()
def estimate_flops(model: nn.Module, image_size: int, device: torch.device) -> int:
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    return int(FlopCountAnalysis(model, dummy).total())

@torch.no_grad()
def measure_speed(model: nn.Module, image_size: int, device: torch.device, batch_size: int, amp: bool,
                  warmup: int = 20, iters: int = 50) -> Tuple[float, float, float]:
    model.eval()
    use_cuda = device.type == "cuda"

    def bench(bs: int) -> float:
        x = torch.randn(bs, 3, image_size, image_size, device=device)
        for _ in range(warmup):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                _ = model(x)
        if use_cuda:
            torch.cuda.synchronize()

        if use_cuda:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(iters):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    _ = model(x)
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) / iters
        else:
            t0 = time.perf_counter()
            for _ in range(iters):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    _ = model(x)
            t1 = time.perf_counter()
            return (t1 - t0) * 1000.0 / iters

    lat1 = bench(1)
    latb = bench(batch_size)
    thr = batch_size / (latb / 1000.0)
    return float(lat1), float(latb), float(thr)

@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, topk=(1,5)) -> Tuple[float, float]:
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1,-1).expand_as(pred))
    res=[]
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum().item()
        res.append(100.0 * correct_k / targets.size(0))
    return res[0], res[1]

def write_row(path: str, row: Row) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(row).keys()))
        if not exists:
            w.writeheader()
        w.writerow(asdict(row))

def save_ckpt(path: str, model: nn.Module, optimizer: Optional[optim.Optimizer], epoch: int, extra: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "extra": extra,
    }, path)

def load_ckpt(path: str, model: nn.Module, device: torch.device, strict: bool = False):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=strict)
    return ckpt

# ---------------------------
# Data: CIFAR-100 strong aug
# ---------------------------
def build_loaders(data_dir: str, image_size: int, batch_size: int, num_workers: int,
                  randaugment: bool, ra_n: int, ra_m: int) -> Tuple[DataLoader, DataLoader]:
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)

    aug_list = []
    if randaugment:
        aug_list.append(transforms.RandAugment(num_ops=ra_n, magnitude=ra_m))

    train_tfm = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(image_size, padding=8, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(p=0.5),
        *aug_list,
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value="random"),
    ])

    val_tfm = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = tv_datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_tfm)
    val_ds   = tv_datasets.CIFAR100(root=data_dir, train=False, download=True, transform=val_tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              pin_memory=True, drop_last=False)
    return train_loader, val_loader

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> Tuple[float, float]:
    model.eval()
    top1_sum=0.0; top5_sum=0.0; n=0
    pbar = tqdm(loader, desc="Eval", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
        b1,b5 = accuracy_topk(logits, targets, topk=(1,5))
        bs = targets.size(0)
        top1_sum += b1*bs; top5_sum += b5*bs; n += bs
        pbar.set_postfix(top1=f"{top1_sum/n:.2f}", top5=f"{top5_sum/n:.2f}")
    return top1_sum/n, top5_sum/n

def train_loop(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
               device: torch.device, amp: bool,
               epochs: int, lr: float, weight_decay: float, label_smoothing: float,
               mixup: float, cutmix: float, mixup_prob: float, mixup_switch_prob: float, mixup_mode: str,
               warmup_epochs: int, min_lr: float,
               eval_every: int, patience: int, min_delta: float,
               out_best: str, out_final: str,
               save_optimizer: bool) -> Tuple[float,float,int,int,float,int,float,float,bool]:

    use_mix = (mixup > 0) or (cutmix > 0)
    mixup_fn = None
    if use_mix:
        mixup_fn = Mixup(
            mixup_alpha=mixup, cutmix_alpha=cutmix, cutmix_minmax=None,
            prob=mixup_prob, switch_prob=mixup_switch_prob, mode=mixup_mode,
            label_smoothing=label_smoothing, num_classes=model.num_classes
        )

    criterion = SoftTargetCrossEntropy() if use_mix else LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    updates_per_epoch = len(train_loader)
    total_updates = epochs * updates_per_epoch
    warmup_updates = warmup_epochs * updates_per_epoch

    scheduler = CosineLRScheduler(
        optimizer, t_initial=total_updates, lr_min=min_lr,
        warmup_lr_init=min_lr, warmup_t=warmup_updates,
        cycle_limit=1, t_in_epochs=False
    )

    best_top1=-1.0; best_top5=-1.0; best_epoch=-1
    no_improve=0; stopped_early=False
    steps_done=0; global_update=0

    for ep in trange(epochs, desc="Epochs"):
        model.train()
        scaler = torch.amp.GradScaler('cuda', enabled=(amp and device.type == "cuda"))
        pbar = tqdm(train_loader, desc=f"Train {ep+1}", leave=False)

        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if mixup_fn is not None:
                images, targets = mixup_fn(images, targets)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            global_update += 1
            scheduler.step_update(global_update)
            steps_done += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        if (ep+1) % eval_every == 0:
            v1, v5 = evaluate(model, val_loader, device, amp)
            improved = (v1 > best_top1 + min_delta)
            if improved:
                best_top1, best_top5 = v1, v5
                best_epoch = ep+1
                no_improve = 0
                save_ckpt(out_best, model, optimizer if save_optimizer else None, ep+1,
                          {"val_top1": v1, "val_top5": v5})
            else:
                no_improve += 1
                if no_improve >= patience:
                    stopped_early = True
                    break

    # final eval + save final
    top1, top5 = evaluate(model, val_loader, device, amp)
    save_ckpt(out_final, model, optimizer if save_optimizer else None, (best_epoch if stopped_early else epochs),
              {"final_val_top1": top1, "final_val_top5": top5, "stopped_early": stopped_early})

    total_train_compute = float(steps_done * model._batch_for_compute * model._flops_for_compute)
    return top1, top5, epochs, steps_done, total_train_compute, best_epoch, best_top1, best_top5, stopped_early

# ---------------------------
# Main pipeline
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["dense", "make_spec", "finetune_pruned", "scratch_pruned"])
    ap.add_argument("--prune_method", default="none",
                    choices=["none", "savit_taylor", "xpruner_masks"])
    ap.add_argument("--spec_path", default="")

    ap.add_argument("--run_name", default="run")
    ap.add_argument("--root_out", default="./runs")  # everything saved under here

    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--model", default="deit_small_patch16_224")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--num_classes", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true")

    # training hyperparams
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--label_smoothing", type=float, default=0.1)

    # mixup/cutmix
    ap.add_argument("--mixup", type=float, default=0.8)
    ap.add_argument("--cutmix", type=float, default=1.0)
    ap.add_argument("--mixup_prob", type=float, default=1.0)
    ap.add_argument("--mixup_switch_prob", type=float, default=0.5)
    ap.add_argument("--mixup_mode", type=str, default="batch", choices=["batch", "pair", "elem"])

    # randaugment
    ap.add_argument("--randaugment", action="store_true")
    ap.add_argument("--ra_n", type=int, default=2)
    ap.add_argument("--ra_m", type=int, default=9)

    # scheduler
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--min_lr", type=float, default=1e-6)

    # early stop
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min_delta", type=float, default=0.05)

    # checkpoints / csv
    ap.add_argument("--save_optimizer", action="store_true")

    # dense checkpoint path (for finetune_pruned and make_spec)
    ap.add_argument("--dense_ckpt", default="")

    # pruning budget knobs (fixed pruned architecture)
    ap.add_argument("--depth_keep", type=int, default=10)
    ap.add_argument("--mlp_keep", type=float, default=0.75)

    # spec generation knobs
    ap.add_argument("--calib_batches", type=int, default=20)
    ap.add_argument("--mask_train_epochs", type=int, default=2)
    ap.add_argument("--mask_lr", type=float, default=5e-3)
    ap.add_argument("--mask_steps_per_epoch", type=int, default=200)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_str = f"{device.type}:{torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'}"
    precision = "amp_fp16" if args.amp else "fp32"

    # folder layout
    # runs/<stage or method>/<run_name>/{results.csv, checkpoints...}
    if args.stage == "dense":
        out_dir = os.path.join(args.root_out, "dense", args.run_name)
        csv_path = os.path.join(out_dir, "results.csv")
    else:
        out_dir = os.path.join(args.root_out, args.prune_method, args.stage, args.run_name)
        csv_path = os.path.join(out_dir, "results.csv")
    os.makedirs(out_dir, exist_ok=True)

    best_ckpt = os.path.join(out_dir, "best.pt")
    final_ckpt = os.path.join(out_dir, "final.pt")

    # loaders
    train_loader, val_loader = build_loaders(
        args.data_dir, args.image_size, args.batch_size, args.num_workers,
        args.randaugment, args.ra_n, args.ra_m
    )

    # base model
    # note: pretrained=False always here; dense training is from scratch as you wanted
    model = timm.create_model(args.model, pretrained=False, num_classes=args.num_classes).to(device)

    # FLOPs & speed always measured on the model we will train/eval
    # (after pruning if pruning stage)
    # We attach these helpers for compute calculation:
    # total_train_compute_flops ≈ steps * batch_size * flops_per_forward
    # (consistent definition across runs)
    def attach_compute_meta(m: nn.Module, flops: int):
        m._batch_for_compute = args.batch_size
        m._flops_for_compute = flops

    # -----------------------
    # STAGE: dense training
    # -----------------------
    if args.stage == "dense":
        flops = estimate_flops(model, args.image_size, device)
        attach_compute_meta(model, flops)
        lat1, latb, thr = measure_speed(model, args.image_size, device, args.batch_size, args.amp)

        top1, top5, epochs, steps, total_compute, best_epoch, best_top1, best_top5, stopped_early = train_loop(
            model, train_loader, val_loader, device, args.amp,
            args.epochs, args.lr, args.weight_decay, args.label_smoothing,
            args.mixup, args.cutmix, args.mixup_prob, args.mixup_switch_prob, args.mixup_mode,
            args.warmup_epochs, args.min_lr,
            args.eval_every, args.patience, args.min_delta,
            best_ckpt, final_ckpt,
            args.save_optimizer
        )

        row = Row(
            run_name=args.run_name,
            stage="dense",
            prune_method="none",
            spec_path="",

            model_name=f"{args.model}(dense_scratch)",
            dataset="CIFAR-100",
            num_classes=args.num_classes,
            image_size=args.image_size,
            batch_size=args.batch_size,
            device=dev_str,
            precision=precision,

            params=count_params(model),
            flops_per_forward=flops,
            latency_ms_bs1=lat1,
            latency_ms_batch=latb,
            throughput_img_s=thr,

            top1=top1, top5=top5,
            epochs=epochs, steps=steps,
            total_train_compute_flops=total_compute,

            best_epoch=best_epoch,
            best_top1=best_top1,
            best_top5=best_top5,
            stopped_early=stopped_early,

            scratchB_steps_rule="steps_B = steps_dense * (FLOPs_dense / FLOPs_pruned)"
        )
        write_row(csv_path, row)
        print("DONE dense. Saved:", csv_path)
        return

    # For all other stages, we need either:
    # - to make spec from dense weights (dense_ckpt required)
    # - to finetune from dense weights (dense_ckpt required)
    if args.stage in ["make_spec", "finetune_pruned"] and not args.dense_ckpt:
        raise RuntimeError("--dense_ckpt is required for make_spec and finetune_pruned")

    # -----------------------
    # STAGE: make pruning spec (Method A/B)
    # -----------------------
    if args.stage == "make_spec":
        # load dense weights into model
        load_ckpt(args.dense_ckpt, model, device, strict=False)

        # use a hard-label criterion for spec generation
        criterion_hard = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)

        # build a small calibration loader (just reuse train_loader, we break early inside)
        if args.prune_method == "savit_taylor":
            spec = make_savit_taylor_spec(
                model, train_loader, device, args.amp, criterion_hard,
                depth_keep=args.depth_keep, mlp_keep=args.mlp_keep,
                calib_batches=args.calib_batches
            )
        elif args.prune_method == "xpruner_masks":
            spec = make_xpruner_mask_spec(
                model, train_loader, device, args.amp, criterion_hard,
                depth_keep=args.depth_keep, mlp_keep=args.mlp_keep,
                mask_train_epochs=args.mask_train_epochs,
                mask_lr=args.mask_lr,
                max_steps_per_epoch=args.mask_steps_per_epoch
            )
        else:
            raise RuntimeError("prune_method must be savit_taylor or xpruner_masks for make_spec")

        spec_path = os.path.join(out_dir, "spec.json")
        with open(spec_path, "w") as f:
            json.dump(spec.to_dict(), f, indent=2)
        print("Saved spec:", spec_path)

        # log a row for traceability
        flops = estimate_flops(model, args.image_size, device)
        attach_compute_meta(model, flops)
        lat1, latb, thr = measure_speed(model, args.image_size, device, args.batch_size, args.amp)

        top1, top5 = evaluate(model, val_loader, device, args.amp)

        row = Row(
            run_name=args.run_name,
            stage="make_spec",
            prune_method=args.prune_method,
            spec_path=spec_path,

            model_name=f"{args.model}(dense_loaded_for_spec)",
            dataset="CIFAR-100",
            num_classes=args.num_classes,
            image_size=args.image_size,
            batch_size=args.batch_size,
            device=dev_str,
            precision=precision,

            params=count_params(model),
            flops_per_forward=flops,
            latency_ms_bs1=lat1,
            latency_ms_batch=latb,
            throughput_img_s=thr,

            top1=top1, top5=top5,
            epochs=0, steps=0, total_train_compute_flops=0.0,
            best_epoch=-1, best_top1=-1.0, best_top5=-1.0, stopped_early=False,
            scratchB_steps_rule="steps_B = steps_dense * (FLOPs_dense / FLOPs_pruned)"
        )
        write_row(csv_path, row)
        return

    # For pruned stages, we need spec_path
    if not args.spec_path:
        raise RuntimeError("--spec_path is required for finetune_pruned and scratch_pruned")
    with open(args.spec_path, "r") as f:
        spec = PruneSpec.from_dict(json.load(f))

    # -----------------------
    # STAGE: finetune pruned (copy dense weights into pruned model)
    # -----------------------
    if args.stage == "finetune_pruned":
        # load dense weights into model
        load_ckpt(args.dense_ckpt, model, device, strict=False)

        # apply spec and COPY weights into new smaller layers
        model = apply_prune_spec_inplace(model, spec, device=device, copy_weights=True).to(device)

        flops = estimate_flops(model, args.image_size, device)
        attach_compute_meta(model, flops)
        lat1, latb, thr = measure_speed(model, args.image_size, device, args.batch_size, args.amp)

        top1, top5, epochs, steps, total_compute, best_epoch, best_top1, best_top5, stopped_early = train_loop(
            model, train_loader, val_loader, device, args.amp,
            args.epochs, args.lr, args.weight_decay, args.label_smoothing,
            args.mixup, args.cutmix, args.mixup_prob, args.mixup_switch_prob, args.mixup_mode,
            args.warmup_epochs, args.min_lr,
            args.eval_every, args.patience, args.min_delta,
            best_ckpt, final_ckpt,
            args.save_optimizer
        )

        row = Row(
            run_name=args.run_name,
            stage="finetune_pruned",
            prune_method=spec.method,
            spec_path=args.spec_path,

            model_name=f"{args.model}(PRUNED:{spec.method})",
            dataset="CIFAR-100",
            num_classes=args.num_classes,
            image_size=args.image_size,
            batch_size=args.batch_size,
            device=dev_str,
            precision=precision,

            params=count_params(model),
            flops_per_forward=flops,
            latency_ms_bs1=lat1,
            latency_ms_batch=latb,
            throughput_img_s=thr,

            top1=top1, top5=top5,
            epochs=epochs, steps=steps, total_train_compute_flops=total_compute,
            best_epoch=best_epoch, best_top1=best_top1, best_top5=best_top5, stopped_early=stopped_early,
            scratchB_steps_rule="steps_B = steps_dense * (FLOPs_dense / FLOPs_pruned)"
        )
        write_row(csv_path, row)
        print("DONE finetune_pruned. Saved:", csv_path)
        return

    # -----------------------
    # STAGE: scratch pruned (same architecture, random init weights)
    # -----------------------
    if args.stage == "scratch_pruned":
        # model is already random init; just apply spec WITHOUT copying weights
        model = apply_prune_spec_inplace(model, spec, device=device, copy_weights=False).to(device)

        flops = estimate_flops(model, args.image_size, device)
        attach_compute_meta(model, flops)
        lat1, latb, thr = measure_speed(model, args.image_size, device, args.batch_size, args.amp)

        top1, top5, epochs, steps, total_compute, best_epoch, best_top1, best_top5, stopped_early = train_loop(
            model, train_loader, val_loader, device, args.amp,
            args.epochs, args.lr, args.weight_decay, args.label_smoothing,
            args.mixup, args.cutmix, args.mixup_prob, args.mixup_switch_prob, args.mixup_mode,
            args.warmup_epochs, args.min_lr,
            args.eval_every, args.patience, args.min_delta,
            best_ckpt, final_ckpt,
            args.save_optimizer
        )

        row = Row(
            run_name=args.run_name,
            stage="scratch_pruned",
            prune_method=spec.method,
            spec_path=args.spec_path,

            model_name=f"{args.model}(PRUNED:{spec.method})",
            dataset="CIFAR-100",
            num_classes=args.num_classes,
            image_size=args.image_size,
            batch_size=args.batch_size,
            device=dev_str,
            precision=precision,

            params=count_params(model),
            flops_per_forward=flops,
            latency_ms_bs1=lat1,
            latency_ms_batch=latb,
            throughput_img_s=thr,

            top1=top1, top5=top5,
            epochs=epochs, steps=steps, total_train_compute_flops=total_compute,
            best_epoch=best_epoch, best_top1=best_top1, best_top5=best_top5, stopped_early=stopped_early,
            scratchB_steps_rule="steps_B = steps_dense * (FLOPs_dense / FLOPs_pruned)"
        )
        write_row(csv_path, row)
        print("DONE scratch_pruned. Saved:", csv_path)
        return


if __name__ == "__main__":
    main()