"""Training script for pruned and unpruned DeiT models on CIFAR-100.

Usage:
  # Fine-tune unpruned baseline
  python src/train.py --mode finetune_unpruned --epochs 20 --lr 1e-4 --tag unpruned

  # Fine-tune pruned model (inherits weights)
  python src/train.py --mode finetune_pruned --prune prune50 --epochs 20 --lr 1e-4 --tag p50_ft

  # Train pruned architecture from scratch (same epochs)
  python src/train.py --mode scratch --prune prune50 --epochs 20 --lr 5e-4 --tag p50_scrE

  # Train pruned architecture from scratch (more epochs = Scratch-B)
  python src/train.py --mode scratch --prune prune50 --epochs 50 --lr 5e-4 --tag p50_scrB
"""

import argparse
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data import get_dataloaders
from models import (
    create_unpruned_model,
    create_pruned_model,
    create_finetuned_pruned_model,
    print_model_summary,
    count_parameters,
)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        total += images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.1f}%")

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        total += images.size(0)

    return total_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["finetune_unpruned", "finetune_pruned", "scratch"])
    parser.add_argument("--prune", default=None, choices=["prune30", "prune50"],
                        help="Pruning config (required for finetune_pruned and scratch)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--tag", required=True, help="Experiment name for saving")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Validate
    if args.mode in ("finetune_pruned", "scratch") and args.prune is None:
        parser.error("--prune is required for finetune_pruned and scratch modes")

    # Reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──
    train_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
    )

    # ── Model ──
    if args.mode == "finetune_unpruned":
        model = create_unpruned_model(num_classes=100)
        print_model_summary(model, "DeiT-Small (unpruned, pretrained)")

    elif args.mode == "finetune_pruned":
        model = create_finetuned_pruned_model(args.prune, num_classes=100)
        print_model_summary(model, f"DeiT-Small ({args.prune}, fine-tune)")

    elif args.mode == "scratch":
        model = create_pruned_model(args.prune, num_classes=100)
        print_model_summary(model, f"DeiT-Small ({args.prune}, scratch)")

    model = model.to(device)

    # ── Optimizer + scheduler ──
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Cosine schedule with linear warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    # ── Training loop ──
    save_dir = os.path.join(args.output_dir, args.tag)
    os.makedirs(save_dir, exist_ok=True)

    best_acc = 0.0
    history = []

    print(f"\nTraining {args.tag} for {args.epochs} epochs (lr={args.lr})\n")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch,
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train {train_acc:.2f}% | Test {test_acc:.2f}% | "
              f"LR {lr_now:.2e}")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "test_loss": test_loss, "test_acc": test_acc, "lr": lr_now,
        })

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))

    elapsed = time.time() - start_time

    # Save final model + metadata
    torch.save(model.state_dict(), os.path.join(save_dir, "last.pth"))

    meta = {
        "mode": args.mode, "prune": args.prune, "epochs": args.epochs,
        "lr": args.lr, "best_acc": best_acc, "final_acc": test_acc,
        "params": count_parameters(model), "elapsed_sec": elapsed,
        "history": history,
    }
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Done: {args.tag}")
    print(f"  Best test acc: {best_acc:.2f}%")
    print(f"  Final test acc: {test_acc:.2f}%")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"  Saved to: {save_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
