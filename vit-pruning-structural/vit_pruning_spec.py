# vit_pruning_spec.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn

# -------------------------
# Utilities
# -------------------------
def is_deit_like(model: nn.Module) -> bool:
    return hasattr(model, "blocks") and hasattr(model, "patch_embed")

def get_blocks(model: nn.Module):
    return list(model.blocks)

def get_num_blocks(model: nn.Module) -> int:
    return len(get_blocks(model))

def get_mlp_hidden(block) -> int:
    return block.mlp.fc1.out_features

def _to_list(x: torch.Tensor) -> List[int]:
    return [int(i) for i in x.detach().cpu().tolist()]

# -------------------------
# Pruning Spec
# -------------------------
@dataclass
class PruneSpec:
    method: str
    depth_keep: int
    keep_block_idx: List[int]          # indices in original model (sorted)
    mlp_keep: float
    mlp_keep_idx: Dict[str, List[int]] # key = original block idx as string; value = kept neuron indices (sorted)

    extra: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "depth_keep": self.depth_keep,
            "keep_block_idx": self.keep_block_idx,
            "mlp_keep": self.mlp_keep,
            "mlp_keep_idx": self.mlp_keep_idx,
            "extra": self.extra,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PruneSpec":
        return PruneSpec(
            method=d["method"],
            depth_keep=int(d["depth_keep"]),
            keep_block_idx=[int(x) for x in d["keep_block_idx"]],
            mlp_keep=float(d["mlp_keep"]),
            mlp_keep_idx={str(k): [int(x) for x in v] for k, v in d["mlp_keep_idx"].items()},
            extra=dict(d.get("extra", {})),
        )

# -------------------------
# Method A: "SAViT-style" Taylor pruning spec (practical)
# -------------------------
def make_savit_taylor_spec(
    model: nn.Module,
    calib_loader,
    device: torch.device,
    amp: bool,
    criterion: nn.Module,
    depth_keep: int,
    mlp_keep: float,
    calib_batches: int = 20,
) -> PruneSpec:
    """
    Practical: use |w * grad_w| aggregated per block for block selection;
             and |w * grad_w| per neuron for MLP neuron selection.
    This avoids gating monkeypatching and stays stable.
    """
    assert is_deit_like(model)
    model.train()

    # zero grads
    for p in model.parameters():
        p.grad = None

    # calibration backward
    it = 0
    for images, targets in calib_loader:
        images, targets = images.to(device), targets.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
            loss = criterion(logits, targets)
        loss.backward()
        it += 1
        if it >= calib_batches:
            break

    blocks = get_blocks(model)
    L = len(blocks)

    # Block importance: use attention+mlp weights Taylor magnitude
    block_scores = []
    for bi, blk in enumerate(blocks):
        s = 0.0
        for name, p in blk.named_parameters():
            if p.grad is None:
                continue
            # focus on larger tensors (proj / fc weights) to get stable signal
            if p.ndim >= 2:
                s += float((p * p.grad).abs().mean().detach().cpu())
        block_scores.append(s)

    # choose top depth_keep blocks
    scores_t = torch.tensor(block_scores)
    keep_block_idx = torch.topk(scores_t, k=depth_keep, largest=True).indices.sort().values
    keep_block_idx_list = _to_list(keep_block_idx)

    # MLP neuron importance per kept block: use fc1 rows Taylor magnitude
    mlp_keep_idx: Dict[str, List[int]] = {}
    for orig_bi in keep_block_idx_list:
        blk = blocks[orig_bi]
        fc1 = blk.mlp.fc1
        if fc1.weight.grad is None:
            # fallback: keep first k neurons if grad missing
            H = fc1.out_features
            k = max(1, int(H * mlp_keep))
            mlp_keep_idx[str(orig_bi)] = list(range(k))
            continue

        # score per neuron = mean over input dim of |w*grad|
        score = (fc1.weight * fc1.weight.grad).abs().mean(dim=1)  # [H]
        H = score.numel()
        k = max(1, int(H * mlp_keep))
        keep = torch.topk(score, k=k, largest=True).indices.sort().values
        mlp_keep_idx[str(orig_bi)] = _to_list(keep)

    return PruneSpec(
        method="savit_taylor",
        depth_keep=depth_keep,
        keep_block_idx=keep_block_idx_list,
        mlp_keep=mlp_keep,
        mlp_keep_idx=mlp_keep_idx,
        extra={"calib_batches": calib_batches},
    )

# -------------------------
# Method B: "X-Pruner-style" mask learning spec (practical)
# -------------------------
class _MLPMaskLearner(nn.Module):
    """Trainable per-neuron mask for one block MLP fc1 output."""
    def __init__(self, hidden: int, init: float = 2.0):
        super().__init__()
        self.logits = nn.Parameter(torch.full((hidden,), float(init)))

    def mask(self):
        return torch.sigmoid(self.logits)

def make_xpruner_mask_spec(
    model: nn.Module,
    train_loader,
    device: torch.device,
    amp: bool,
    criterion: nn.Module,
    depth_keep: int,
    mlp_keep: float,
    mask_train_epochs: int = 2,
    mask_lr: float = 5e-3,
    max_steps_per_epoch: int = 200,
) -> PruneSpec:
    """
    Practical: learn per-neuron masks on fc1 outputs with frozen backbone.
    Block selection is based on mean mask strength across blocks (top-k).
    """
    assert is_deit_like(model)
    model.train()

    blocks = get_blocks(model)
    L = len(blocks)

    # attach mask learners per block (do NOT permanently change model weights)
    maskers: List[_MLPMaskLearner] = []
    for blk in blocks:
        H = blk.mlp.fc1.out_features
        maskers.append(_MLPMaskLearner(H).to(device))

    # freeze model weights
    for p in model.parameters():
        p.requires_grad = False
    for m in maskers:
        for p in m.parameters():
            p.requires_grad = True

    opt = torch.optim.AdamW([p for m in maskers for p in m.parameters()], lr=mask_lr, weight_decay=0.0)

    # forward hook: apply mask after fc1 activation
    hooks = []
    def _make_hook(bi: int):
        def hook_fn(module, inp, out):
            # out shape [B,T,H]
            m = maskers[bi].mask().view(1, 1, -1)
            return out * m
        return hook_fn

    for bi, blk in enumerate(blocks):
        hooks.append(blk.mlp.fc1.register_forward_hook(_make_hook(bi)))

    # train masks briefly
    for ep in range(mask_train_epochs):
        steps = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
                loss = criterion(logits, targets)
            loss.backward()
            opt.step()
            steps += 1
            if steps >= max_steps_per_epoch:
                break

    # remove hooks
    for h in hooks:
        h.remove()

    # unfreeze model again for later training
    for p in model.parameters():
        p.requires_grad = True

    # Block score = mean mask value across neurons
    block_scores = torch.tensor([float(maskers[bi].mask().mean().detach().cpu()) for bi in range(L)])
    keep_block_idx = torch.topk(block_scores, k=depth_keep, largest=True).indices.sort().values
    keep_block_idx_list = _to_list(keep_block_idx)

    # Neuron keep per kept block = top-k mask values
    mlp_keep_idx: Dict[str, List[int]] = {}
    for orig_bi in keep_block_idx_list:
        m = maskers[orig_bi].mask().detach().cpu()
        H = m.numel()
        k = max(1, int(H * mlp_keep))
        keep = torch.topk(m, k=k, largest=True).indices.sort().values
        mlp_keep_idx[str(orig_bi)] = _to_list(keep)

    return PruneSpec(
        method="xpruner_masks",
        depth_keep=depth_keep,
        keep_block_idx=keep_block_idx_list,
        mlp_keep=mlp_keep,
        mlp_keep_idx=mlp_keep_idx,
        extra={"mask_train_epochs": mask_train_epochs, "mask_lr": mask_lr, "max_steps_per_epoch": max_steps_per_epoch},
    )

# -------------------------
# Apply Spec (core requirement!)
# -------------------------
def apply_prune_spec_inplace(
    model: nn.Module,
    spec: PruneSpec,
    device: torch.device,
    copy_weights: bool,
) -> nn.Module:
    """
    Make the model match the spec exactly.
    - depth: keep only spec.keep_block_idx blocks, in that order
    - mlp: shrink each kept block MLP fc1/fc2 to kept neuron indices
      * if copy_weights=True: slice from existing weights (dense init)
      * if copy_weights=False: create new layers and random-init them
    """
    assert is_deit_like(model)
    blocks = get_blocks(model)

    # depth prune
    kept_blocks = [blocks[i] for i in spec.keep_block_idx]
    model.blocks = nn.Sequential(*kept_blocks)

    # mlp width prune per kept block (using original block idx mapping)
    for new_pos, orig_idx in enumerate(spec.keep_block_idx):
        blk = model.blocks[new_pos]
        keep = spec.mlp_keep_idx[str(orig_idx)]
        keep_t = torch.tensor(keep, dtype=torch.long, device=device)

        fc1: nn.Linear = blk.mlp.fc1
        fc2: nn.Linear = blk.mlp.fc2
        dim = fc1.in_features
        k = len(keep)

        new_fc1 = nn.Linear(dim, k, bias=(fc1.bias is not None)).to(device)
        new_fc2 = nn.Linear(k, dim, bias=(fc2.bias is not None)).to(device)

        if copy_weights:
            with torch.no_grad():
                new_fc1.weight.copy_(fc1.weight[keep_t, :])
                if fc1.bias is not None:
                    new_fc1.bias.copy_(fc1.bias[keep_t])
                new_fc2.weight.copy_(fc2.weight[:, keep_t])
                if fc2.bias is not None:
                    new_fc2.bias.copy_(fc2.bias)
        # else: keep random init from reset_parameters

        blk.mlp.fc1 = new_fc1
        blk.mlp.fc2 = new_fc2

    return model