"""Pruned DeiT model creation and weight transfer.

Pruning strategy: remove attention heads + reduce MLP hidden dim.
The embed_dim (384) stays the same — only attention and MLP internals shrink.
This preserves residual connections and lets us cleanly transfer weights.

Configs:
  prune30: keep 4/6 heads, MLP hidden 1024/1536 (~30% param reduction)
  prune50: keep 3/6 heads, MLP hidden 768/1536  (~50% param reduction)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ──────────────────────────────────────────────────────────────────────
# Pruned modules
# ──────────────────────────────────────────────────────────────────────

class PrunedAttention(nn.Module):
    """Multi-head attention with arbitrary num_heads decoupled from embed_dim."""

    def __init__(self, dim, num_heads, head_dim=64, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        inner_dim = num_heads * head_dim

        self.qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            attn = attn + attn_mask
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PrunedMlp(nn.Module):
    """MLP with configurable hidden dim."""

    def __init__(self, in_features, hidden_features, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ──────────────────────────────────────────────────────────────────────
# Pruning configs
# ──────────────────────────────────────────────────────────────────────

PRUNE_CONFIGS = {
    "prune30": {"num_heads_keep": 4, "mlp_hidden": 1024},
    "prune50": {"num_heads_keep": 3, "mlp_hidden": 768},
}

DEIT_SMALL_CONFIG = {
    "embed_dim": 384,
    "num_heads": 6,
    "head_dim": 64,
    "depth": 12,
    "mlp_hidden": 1536,
}


# ──────────────────────────────────────────────────────────────────────
# Head importance scoring
# ──────────────────────────────────────────────────────────────────────

def compute_head_importance(model):
    """Compute importance of each head position averaged across all layers.

    Uses L1-norm of each head's output projection weights.
    Returns: Tensor of shape (num_heads,) with importance scores.
    """
    cfg = DEIT_SMALL_CONFIG
    num_heads = cfg["num_heads"]
    head_dim = cfg["head_dim"]
    importance = torch.zeros(num_heads)

    for block in model.blocks:
        proj_w = block.attn.proj.weight.data  # (384, 384)
        for h in range(num_heads):
            cols = slice(h * head_dim, (h + 1) * head_dim)
            importance[h] += proj_w[:, cols].abs().sum().item()

    importance /= len(model.blocks)  # average across layers
    return importance


def select_heads_to_keep(model, num_keep):
    """Select the most important head indices to keep."""
    importance = compute_head_importance(model)
    _, indices = importance.topk(num_keep)
    return sorted(indices.tolist())


# ──────────────────────────────────────────────────────────────────────
# MLP neuron importance
# ──────────────────────────────────────────────────────────────────────

def select_mlp_neurons(fc1_weight, num_keep):
    """Select top-k MLP neurons by L1-norm of fc1 rows."""
    importance = fc1_weight.data.abs().sum(dim=1)  # (mlp_hidden,)
    _, indices = importance.topk(num_keep)
    return sorted(indices.tolist())


# ──────────────────────────────────────────────────────────────────────
# Build pruned model (random init — for scratch training)
# ──────────────────────────────────────────────────────────────────────

def create_pruned_model(prune_config, num_classes=100):
    """Create a pruned DeiT-Small with random initialization."""
    cfg = DEIT_SMALL_CONFIG
    pc = PRUNE_CONFIGS[prune_config]

    # Start from a randomly initialized DeiT-Small
    model = timm.create_model(
        "deit_small_patch16_224", pretrained=False, num_classes=num_classes,
    )

    # Replace attention and MLP in each block
    for block in model.blocks:
        block.attn = PrunedAttention(
            dim=cfg["embed_dim"],
            num_heads=pc["num_heads_keep"],
            head_dim=cfg["head_dim"],
        )
        block.mlp = PrunedMlp(
            in_features=cfg["embed_dim"],
            hidden_features=pc["mlp_hidden"],
        )

    return model


# ──────────────────────────────────────────────────────────────────────
# Build pruned model from pretrained (for fine-tuning)
# ──────────────────────────────────────────────────────────────────────

def create_finetuned_pruned_model(prune_config, num_classes=100):
    """Load pretrained DeiT-Small, prune heads + MLP, transfer weights."""
    cfg = DEIT_SMALL_CONFIG
    pc = PRUNE_CONFIGS[prune_config]
    head_dim = cfg["head_dim"]

    # Load pretrained
    pretrained = timm.create_model(
        "deit_small_patch16_224", pretrained=True, num_classes=num_classes,
    )

    # Determine which heads to keep
    heads_to_keep = select_heads_to_keep(pretrained, pc["num_heads_keep"])
    print(f"Keeping heads: {heads_to_keep}")

    # Create pruned model (random init) — we'll overwrite weights
    pruned = create_pruned_model(prune_config, num_classes=num_classes)

    # ── Transfer weights ──

    # Patch embedding, cls_token, pos_embed, final norm — no change in shape
    pruned.patch_embed.load_state_dict(pretrained.patch_embed.state_dict())
    pruned.cls_token.data.copy_(pretrained.cls_token.data)
    pruned.pos_embed.data.copy_(pretrained.pos_embed.data)
    pruned.norm.load_state_dict(pretrained.norm.state_dict())

    # Head (classifier) — random init for new num_classes if pretrained was 1000-class
    # If num_classes matches, we could transfer; for CIFAR-100 vs ImageNet, skip.
    # (head is randomly initialized in both pretrained and pruned when num_classes != 1000)

    # Each transformer block
    for i, (old_block, new_block) in enumerate(zip(pretrained.blocks, pruned.blocks)):
        # ── Norm layers (same shape) ──
        new_block.norm1.load_state_dict(old_block.norm1.state_dict())
        new_block.norm2.load_state_dict(old_block.norm2.state_dict())

        # ── Attention QKV ──
        old_qkv_w = old_block.attn.qkv.weight.data   # (1152, 384)
        old_qkv_b = old_block.attn.qkv.bias.data      # (1152,)
        embed_dim = cfg["embed_dim"]

        # Select rows for kept heads in Q, K, V sections
        rows = []
        for qkv_idx in range(3):
            for h in heads_to_keep:
                start = qkv_idx * embed_dim + h * head_dim
                rows.extend(range(start, start + head_dim))

        new_block.attn.qkv.weight.data.copy_(old_qkv_w[rows])
        new_block.attn.qkv.bias.data.copy_(old_qkv_b[rows])

        # ── Attention output projection ──
        old_proj_w = old_block.attn.proj.weight.data   # (384, 384)
        old_proj_b = old_block.attn.proj.bias.data     # (384,)

        cols = []
        for h in heads_to_keep:
            cols.extend(range(h * head_dim, (h + 1) * head_dim))

        new_block.attn.proj.weight.data.copy_(old_proj_w[:, cols])
        new_block.attn.proj.bias.data.copy_(old_proj_b)

        # ── MLP ──
        old_fc1_w = old_block.mlp.fc1.weight.data  # (1536, 384)
        old_fc1_b = old_block.mlp.fc1.bias.data     # (1536,)
        old_fc2_w = old_block.mlp.fc2.weight.data   # (384, 1536)
        old_fc2_b = old_block.mlp.fc2.bias.data     # (384,)

        neurons = select_mlp_neurons(old_fc1_w, pc["mlp_hidden"])

        new_block.mlp.fc1.weight.data.copy_(old_fc1_w[neurons])
        new_block.mlp.fc1.bias.data.copy_(old_fc1_b[neurons])
        new_block.mlp.fc2.weight.data.copy_(old_fc2_w[:, neurons])
        new_block.mlp.fc2.bias.data.copy_(old_fc2_b)

        # ── Copy drop_path, layer_scale if they exist ──
        if hasattr(old_block, 'drop_path1') and hasattr(old_block.drop_path1, 'drop_prob'):
            new_block.drop_path1.drop_prob = old_block.drop_path1.drop_prob
            new_block.drop_path2.drop_prob = old_block.drop_path2.drop_prob

    return pruned


# ──────────────────────────────────────────────────────────────────────
# Unpruned baseline (pretrained, fine-tuned on CIFAR-100)
# ──────────────────────────────────────────────────────────────────────

def create_unpruned_model(num_classes=100):
    """Load pretrained DeiT-Small for fine-tuning on CIFAR-100."""
    return timm.create_model(
        "deit_small_patch16_224", pretrained=True, num_classes=num_classes,
    )


# ──────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model, name="Model"):
    total = count_parameters(model)
    attn_params = sum(
        sum(p.numel() for p in block.attn.parameters())
        for block in model.blocks
    )
    mlp_params = sum(
        sum(p.numel() for p in block.mlp.parameters())
        for block in model.blocks
    )
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"  Total params:     {total:,}")
    print(f"  Attention params: {attn_params:,}")
    print(f"  MLP params:       {mlp_params:,}")
    print(f"  Other params:     {total - attn_params - mlp_params:,}")
    print(f"{'='*50}\n")
