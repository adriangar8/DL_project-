"""2D loss surface visualization using filter-normalized random directions.

Based on: Li et al., "Visualizing the Loss Landscape of Neural Nets" (NeurIPS 2018).

Usage:
  python src/analyze_surface.py --checkpoint outputs/p50_ft/best.pth \
      --prune prune50 --mode finetuned --resolution 41
"""

import argparse
import os
import sys
import copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

from data import get_landscape_loader
from models import (
    create_pruned_model,
    create_unpruned_model,
    DEIT_SMALL_CONFIG,
)
import timm


# ──────────────────────────────────────────────────────────────────────
# Parameter vector utilities
# ──────────────────────────────────────────────────────────────────────

def get_params(model):
    """Get list of parameter tensors (detached copies)."""
    return [p.data.clone() for p in model.parameters()]


def set_params(model, param_list):
    """Set model parameters from a list of tensors."""
    for p, new_p in zip(model.parameters(), param_list):
        p.data.copy_(new_p)


def params_add(params, direction, alpha):
    """Return params + alpha * direction."""
    return [p + alpha * d for p, d in zip(params, direction)]


# ──────────────────────────────────────────────────────────────────────
# Filter-normalized random direction (Li et al. 2018)
# ──────────────────────────────────────────────────────────────────────

def get_random_direction(model):
    """Generate a random direction with the same structure as model params."""
    direction = []
    for p in model.parameters():
        d = torch.randn_like(p)
        direction.append(d)
    return direction


def normalize_direction(direction, model_params):
    """Filter normalization: scale each layer's direction to match the layer's param norm.

    For conv/linear layers (>=2D), normalize per filter.
    For bias/1D params, normalize globally.
    """
    normalized = []
    for d, p in zip(direction, model_params):
        if p.dim() <= 1:
            # Bias or 1D param: scale to match norm
            d_norm = d.norm()
            p_norm = p.norm()
            if d_norm > 0:
                normalized.append(d * (p_norm / d_norm))
            else:
                normalized.append(d)
        else:
            # Multi-dim param: normalize per filter (first dimension)
            new_d = torch.zeros_like(d)
            for i in range(d.shape[0]):
                d_filter = d[i]
                p_filter = p[i]
                d_norm = d_filter.norm()
                p_norm = p_filter.norm()
                if d_norm > 0:
                    new_d[i] = d_filter * (p_norm / d_norm)
            normalized.append(new_d)
    return normalized


# ──────────────────────────────────────────────────────────────────────
# Loss evaluation
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_loss(model, loader, criterion, device):
    model.eval()
    total_loss, total = 0.0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
    return total_loss / total


# ──────────────────────────────────────────────────────────────────────
# Compute 2D loss surface
# ──────────────────────────────────────────────────────────────────────

def compute_loss_surface(model, loader, device, resolution=41, distance=1.0, seed=0):
    """Compute 2D loss surface around the model's converged parameters."""
    torch.manual_seed(seed)
    criterion = nn.CrossEntropyLoss()

    # Save original parameters
    theta_star = get_params(model)

    # Generate two filter-normalized random directions
    dir1 = normalize_direction(get_random_direction(model), theta_star)
    dir2 = normalize_direction(get_random_direction(model), theta_star)

    # Move directions to same device
    dir1 = [d.to(device) for d in dir1]
    dir2 = [d.to(device) for d in dir2]
    theta_star = [t.to(device) for t in theta_star]

    alphas = np.linspace(-distance, distance, resolution)
    betas = np.linspace(-distance, distance, resolution)
    losses = np.zeros((resolution, resolution))

    total_evals = resolution * resolution
    pbar = tqdm(total=total_evals, desc="Loss surface")

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            new_params = [
                t + alpha * d1 + beta * d2
                for t, d1, d2 in zip(theta_star, dir1, dir2)
            ]
            set_params(model, new_params)
            losses[i, j] = eval_loss(model, loader, criterion, device)
            pbar.update(1)

    pbar.close()

    # Restore original parameters
    set_params(model, theta_star)

    return alphas, betas, losses


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

def plot_surface(alphas, betas, losses, title="Loss Surface", save_path=None):
    """Plot 2D contour and 3D surface."""
    A, B = np.meshgrid(alphas, betas, indexing='ij')

    fig = plt.figure(figsize=(14, 5))

    # Contour plot
    ax1 = fig.add_subplot(121)
    vmin, vmax = losses.min(), np.percentile(losses, 95)
    levels = np.linspace(vmin, vmax, 30)

    cs = ax1.contourf(A, B, losses, levels=levels, cmap='viridis')
    ax1.contour(A, B, losses, levels=levels, colors='white', linewidths=0.3, alpha=0.5)
    ax1.plot(0, 0, 'r*', markersize=15, label='Converged')
    ax1.set_xlabel('Direction 1')
    ax1.set_ylabel('Direction 2')
    ax1.set_title(f'{title} (contour)')
    ax1.legend()
    fig.colorbar(cs, ax=ax1, label='Loss')

    # 3D surface
    ax3d = fig.add_subplot(122, projection='3d')
    ax3d.plot_surface(A, B, losses, cmap='viridis', alpha=0.8,
                      rstride=2, cstride=2, edgecolor='none')
    ax3d.set_xlabel('Direction 1')
    ax3d.set_ylabel('Direction 2')
    ax3d.set_zlabel('Loss')
    ax3d.set_title(f'{title} (3D)')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def load_model(checkpoint, prune_config, mode):
    """Load a trained model from checkpoint.

    Creates the architecture shell (no pretrained download needed)
    and loads the saved weights on top.
    """
    if prune_config is None:
        # Unpruned DeiT-Small architecture (random init, overwritten by checkpoint)
        model = timm.create_model(
            "deit_small_patch16_224", pretrained=False, num_classes=100,
        )
    else:
        # Pruned architecture (random init, overwritten by checkpoint)
        model = create_pruned_model(prune_config, num_classes=100)

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prune", default=None, choices=["prune30", "prune50"])
    parser.add_argument("--mode", default="scratch", choices=["finetuned", "scratch", "unpruned"])
    parser.add_argument("--resolution", type=int, default=41)
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", default="./outputs/landscape")
    parser.add_argument("--tag", default=None, help="Label for the plot")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint, args.prune, args.mode)
    model = model.to(device)

    # Data
    loader = get_landscape_loader(
        data_dir=args.data_dir, num_samples=args.num_samples,
    )

    tag = args.tag or os.path.basename(os.path.dirname(args.checkpoint))
    print(f"\nComputing loss surface for: {tag}")
    print(f"  Resolution: {args.resolution}x{args.resolution}")
    print(f"  Distance: ±{args.distance}")
    print(f"  Samples: {args.num_samples}")

    # Compute
    alphas, betas, losses = compute_loss_surface(
        model, loader, device,
        resolution=args.resolution,
        distance=args.distance,
    )

    # Save raw data
    np.savez(
        os.path.join(args.output_dir, f"surface_{tag}.npz"),
        alphas=alphas, betas=betas, losses=losses,
    )

    # Plot
    plot_surface(
        alphas, betas, losses,
        title=tag,
        save_path=os.path.join(args.output_dir, f"surface_{tag}.png"),
    )

    print(f"  Loss at center: {losses[len(alphas)//2, len(betas)//2]:.4f}")
    print(f"  Loss min: {losses.min():.4f}")
    print(f"  Loss max: {losses.max():.4f}")


if __name__ == "__main__":
    main()
