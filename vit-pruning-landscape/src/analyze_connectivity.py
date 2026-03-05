"""Linear mode connectivity between fine-tuned and scratch-trained models.

Interpolates weights: θ(α) = α·θ_A + (1-α)·θ_B for α ∈ [0, 1]
and evaluates loss/accuracy along the path.

If no barrier exists → same basin → architecture determines solution.
If high barrier → different basins → inherited weights find a different region.

Usage:
  python src/analyze_connectivity.py \
      --checkpoint_a outputs/p50_ft/best.pth \
      --checkpoint_b outputs/p50_scrB/best.pth \
      --prune prune50 --n_points 21
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data import get_landscape_loader
from models import create_pruned_model
import timm


# ──────────────────────────────────────────────────────────────────────
# Interpolation
# ──────────────────────────────────────────────────────────────────────

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


def linear_interpolation(model, state_a, state_b, alpha):
    """Set model parameters to α * A + (1 - α) * B."""
    interpolated = {}
    for key in state_a:
        interpolated[key] = alpha * state_a[key] + (1 - alpha) * state_b[key]
    model.load_state_dict(interpolated)


def compute_connectivity(model, state_a, state_b, loader, device, n_points=21):
    """Evaluate loss and accuracy along the linear interpolation path."""
    criterion = nn.CrossEntropyLoss()
    alphas = np.linspace(0, 1, n_points)

    losses = []
    accuracies = []

    for alpha in alphas:
        linear_interpolation(model, state_a, state_b, alpha)
        model = model.to(device)
        loss, acc = evaluate(model, loader, criterion, device)
        losses.append(loss)
        accuracies.append(acc)
        print(f"  α={alpha:.2f}: loss={loss:.4f}, acc={acc:.2f}%")

    return alphas, np.array(losses), np.array(accuracies)


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

def plot_connectivity(alphas, losses, accuracies, label_a="Model A", label_b="Model B",
                      title="Linear Mode Connectivity", save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss
    ax1.plot(alphas, losses, 'b-o', markersize=4, linewidth=2)
    ax1.axvline(0, color='red', linestyle='--', alpha=0.5, label=label_a)
    ax1.axvline(1, color='green', linestyle='--', alpha=0.5, label=label_b)
    ax1.set_xlabel('α (interpolation)')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss along interpolation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Barrier metric
    endpoint_loss = max(losses[0], losses[-1])
    barrier = losses.max() - endpoint_loss
    ax1.annotate(f'Barrier: {barrier:.4f}', xy=(alphas[losses.argmax()], losses.max()),
                 fontsize=10, ha='center', va='bottom', color='red')

    # Accuracy
    ax2.plot(alphas, accuracies, 'g-o', markersize=4, linewidth=2)
    ax2.axvline(0, color='red', linestyle='--', alpha=0.5, label=label_a)
    ax2.axvline(1, color='green', linestyle='--', alpha=0.5, label=label_b)
    ax2.set_xlabel('α (interpolation)')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy along interpolation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_a", required=True, help="First model (e.g., fine-tuned)")
    parser.add_argument("--checkpoint_b", required=True, help="Second model (e.g., scratch)")
    parser.add_argument("--label_a", default="Fine-tuned")
    parser.add_argument("--label_b", default="Scratch-B")
    parser.add_argument("--prune", default=None, choices=["prune30", "prune50"])
    parser.add_argument("--n_points", type=int, default=21)
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", default="./outputs/connectivity")
    parser.add_argument("--tag", default="connectivity")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Create model shell (same architecture for both checkpoints)
    if args.prune:
        model = create_pruned_model(args.prune, num_classes=100)
    else:
        model = timm.create_model(
            "deit_small_patch16_224", pretrained=False, num_classes=100,
        )

    # Load both state dicts
    state_a = torch.load(args.checkpoint_a, map_location="cpu", weights_only=True)
    state_b = torch.load(args.checkpoint_b, map_location="cpu", weights_only=True)

    # Data
    loader = get_landscape_loader(
        data_dir=args.data_dir, num_samples=args.num_samples,
    )

    print(f"\nMode connectivity: {args.tag}")
    print(f"  A: {args.checkpoint_a} ({args.label_a})")
    print(f"  B: {args.checkpoint_b} ({args.label_b})")
    print(f"  Points: {args.n_points}")
    print()

    alphas, losses, accuracies = compute_connectivity(
        model, state_a, state_b, loader, device, n_points=args.n_points,
    )

    # Compute barrier
    endpoint_loss = max(losses[0], losses[-1])
    barrier = losses.max() - endpoint_loss

    results = {
        "tag": args.tag,
        "checkpoint_a": args.checkpoint_a,
        "checkpoint_b": args.checkpoint_b,
        "alphas": alphas.tolist(),
        "losses": losses.tolist(),
        "accuracies": accuracies.tolist(),
        "barrier": float(barrier),
        "loss_a": float(losses[0]),
        "loss_b": float(losses[-1]),
        "acc_a": float(accuracies[0]),
        "acc_b": float(accuracies[-1]),
    }

    print(f"\n{'='*50}")
    print(f"Barrier (max_loss - max_endpoint): {barrier:.4f}")
    print(f"  → {'LOW barrier (same basin)' if barrier < 0.5 else 'HIGH barrier (different basins)'}")
    print(f"{'='*50}")

    with open(os.path.join(args.output_dir, f"{args.tag}.json"), "w") as f:
        json.dump(results, f, indent=2)

    plot_connectivity(
        alphas, losses, accuracies,
        label_a=args.label_a, label_b=args.label_b,
        title=f"Mode Connectivity: {args.tag}",
        save_path=os.path.join(args.output_dir, f"{args.tag}.png"),
    )


if __name__ == "__main__":
    main()
