"""Hessian eigenspectrum analysis via power iteration.

Computes top-k eigenvalues of the Hessian to measure sharpness/flatness
of the loss landscape at a converged solution.

Usage:
  python src/analyze_hessian.py --checkpoint outputs/p50_ft/best.pth \
      --prune prune50 --mode finetuned --tag p50_ft --top_k 3
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

# Disable fused attention — it doesn't support double backward for Hessian
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

import torch.nn as nn
from tqdm import tqdm

from data import get_landscape_loader
from analyze_surface import load_model

# ──────────────────────────────────────────────────────────────────────
# Hessian-vector product
# ──────────────────────────────────────────────────────────────────────

def hessian_vector_product(model, criterion, images, targets, vector):
    """Compute Hessian-vector product Hv using two backward passes."""
    model.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, targets)

    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True)

    grad_dot_v = sum((g * v).sum() for g, v in zip(grads, vector))

    Hv = torch.autograd.grad(grad_dot_v, params)

    return [hv.detach() for hv in Hv]


# ──────────────────────────────────────────────────────────────────────
# Top-k eigenvalues via power iteration with deflation
# ──────────────────────────────────────────────────────────────────────

def top_k_eigenvalues(model, criterion, loader, device, k=3, n_iter=30):
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    eigenvalues = []
    eigenvectors = []

    for ki in range(k):
        print(f"  Computing eigenvalue {ki+1}/{k}...")
        torch.cuda.empty_cache()

        v = [torch.randn_like(p).to(device) for p in params]

        for iteration in range(n_iter):
            # Orthogonalize against previous eigenvectors
            for prev_v in eigenvectors:
                dot = sum((vi * pvi).sum() for vi, pvi in zip(v, prev_v))
                v = [vi - dot * pvi for vi, pvi in zip(v, prev_v)]

            # Normalize
            v_norm = sum((vi ** 2).sum() for vi in v).sqrt()
            if v_norm > 0:
                v = [vi / v_norm for vi in v]

            # Compute Hv over multiple batches
            Hv = [torch.zeros_like(p).to(device) for p in params]
            n_batches = 0

            for images, targets in loader:
                images, targets = images.to(device), targets.to(device)
                hvp = hessian_vector_product(model, criterion, images, targets, v)
                for i in range(len(Hv)):
                    Hv[i] += hvp[i]
                n_batches += 1
                # Free memory AFTER using hvp
                del images, targets, hvp
                torch.cuda.empty_cache()
                if n_batches >= 5:
                    break

            Hv = [hv / n_batches for hv in Hv]

            eig = sum((hv * vi).sum() for hv, vi in zip(Hv, v)).item()

            hv_norm = sum((hv ** 2).sum() for hv in Hv).sqrt()
            if hv_norm > 0:
                v = [hv / hv_norm for hv in Hv]

        eigenvalues.append(eig)
        eigenvectors.append([vi.clone() for vi in v])
        print(f"    lambda_{ki+1} = {eig:.6f}")

    return eigenvalues


# ──────────────────────────────────────────────────────────────────────
# Trace of Hessian (Hutchinson estimator)
# ──────────────────────────────────────────────────────────────────────

def hessian_trace(model, criterion, loader, device, n_samples=20):
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    trace_sum = 0.0

    for _ in tqdm(range(n_samples), desc="  Trace estimation"):
        v = [torch.randn_like(p).to(device) for p in params]

        Hv = [torch.zeros_like(p).to(device) for p in params]
        n_batches = 0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            hvp = hessian_vector_product(model, criterion, images, targets, v)
            for i in range(len(Hv)):
                Hv[i] += hvp[i]
            n_batches += 1
            del images, targets, hvp
            torch.cuda.empty_cache()
            if n_batches >= 3:
                break
        Hv = [hv / n_batches for hv in Hv]

        trace_sum += sum((hv * vi).sum() for hv, vi in zip(Hv, v)).item()

    return trace_sum / n_samples


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prune", default=None, choices=["prune30", "prune50"])
    parser.add_argument("--mode", default="scratch", choices=["finetuned", "scratch", "unpruned"])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--trace_samples", type=int, default=20)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", default="./outputs/hessian")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.checkpoint, args.prune, args.mode)
    model = model.to(device)

    # SMALL batch size: create_graph=True roughly doubles memory
    loader = get_landscape_loader(data_dir=args.data_dir, num_samples=1000, batch_size=8)
    criterion = nn.CrossEntropyLoss()

    print(f"\nHessian analysis: {args.tag}")
    print(f"{'='*50}")

    # Top-k eigenvalues
    print(f"\nComputing top-{args.top_k} eigenvalues...")
    eigenvalues = top_k_eigenvalues(model, criterion, loader, device, k=args.top_k)

    # Trace
    print(f"\nEstimating Hessian trace...")
    trace = hessian_trace(model, criterion, loader, device, n_samples=args.trace_samples)

    # Results
    results = {
        "tag": args.tag,
        "top_eigenvalues": eigenvalues,
        "trace": trace,
        "max_eigenvalue": eigenvalues[0] if eigenvalues else None,
    }

    print(f"\n{'='*50}")
    print(f"Results for {args.tag}:")
    print(f"  Top eigenvalues: {[f'{e:.4f}' for e in eigenvalues]}")
    print(f"  Max eigenvalue (sharpness): {eigenvalues[0]:.6f}")
    print(f"  Trace: {trace:.6f}")
    print(f"{'='*50}")

    save_path = os.path.join(args.output_dir, f"hessian_{args.tag}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()