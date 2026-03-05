"""Generate summary figures comparing all experimental conditions.

Run after all training and analysis is complete.

Usage:
  python src/plot_results.py --output_dir outputs
"""

import argparse
import os
import json
import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
})


def load_meta(path):
    with open(path) as f:
        return json.load(f)


def plot_accuracy_comparison(output_dir, save_dir):
    """Bar chart: accuracy comparison across conditions for each pruning level."""
    configs = {}
    for meta_path in sorted(glob.glob(os.path.join(output_dir, "*/meta.json"))):
        tag = os.path.basename(os.path.dirname(meta_path))
        meta = load_meta(meta_path)
        configs[tag] = meta

    if not configs:
        print("No meta.json files found. Skipping accuracy comparison.")
        return

    # Group by pruning level
    fig, ax = plt.subplots(figsize=(10, 5))
    groups = {}
    for tag, meta in configs.items():
        prune = meta.get("prune") or "none"
        mode = meta["mode"]
        key = f"{prune}_{mode}"
        groups[key] = {"tag": tag, "acc": meta["best_acc"], "prune": prune, "mode": mode}

    # Organize for plotting
    prune_levels = sorted(set(g["prune"] for g in groups.values()))
    modes = ["finetune_pruned", "scratch"]  # scratch-E and scratch-B are both "scratch"

    tags_by_prune = {}
    for key, g in groups.items():
        pl = g["prune"]
        if pl not in tags_by_prune:
            tags_by_prune[pl] = []
        tags_by_prune[pl].append(g)

    # Print table
    print("\n" + "="*70)
    print(f"{'Tag':<20} {'Mode':<20} {'Prune':<10} {'Best Acc':>10} {'Params':>12}")
    print("-"*70)
    for tag, meta in sorted(configs.items()):
        prune = meta.get("prune") or "none"
        print(f"{tag:<20} {meta['mode']:<20} {prune:<10} {meta['best_acc']:>10.2f}% {meta['params']:>12,}")
    print("="*70)

    # Bar chart
    tags = list(configs.keys())
    accs = [configs[t]["best_acc"] for t in tags]
    colors = []
    for t in tags:
        m = configs[t]["mode"]
        if "unpruned" in m:
            colors.append("#2196F3")
        elif "finetune" in m:
            colors.append("#FF9800")
        else:
            colors.append("#4CAF50")

    bars = ax.bar(range(len(tags)), accs, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy Comparison: Fine-tuned vs. Scratch")
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{acc:.1f}', ha='center', va='bottom', fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2196F3', label='Unpruned baseline'),
        Patch(facecolor='#FF9800', label='Pruned + Fine-tuned'),
        Patch(facecolor='#4CAF50', label='Pruned + Scratch'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "accuracy_comparison.png"), dpi=150)
    print(f"Saved: {save_dir}/accuracy_comparison.png")
    plt.close(fig)


def plot_hessian_comparison(output_dir, save_dir):
    """Bar chart comparing Hessian sharpness metrics across conditions."""
    hessian_dir = os.path.join(output_dir, "hessian")
    if not os.path.exists(hessian_dir):
        print("No hessian results found. Skipping.")
        return

    results = {}
    for path in sorted(glob.glob(os.path.join(hessian_dir, "hessian_*.json"))):
        data = load_meta(path)
        results[data["tag"]] = data

    if not results:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    tags = list(results.keys())
    max_eigs = [results[t]["max_eigenvalue"] for t in tags]
    traces = [results[t]["trace"] for t in tags]

    colors = ['#FF9800' if 'ft' in t else '#4CAF50' if 'scr' in t else '#2196F3' for t in tags]

    ax1.bar(range(len(tags)), max_eigs, color=colors, edgecolor='white')
    ax1.set_xticks(range(len(tags)))
    ax1.set_xticklabels(tags, rotation=30, ha='right', fontsize=9)
    ax1.set_ylabel("Max Hessian Eigenvalue")
    ax1.set_title("Sharpness (λ_max) — Lower = Flatter")
    ax1.grid(axis='y', alpha=0.3)

    ax2.bar(range(len(tags)), traces, color=colors, edgecolor='white')
    ax2.set_xticks(range(len(tags)))
    ax2.set_xticklabels(tags, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel("Hessian Trace")
    ax2.set_title("Trace(H) — Lower = Flatter")
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle("Hessian Eigenspectrum Comparison", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "hessian_comparison.png"), dpi=150)
    print(f"Saved: {save_dir}/hessian_comparison.png")
    plt.close(fig)

    # Print table
    print("\n" + "="*60)
    print(f"{'Tag':<20} {'λ_max':>12} {'Trace':>12}")
    print("-"*60)
    for t in tags:
        r = results[t]
        print(f"{t:<20} {r['max_eigenvalue']:>12.4f} {r['trace']:>12.4f}")
    print("="*60)


def plot_all_surfaces(output_dir, save_dir):
    """Side-by-side comparison of loss surfaces."""
    landscape_dir = os.path.join(output_dir, "landscape")
    if not os.path.exists(landscape_dir):
        print("No landscape results found. Skipping.")
        return

    npz_files = sorted(glob.glob(os.path.join(landscape_dir, "surface_*.npz")))
    if not npz_files:
        return

    n = len(npz_files)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, npz_path in zip(axes, npz_files):
        data = np.load(npz_path)
        alphas, betas, losses = data['alphas'], data['betas'], data['losses']
        A, B = np.meshgrid(alphas, betas, indexing='ij')

        tag = os.path.basename(npz_path).replace("surface_", "").replace(".npz", "")
        vmin, vmax = losses.min(), np.percentile(losses, 90)
        levels = np.linspace(vmin, vmax, 25)

        cs = ax.contourf(A, B, losses, levels=levels, cmap='viridis')
        ax.contour(A, B, losses, levels=levels, colors='white', linewidths=0.3, alpha=0.4)
        ax.plot(0, 0, 'r*', markersize=12)
        ax.set_title(tag, fontsize=11)
        ax.set_xlabel('Dir 1')
        ax.set_ylabel('Dir 2')
        fig.colorbar(cs, ax=ax, shrink=0.8)

    fig.suptitle("Loss Landscape Comparison", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "surfaces_comparison.png"), dpi=150)
    print(f"Saved: {save_dir}/surfaces_comparison.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./outputs")
    args = parser.parse_args()

    save_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(save_dir, exist_ok=True)

    plot_accuracy_comparison(args.output_dir, save_dir)
    plot_hessian_comparison(args.output_dir, save_dir)
    plot_all_surfaces(args.output_dir, save_dir)

    print(f"\nAll figures saved to: {save_dir}/")


if __name__ == "__main__":
    main()
