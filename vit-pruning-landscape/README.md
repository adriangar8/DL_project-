# Loss Landscape Geometry of Pruned Vision Transformers

Investigating whether Liu et al.'s (2019) "Rethinking the Value of Network Pruning" findings
generalize to Vision Transformers, and explaining the results through loss landscape analysis.

## Quick Start

```bash
# 1. Setup environment
conda env create -f environment.yml
conda activate vit-prune

# 2. Download CIFAR-100 (automatic on first run)
#    No manual dataset setup needed.

# 3. Train all models (~6 hours total on RTX 4000 Ada)
bash scripts/01_train_all.sh

# 4. Run landscape analysis (~3-4 hours)
bash scripts/02_analyze_all.sh

# 5. Results are in outputs/figures/
```

## Project Structure

```
├── environment.yml          # Conda environment
├── src/
│   ├── data.py              # CIFAR-100 data loading
│   ├── models.py            # Pruned DeiT creation + weight transfer
│   ├── train.py             # Training / fine-tuning script
│   ├── analyze_surface.py   # 2D loss surface visualization
│   ├── analyze_hessian.py   # Hessian eigenspectrum
│   ├── analyze_connectivity.py  # Linear mode connectivity
│   └── plot_results.py      # Summary figures
├── scripts/
│   ├── 01_train_all.sh      # All training commands
│   └── 02_analyze_all.sh    # All analysis commands
└── outputs/                 # Results (created automatically)
    ├── unpruned_ft/         # Unpruned baseline
    ├── p50_ft/              # 50% pruned, fine-tuned
    ├── p50_scrE/            # 50% pruned, scratch (same epochs)
    ├── p50_scrB/            # 50% pruned, scratch (more epochs)
    ├── p30_ft/              # 30% pruned, fine-tuned
    ├── p30_scrE/            # 30% pruned, scratch-E
    ├── p30_scrB/            # 30% pruned, scratch-B
    ├── hessian/             # Hessian eigenvalue results
    ├── landscape/           # 2D loss surfaces
    ├── connectivity/        # Mode connectivity plots
    └── figures/             # Summary comparison figures
```

## What We Do

**Pruning:** Remove attention heads + reduce MLP hidden dimension in DeiT-Small.
- `prune30`: keep 4/6 heads, MLP 1024/1536 (~30% param reduction in attn+MLP)
- `prune50`: keep 3/6 heads, MLP 768/1536 (~50% param reduction in attn+MLP)

**Three conditions per pruning level (following Liu et al.):**
- **Fine-tuned:** Pretrained → prune (transfer best weights) → fine-tune 20 epochs
- **Scratch-E:** Pruned architecture → random init → 20 epochs (same as fine-tune)
- **Scratch-B:** Pruned architecture → random init → 50 epochs (more compute budget)

**Landscape analysis:**
- 2D loss surface visualization (Li et al. 2018)
- Hessian top eigenvalues (sharpness metric)
- Linear mode connectivity between fine-tuned and scratch models

## Results
**Accuracy: Fine-tuning vs. Scratch**

Table 4.2. Head+MLP pruned DeiT-Small on CIFAR-100. The 25–29 point gap directly contradicts Liu et al.’s findings for
CNNs.
| Model            | Params | Fine-tuned | Scratch-E | Scratch-B |
|------------------|-------:|-----------:|----------:|----------:|
| Unpruned DeiT-S  | 21.7M  | 88.53%     | ---       | ---       |
| Prune-30%        | 14.6M  | **83.60%** | 54.20%    | 55.04%    |
| Prune-50%        | 11.1M  | **79.75%** | 54.16%    | 54.74%    |

**Hessian Eigenspectrum**

Table 4.3. Hessian analysis. The fine-tuned model occupies the sharpest minimum (𝜆max = 1444) yet achieves the best accuracy
among pruned models.
| Model                      | λ_max | Top-3 eigenvalues   | Trace | Test Acc |
|----------------------------|------:|---------------------|------:|---------:|
| Unpruned (fine-tuned)      | 1085  | 1085, 1027, 706     | 228   | 88.53%   |
| Prune-50% (fine-tuned)     | **1444** | 1444, 1253, 1182 | 4692  | 79.75%   |
| Prune-50% (Scratch-E)      | 268   | 268, 244, 240       | 2805  | 54.16%   |
| Prune-50% (Scratch-B)      | 298   | 298, 210, 205       | 1753  | 54.74%   |

**Mode Connectivity**

Table 4.4. Linear mode connectivity. Fine-tuned and scratch models occupy completely disconnected basins; both scratch
models lie in the same basin.
| Pair                         | Barrier | Min acc. along path | Same basin? |
|------------------------------|---------|---------------------|-------------|
| p50: FT ↔ Scratch-B          | 2.54    | 1.35%               | No          |
| p50: FT ↔ Scratch-E          | 3.01    | 1.25%               | No          |
| p30: FT ↔ Scratch-B          | 2.32    | 1.05%               | No          |
| p50: Scratch-E ↔ Scratch-B   | **0.00**| 53.50%              | **Yes**     |

Figure 4.1. Fine-tuned vs. Scratch-B—accuracy collapses to ∼1% at the midpoint, confirming disconnected basins. 
<img width="1783" height="667" alt="image" src="https://github.com/user-attachments/assets/35f1bdc8-158f-4f5f-9621-4c96f9a16441" />

Figure 4.1. Scratch-E vs. Scratch-B—monotonically connected with zero barrier.
<img width="1784" height="667" alt="image" src="https://github.com/user-attachments/assets/f83c2f62-992a-43db-9317-4e26b453ef29" />

**2D Loss Surfaces**

Figure 4.2. Left: Pruned fine-tuned with sharp, asymmetric basin, min. loss 0.98. Center: Pruned scratch with smooth,
symmetric bowl, min. loss 2.94. Right: Unpruned fine-tuned with deep, irregular basin. The visual smoothness of the scratch
landscape is deceptive.
<img width="2250" height="675" alt="image" src="https://github.com/user-attachments/assets/0a94269f-fc3a-40ca-8401-523239e6fe9e" />

## Running Individual Experiments

```bash
# Train a single model
python src/train.py --mode scratch --prune prune50 --epochs 50 --lr 5e-4 --tag my_experiment

# Analyze a single checkpoint
python src/analyze_hessian.py --checkpoint outputs/p50_ft/best.pth --prune prune50 --mode finetuned --tag p50_ft
python src/analyze_surface.py --checkpoint outputs/p50_ft/best.pth --prune prune50 --mode finetuned --tag p50_ft
python src/analyze_connectivity.py --checkpoint_a outputs/p50_ft/best.pth --checkpoint_b outputs/p50_scrB/best.pth --prune prune50 --tag test
```

## Hardware Requirements

- GPU: ~8 GB VRAM minimum (tested on RTX 4000 Ada 20GB)
- Training time: ~6 hours for all 7 runs
- Analysis time: ~3-4 hours for Hessian + connectivity + surfaces
- Disk: ~2 GB for CIFAR-100 + model checkpoints
