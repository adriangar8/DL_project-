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
