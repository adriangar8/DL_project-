#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: TRAINING (Days 1-2)
# Launch all training runs. Run overnight if needed.
# ═══════════════════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")/.."

DATA_DIR="./data"
OUT_DIR="./outputs"

echo "============================================"
echo " PHASE 1: Training all models"
echo "============================================"

# ─── 1. Unpruned baseline (pretrained DeiT-Small, fine-tuned on CIFAR-100) ───
echo "[1/7] Fine-tuning unpruned baseline..."
python src/train.py \
    --mode finetune_unpruned \
    --epochs 40 --lr 1e-4 --batch_size 128 \
    --data_dir $DATA_DIR --output_dir $OUT_DIR \
    --tag unpruned_ft

# ─── 2. Pruned 50% → Fine-tune (inherits pretrained weights) ───
echo "[2/7] Fine-tuning pruned-50% model..."
python src/train.py \
    --mode finetune_pruned --prune prune50 \
    --epochs 40 --lr 1e-4 --batch_size 128 \
    --data_dir $DATA_DIR --output_dir $OUT_DIR \
    --tag p50_ft

# ─── 3. Pruned 50% → Scratch-E (same epochs as fine-tune) ───
echo "[3/7] Training pruned-50% from scratch (Scratch-E)..."
python src/train.py \
    --mode scratch --prune prune50 \
    --epochs 40 --lr 5e-4 --batch_size 128 \
    --data_dir $DATA_DIR --output_dir $OUT_DIR \
    --tag p50_scrE

# ─── 4. Pruned 50% → Scratch-B (more epochs = more compute budget) ───
echo "[4/7] Training pruned-50% from scratch (Scratch-B)..."
python src/train.py \
    --mode scratch --prune prune50 \
    --epochs 100 --lr 5e-4 --batch_size 128 \
    --data_dir $DATA_DIR --output_dir $OUT_DIR \
    --tag p50_scrB

# ─── 5. Pruned 30% → Fine-tune ───
echo "[5/7] Fine-tuning pruned-30% model..."
python src/train.py \
    --mode finetune_pruned --prune prune30 \
    --epochs 40 --lr 1e-4 --batch_size 128 \
    --data_dir $DATA_DIR --output_dir $OUT_DIR \
    --tag p30_ft

# ─── 6. Pruned 30% → Scratch-E ───
echo "[6/7] Training pruned-30% from scratch (Scratch-E)..."
python src/train.py \
    --mode scratch --prune prune30 \
    --epochs 40 --lr 5e-4 --batch_size 128 \
    --data_dir $DATA_DIR --output_dir $OUT_DIR \
    --tag p30_scrE

# ─── 7. Pruned 30% → Scratch-B ───
echo "[7/7] Training pruned-30% from scratch (Scratch-B)..."
python src/train.py \
    --mode scratch --prune prune30 \
    --epochs 100 --lr 5e-4 --batch_size 128 \
    --data_dir $DATA_DIR --output_dir $OUT_DIR \
    --tag p30_scrB

echo ""
echo "============================================"
echo " PHASE 1 COMPLETE — All models trained."
echo "============================================"
