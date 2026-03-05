#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: LANDSCAPE ANALYSIS (Days 3-4)
# Run after all training is complete.
# ═══════════════════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")/.."

DATA_DIR="./data"
OUT_DIR="./outputs"

echo "============================================"
echo " PHASE 2A: Hessian Eigenspectrum"
echo "============================================"

# Hessian for each condition at 50% pruning
for TAG in unpruned_ft p50_ft p50_scrE p50_scrB; do
    if [ "$TAG" = "unpruned_ft" ]; then
        PRUNE_FLAG=""
        MODE_FLAG="--mode unpruned"
    elif [[ "$TAG" == *_ft ]]; then
        PRUNE_FLAG="--prune prune50"
        MODE_FLAG="--mode finetuned"
    else
        PRUNE_FLAG="--prune prune50"
        MODE_FLAG="--mode scratch"
    fi

    echo "Hessian: $TAG"
    python src/analyze_hessian.py \
        --checkpoint $OUT_DIR/$TAG/best.pth \
        $PRUNE_FLAG $MODE_FLAG \
        --tag $TAG --top_k 3 \
        --data_dir $DATA_DIR --output_dir $OUT_DIR/hessian
done

echo ""
echo "============================================"
echo " PHASE 2B: Mode Connectivity"
echo "============================================"

# Fine-tuned vs Scratch-B (50% pruning)
echo "Connectivity: p50 fine-tuned ↔ scratch-B"
python src/analyze_connectivity.py \
    --checkpoint_a $OUT_DIR/p50_ft/best.pth \
    --checkpoint_b $OUT_DIR/p50_scrB/best.pth \
    --label_a "Fine-tuned" --label_b "Scratch-B" \
    --prune prune50 --n_points 21 \
    --data_dir $DATA_DIR --output_dir $OUT_DIR/connectivity \
    --tag p50_ft_vs_scrB

# Fine-tuned vs Scratch-E (50% pruning)
echo "Connectivity: p50 fine-tuned ↔ scratch-E"
python src/analyze_connectivity.py \
    --checkpoint_a $OUT_DIR/p50_ft/best.pth \
    --checkpoint_b $OUT_DIR/p50_scrE/best.pth \
    --label_a "Fine-tuned" --label_b "Scratch-E" \
    --prune prune50 --n_points 21 \
    --data_dir $DATA_DIR --output_dir $OUT_DIR/connectivity \
    --tag p50_ft_vs_scrE

# Scratch-E vs Scratch-B (50% pruning, same arch, both random init)
echo "Connectivity: p50 scratch-E ↔ scratch-B"
python src/analyze_connectivity.py \
    --checkpoint_a $OUT_DIR/p50_scrE/best.pth \
    --checkpoint_b $OUT_DIR/p50_scrB/best.pth \
    --label_a "Scratch-E" --label_b "Scratch-B" \
    --prune prune50 --n_points 21 \
    --data_dir $DATA_DIR --output_dir $OUT_DIR/connectivity \
    --tag p50_scrE_vs_scrB

# 30% pruning: fine-tuned vs scratch-B
echo "Connectivity: p30 fine-tuned ↔ scratch-B"
python src/analyze_connectivity.py \
    --checkpoint_a $OUT_DIR/p30_ft/best.pth \
    --checkpoint_b $OUT_DIR/p30_scrB/best.pth \
    --label_a "Fine-tuned" --label_b "Scratch-B" \
    --prune prune30 --n_points 21 \
    --data_dir $DATA_DIR --output_dir $OUT_DIR/connectivity \
    --tag p30_ft_vs_scrB

echo ""
echo "============================================"
echo " PHASE 2C: Loss Surface Visualization"
echo "============================================"

# Surface for each key condition (50% pruning)
for TAG in unpruned_ft p50_ft p50_scrB; do
    if [ "$TAG" = "unpruned_ft" ]; then
        PRUNE_FLAG=""
        MODE_FLAG="--mode unpruned"
    elif [[ "$TAG" == *_ft ]]; then
        PRUNE_FLAG="--prune prune50"
        MODE_FLAG="--mode finetuned"
    else
        PRUNE_FLAG="--prune prune50"
        MODE_FLAG="--mode scratch"
    fi

    echo "Surface: $TAG"
    python src/analyze_surface.py \
        --checkpoint $OUT_DIR/$TAG/best.pth \
        $PRUNE_FLAG $MODE_FLAG \
        --resolution 41 --distance 1.0 --num_samples 2000 \
        --data_dir $DATA_DIR --output_dir $OUT_DIR/landscape \
        --tag $TAG
done

echo ""
echo "============================================"
echo " PHASE 2 COMPLETE"
echo "============================================"

# ── Generate summary figures ──
echo ""
echo "Generating summary figures..."
python src/plot_results.py --output_dir $OUT_DIR

echo ""
echo "============================================"
echo " ALL DONE. Check outputs/figures/"
echo "============================================"
