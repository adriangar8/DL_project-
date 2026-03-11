
# Vision Transformer Pruning Pipeline (DeiT on CIFAR-100)

This project implements a structured pruning pipeline for Vision Transformers (ViT) using a DeiT-style architecture trained on CIFAR-100.

The repository allows you to:

1. Train a dense ViT model from scratch
2. Generate structured pruning specifications
3. Apply pruning using two methods
   - Method A — SAViT-style Taylor pruning
   - Method B — X-Pruner-style mask learning
4. Compare two training strategies for pruned models
   - Fine-tuning from dense weights
   - Training the pruned architecture from scratch

The pipeline logs:
- Parameter count
- FLOPs
- Latency
- Throughput
- Top-1 / Top-5 accuracy
- Estimated training compute

All results are saved to CSV files for reproducibility.

---

# Project Structure

```
.
├── vit_pruning_spec.py

├── run_deit_cifar100_pipeline.py

├── runs/

│   ├── dense/

│   ├── savit_taylor/

│   └── xpruner_masks/

└── data/
```
---

# Step-by-Step Instructions

Follow these steps in order.

---

# Step 1 — Install Requirements

```
pip install torch torchvision timm fvcore tqdm
```
---

# Step 2 — Train the Dense Model From Scratch


```
python run_deit_cifar100_pipeline.py   --stage dense   --run_name dense_baseline   --data_dir ./data   --root_out ./runs   --model deit_small_patch16_224   --image_size 224   --num_classes 100   --batch_size 128   --num_workers 8   --epochs 200   --lr 5e-4   --weight_decay 0.05   --label_smoothing 0.1   --mixup 0.8   --cutmix 1.0   --mixup_prob 1.0   --mixup_switch_prob 0.5   --mixup_mode batch   --warmup_epochs 10   --min_lr 1e-6   --eval_every 1   --patience 20   --min_delta 0.05   --randaugment   --ra_n 2   --ra_m 9   --amp
```

Dense checkpoint:

```
runs/dense/dense_baseline/best.pt
```

---

# Step 3 — Generate Pruning Spec Using Method A (SAViT-Style Taylor)

```
python run_deit_cifar100_pipeline.py   --stage make_spec   --prune_method savit_taylor   --run_name methodA_spec   --dense_ckpt ./runs/dense/dense_baseline/best.pt   --data_dir ./data   --root_out ./runs   --model deit_small_patch16_224   --image_size 224   --num_classes 100   --batch_size 128   --num_workers 8   --depth_keep 10   --mlp_keep 0.75   --calib_batches 20   --randaugment   --ra_n 2   --ra_m 9   --amp
```
---

# Step 4 — Generate Pruning Spec Using Method B (X-Pruner Masks)

```
python run_deit_cifar100_pipeline.py   --stage make_spec   --prune_method xpruner_masks   --run_name methodB_spec   --dense_ckpt ./runs/dense/dense_baseline/best.pt   --data_dir ./data   --root_out ./runs   --model deit_small_patch16_224   --image_size 224   --num_classes 100   --batch_size 128   --num_workers 8   --depth_keep 10   --mlp_keep 0.75   --mask_train_epochs 2   --mask_lr 5e-3   --mask_steps_per_epoch 200   --randaugment   --ra_n 2   --ra_m 9   --amp
```

---

# Step 5 — Fine-Tune Method A Pruned Model

```
python run_deit_cifar100_pipeline.py   --stage finetune_pruned   --run_name methodA_finetune   --dense_ckpt ./runs/dense/dense_baseline/best.pt   --spec_path ./runs/savit_taylor/make_spec/methodA_spec/spec.json   --data_dir ./data   --root_out ./runs   --epochs 100   --randaugment   --ra_n 2   --ra_m 9   --amp
```
---

# Step 6 — Train Method A Pruned Architecture From Scratch
```
python run_deit_cifar100_pipeline.py   --stage scratch_pruned   --run_name methodA_scratch   --spec_path ./runs/savit_taylor/make_spec/methodA_spec/spec.json   --data_dir ./data   --root_out ./runs   --epochs 100   --randaugment   --ra_n 2   --ra_m 9   --amp
```
---

# Step 7 — Fine-Tune Method B Pruned Model

```
python run_deit_cifar100_pipeline.py   --stage finetune_pruned   --run_name methodB_finetune   --dense_ckpt ./runs/dense/dense_baseline/best.pt   --spec_path ./runs/xpruner_masks/make_spec/methodB_spec/spec.json   --data_dir ./data   --root_out ./runs   --epochs 100   --randaugment   --ra_n 2   --ra_m 9   --amp
```
---

# Step 8 — Train Method B Pruned Architecture From Scratch
```
python run_deit_cifar100_pipeline.py   --stage scratch_pruned   --run_name methodB_scratch   --spec_path ./runs/xpruner_masks/make_spec/methodB_spec/spec.json   --data_dir ./data   --root_out ./runs   --epochs 100   --randaugment   --ra_n 2   --ra_m 9   --amp
```
---

# Metrics Logged

Each experiment records:

- Parameters
- FLOPs
- Latency
- Throughput
- Top-1 accuracy
- Top-5 accuracy
- Training steps
- Total training compute
- Best validation epoch
- Early stopping status


