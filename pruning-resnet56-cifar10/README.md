# Comparison of Neural Network Pruning Methods on ResNet-56 (CIFAR-10)

This project compares several neural network pruning methods applied to the **ResNet-56** convolutional neural network on the **CIFAR-10** image classification dataset.

The goal is to study how different pruning strategies reduce model complexity while preserving classification accuracy.

The experiments evaluate both **structured pruning methods** and **pruning-at-initialization techniques**, following the standard pruning pipeline:

1. Train a baseline model  
2. Apply pruning  
3. Fine-tune the pruned network  
4. Evaluate performance on the test set

The work builds on the ideas presented in Liu et al. and related pruning literature.

---

# Dataset

Experiments are performed on the **CIFAR-10** dataset.

CIFAR-10 contains:

- 60,000 color images  
- image size: 32×32  
- 10 object classes  
- 50,000 training images  
- 10,000 test images  

The dataset is automatically downloaded during training, therefore it is not included in this repository.

---

# Model

All experiments use the **ResNet-56 architecture**, a residual convolutional neural network designed for CIFAR-10 classification.

Residual connections allow deeper networks to be trained efficiently while maintaining stable gradient flow.

---

# Pruning Methods

The repository includes implementations of the following pruning methods.

## L1-Norm Pruning

A structured pruning method where the importance of a convolutional filter is estimated using the L1 norm of its weights:

\[
\|W\|_1 = \sum_i |w_i|
\]

Filters with the smallest norms are removed, and the resulting network is fine-tuned.

---

## Network Slimming

Network Slimming uses the scaling parameters of **Batch Normalization layers** to determine channel importance.

During training, L1 regularization is applied to the BN scaling parameters:

\[
L_{total} = L_{task} + \lambda \sum |\gamma|
\]

Channels with small scaling values are pruned after training.

---

## SNIP

SNIP (Single-shot Network Pruning) performs pruning **before training begins**.

The importance of each parameter is computed using the gradient of the loss with respect to that parameter:

\[
S_i = \left| w_i \frac{\partial L}{\partial w_i} \right|
\]

Weights with the smallest scores are removed, producing a sparse network that is then trained normally.

---

## GraSP

GraSP (Gradient Signal Preservation) extends SNIP by considering the **effect of pruning on gradient flow**.

Instead of only measuring loss sensitivity, GraSP selects parameters whose removal minimally disrupts gradient propagation during training.

---

## SynFlow

SynFlow is a **data-free pruning method** that avoids layer collapse during pruning.

It computes parameter importance by propagating a synthetic signal through the network using an input tensor of ones.

---

# Repository Structure

```
pruning-resnet56-cifar10
│
├── l1-norm-pruning/
├── network-slimming/
├── snip/
├── grasp/
├── synflow/
│
├── README.md
└── .gitignore
```

Each folder contains the code required to run the corresponding pruning method.

---

# Running Experiments

## Train the baseline model

```bash
python main.py --dataset cifar10 --arch resnet --depth 56
```

## L1-norm pruning

```bash
python res56prune.py
```

## Network Slimming

```bash
python main_B.py
```

## SNIP

```bash
python one_shot_prune.py
```

## GraSP

```bash
python one_shot_prune.py
```

## SynFlow

```bash
python one_shot_prune.py
```

---

# Results

Baseline ResNet-56 accuracy on CIFAR-10:

**94.12%**

Accuracy after pruning 50% of parameters:

| Method | Accuracy |
|------|------|
| L1-Norm | 92.79 |
| Network Slimming | 93.47 |
| SNIP | 92.97 |
| GraSP | 93.04 |
| SynFlow | 91.64 |

These results show that moderate pruning can significantly reduce model size while maintaining most of the original performance.

---

# Notes

Dataset files, training logs and model checkpoints are excluded from the repository to keep it lightweight.
