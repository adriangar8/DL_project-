from __future__ import print_function
import os
import argparse
import torch

import models
from pruners.synflow import synflow_scores
from pruners.utils import apply_mask_, prunable_named_parameters

def global_topk_mask(score_dict: dict, sparsity: float):
    flats = [s.reshape(-1) for s in score_dict.values()]
    all_scores = torch.cat(flats)
    total = all_scores.numel()
    k_keep = max(1, int(total * (1.0 - sparsity)))
    kth = total - k_keep + 1
    threshold, _ = torch.kthvalue(all_scores, kth)

    mask_dict = {}
    for name, s in score_dict.items():
        mask_dict[name] = (s >= threshold).to(dtype=s.dtype)
    return mask_dict

def load_checkpoint_into_model(model, resume_path, device):
    ckpt = torch.load(resume_path, map_location=device)
    sd = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    model.load_state_dict(sd, strict=True)

def main():
    p = argparse.ArgumentParser("SynFlow one-shot pruning (data-free) for CIFAR")
    p.add_argument('--dataset', type=str, default='cifar10')  # csak hogy passzoljon a többihez
    p.add_argument('--arch', type=str, default='resnet')
    p.add_argument('--depth', type=int, default=56)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--no-cuda', action='store_true', default=False)

    p.add_argument('--sparsity', type=float, default=0.5)
    p.add_argument('--resume', type=str, default='', help="optional checkpoint")
    p.add_argument('--save', type=str, default='./logs/synflow_run')

    args = p.parse_args()
    cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.save, exist_ok=True)

    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth).to(device)
    model.train()

    if args.resume:
        print(f"=> loading weights from {args.resume}")
        load_checkpoint_into_model(model, args.resume, device)

    # SynFlow score data nélkül
    score_dict = synflow_scores(model, input_shape=(1, 3, 32, 32), device=device)
    mask_dict = global_topk_mask(score_dict, sparsity=args.sparsity)

    apply_mask_(model, mask_dict)

    torch.save(mask_dict, os.path.join(args.save, 'mask.pth'))
    torch.save({
        'epoch': 0,
        'state_dict': model.state_dict(),
        'best_prec1': 0.0,
        'optimizer': None,
    }, os.path.join(args.save, 'masked_init_checkpoint.pth.tar'))

    # extra info: keep ratio
    kept = sum(v.sum().item() for v in mask_dict.values())
    total = sum(v.numel() for v in mask_dict.values())
    print(f"[OK] Saved to {args.save}. Kept {kept}/{total} ({100*kept/total:.2f}%).")

if __name__ == "__main__":
    main()