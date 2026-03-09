from __future__ import print_function
import os
import argparse
import torch
from torchvision import datasets, transforms

import models
from pruners.snip import snip_mask
from pruners.utils import apply_mask_

def build_loaders(dataset: str, batch_size: int, test_batch_size: int, cuda: bool):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    if dataset == 'cifar10':
        train_ds = datasets.CIFAR10('./data.cifar10', train=True, download=True,
            transform=transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            ]))
    else:
        train_ds = datasets.CIFAR100('./data.cifar100', train=True, download=True,
            transform=transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            ]))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader

def load_checkpoint_into_model(model, resume_path, cuda):
    ckpt = torch.load(resume_path, map_location='cuda' if cuda else 'cpu')
    sd = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    model.load_state_dict(sd, strict=True)

def main():
    parser = argparse.ArgumentParser("SNIP one-shot pruning (unstructured) for CIFAR")
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--arch', type=str, default='resnet')
    parser.add_argument('--depth', type=int, default=56)

    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no-cuda', action='store_true', default=False)

    parser.add_argument('--sparsity', type=float, default=0.9)
    parser.add_argument('--num-batches', type=int, default=1)
    parser.add_argument('--resume', type=str, default='',
                        help="optional baseline checkpoint (model_best.pth.tar)")
    parser.add_argument('--save', type=str, default='./logs/snip_run')

    args = parser.parse_args()
    cuda = (not args.no_cuda) and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.save, exist_ok=True)

    train_loader = build_loaders(args.dataset, args.batch_size, args.batch_size, cuda)

    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
    if cuda:
        model.cuda()

    if args.resume:
        print(f"=> loading weights from {args.resume}")
        load_checkpoint_into_model(model, args.resume, cuda)

    model.train()

    mask_dict = None
    for i, (data, target) in enumerate(train_loader):
        if i >= args.num_batches:
            break
        if cuda:
            data, target = data.cuda(), target.cuda()

        m = snip_mask(model, data, target, sparsity=args.sparsity)
        if mask_dict is None:
            mask_dict = {k: v.clone() for k, v in m.items()}
        else:
            for k in mask_dict:
                mask_dict[k] = torch.maximum(mask_dict[k], m[k])

    apply_mask_(model, mask_dict)

    torch.save(mask_dict, os.path.join(args.save, 'mask.pth'))
    torch.save({
        'epoch': 0,
        'state_dict': model.state_dict(),
        'best_prec1': 0.0,
        'optimizer': None,
    }, os.path.join(args.save, 'masked_init_checkpoint.pth.tar'))

    print(f"[OK] Saved: {args.save}/mask.pth and masked_init_checkpoint.pth.tar")

if __name__ == "__main__":
    main()