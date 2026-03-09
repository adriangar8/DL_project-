from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import models
from pruners.utils import enforce_mask_after_step_

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
        test_ds = datasets.CIFAR10('./data.cifar10', train=False, download=True,
            transform=transforms.Compose([
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
        test_ds = datasets.CIFAR100('./data.cifar100', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            ]))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader

def save_checkpoint(state, is_best, filepath):
    os.makedirs(filepath, exist_ok=True)
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'),
                        os.path.join(filepath, 'model_best.pth.tar'))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        out = model(data)
        loss_sum += F.cross_entropy(out, target, reduction='sum').item()
        pred = out.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.numel()
    return loss_sum / total, correct / total

def train_one_epoch(model, loader, optimizer, device, mask_dict, log_interval, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(data)
        loss = F.cross_entropy(out, target)
        loss.backward()
        optimizer.step()
        enforce_mask_after_step_(model, mask_dict)

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx*len(data)}/{len(loader.dataset)} "
                  f"({100.*batch_idx/len(loader):.1f}%)]\tLoss: {loss.item():.6f}")

def main():
    parser = argparse.ArgumentParser("Finetune under fixed SNIP mask")
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--arch', type=str, default='resnet')
    parser.add_argument('--depth', type=int, default=56)

    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--log-interval', type=int, default=100)

    parser.add_argument('--resume', type=str, required=True,
                        help="masked_init_checkpoint.pth.tar")
    parser.add_argument('--mask', type=str, required=True,
                        help="mask.pth")
    parser.add_argument('--save', type=str, default='./logs/snip_finetune')

    args = parser.parse_args()
    cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.save, exist_ok=True)
    train_loader, test_loader = build_loaders(args.dataset, args.batch_size, args.test_batch_size, cuda)

    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth).to(device)

    ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt['state_dict'], strict=True)

    mask_dict = torch.load(args.mask, map_location=device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    best_acc = 0.0
    history = []

    milestones = {int(args.epochs*0.5), int(args.epochs*0.75)}

    for epoch in range(args.epochs):
        if epoch in milestones:
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.1

        train_one_epoch(model, train_loader, optimizer, device, mask_dict, args.log_interval, epoch)
        test_loss, test_acc = evaluate(model, test_loader, device)

        print(f"\nTest: loss={test_loss:.4f} acc={test_acc*100:.2f}%\n")

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc

        history.append([epoch, test_loss, test_acc])
        np.savetxt(os.path.join(args.save, 'record.txt'),
                   np.asarray(history), fmt='%10.5f', delimiter=',')

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best=is_best, filepath=args.save)

    print("Best accuracy:", best_acc)

if __name__ == "__main__":
    main()