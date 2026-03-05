"""CIFAR-100 data loading with DeiT-compatible transforms."""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(is_train=True, img_size=224):
    if is_train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def get_dataloaders(data_dir="./data", batch_size=128, num_workers=4, img_size=224):
    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True,
        transform=get_transforms(is_train=True, img_size=img_size),
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True,
        transform=get_transforms(is_train=False, img_size=img_size),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


def get_landscape_loader(data_dir="./data", batch_size=128, num_samples=2000, img_size=224):
    """Smaller loader for landscape analysis (faster evaluation)."""
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True,
        transform=get_transforms(is_train=False, img_size=img_size),
    )
    if num_samples < len(test_dataset):
        indices = torch.randperm(len(test_dataset))[:num_samples].tolist()
        test_dataset = Subset(test_dataset, indices)

    return DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
