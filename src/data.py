from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10


DATA_DIR = Path("data")
SPLIT_DIR = Path("splits")


def load_split_indices() -> tuple[np.ndarray, np.ndarray]:
    #Load the fixed indices created during dataset preparation
    train_indices = np.load(
        SPLIT_DIR / "train_indices.npy",
        allow_pickle=False,
    )
    val_indices = np.load(
        SPLIT_DIR / "val_indices.npy",
        allow_pickle=False,
    )

    #Check the expected subset sizes
    if len(train_indices) != 45000 or len(val_indices) != 5000:
        raise ValueError("Unexpected training or validation size")

    #Verify that all original training images appear exactly once
    combined_indices = np.concatenate(
        (train_indices, val_indices)
    )

    if not np.array_equal(
        np.sort(combined_indices),
        np.arange(50000),
    ):
        raise ValueError("Invalid or overlapping dataset indices")

    return train_indices, val_indices


def create_data_loaders(
    batch_size: int = 128,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    if batch_size < 1:
        raise ValueError("Batch size must be positive")

    #Use simple augmentation to improve generalization
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    #Keep validation images unchanged except for conversion to tensors
    val_transform = transforms.ToTensor()

    #Load both subsets from the original CIFAR-10 training partition
    train_dataset = CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=train_transform,
    )

    val_dataset = CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=val_transform,
    )

    train_indices, val_indices = load_split_indices()

    train_subset = Subset(
        train_dataset,
        train_indices.tolist(),
    )

    val_subset = Subset(
        val_dataset,
        val_indices.tolist(),
    )

    #Use a fixed generator to reproduce the training shuffle
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    #Use one loading process to keep memory usage predictable
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=train_generator,
    )

    #Keep validation order fixed and disable random augmentation
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, val_loader
