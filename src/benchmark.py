import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from src.model import CompactCNN


RANDOM_SEED = 42
BATCH_SIZE = 128
WARMUP_BATCHES = 5
MEASURED_BATCHES = 25
LEARNING_RATE = 0.001


def set_seed(seed: int) -> None:
    #Set random seeds for more reproducible results
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_train_loader() -> DataLoader:
    #Apply simple image augmentation during training
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    #Load the original training images without downloading again
    dataset = CIFAR10(
        root="data",
        train=True,
        download=False,
        transform=transform,
    )

    #Use only the saved training indices
    indices = np.load(
        "splits/train_indices.npy",
        allow_pickle=False,
    )

    train_subset = Subset(dataset, indices.tolist())

    #Keep data loading simple and memory usage manageable
    return DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )


def run_benchmark() -> None:
    set_seed(RANDOM_SEED)

    #Require MPS so the benchmark measures the intended hardware
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    device = torch.device("mps")

    train_loader = create_train_loader()
    model = CompactCNN().to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    #Count the trainable parameters in our CNN
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    #Verify the output dimensions before training
    sample_input = torch.zeros(2, 3, 32, 32, device=device)

    with torch.no_grad():
        sample_output = model(sample_input)

    if sample_output.shape != (2, 10):
        raise RuntimeError("Unexpected model output dimensions")

    print(f"Device: {device}", flush=True)
    print(f"Trainable parameters: {parameter_count:,}", flush=True)
    print(f"Training images: {len(train_loader.dataset)}", flush=True)
    print(f"Batch size: {BATCH_SIZE}", flush=True)
    print("Starting benchmark...", flush=True)

    total_images = 0
    start_time = None
    last_loss = None

    for step, (images, labels) in enumerate(train_loader):
        if step >= WARMUP_BATCHES + MEASURED_BATCHES:
            break

        #Exclude warmup batches from the timing measurement
        if step == WARMUP_BATCHES:
            torch.mps.synchronize()
            start_time = time.perf_counter()

        images = images.to(device)
        labels = labels.to(device)

        #Calculate the classification loss
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)

        #Compute parameter gradients and update the CNN
        loss.backward()
        optimizer.step()

        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite loss during benchmark")

        last_loss = loss.detach()

        if step >= WARMUP_BATCHES:
            total_images += images.size(0)

    if start_time is None or total_images == 0:
        raise RuntimeError("Benchmark did not process enough batches")

    #Wait for MPS operations to finish before stopping the timer
    torch.mps.synchronize()
    elapsed_seconds = time.perf_counter() - start_time

    images_per_second = total_images / elapsed_seconds
    seconds_per_epoch = (
        len(train_loader.dataset) / images_per_second
    )

    results = {
        "device": str(device),
        "random_seed": RANDOM_SEED,
        "model": "CompactCNN",
        "parameters": parameter_count,
        "training_images": len(train_loader.dataset),
        "batch_size": BATCH_SIZE,
        "warmup_batches": WARMUP_BATCHES,
        "measured_batches": MEASURED_BATCHES,
        "measured_images": total_images,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "images_per_second": round(images_per_second, 2),
        "estimated_epoch_seconds": round(seconds_per_epoch, 2),
        "last_training_loss": round(last_loss.item(), 4),
        "mps_allocated_memory_mb": round(
            torch.mps.current_allocated_memory() / 1024**2,
            2,
        ),
        "mps_driver_memory_mb": round(
            torch.mps.driver_allocated_memory() / 1024**2,
            2,
        ),
    }

    #Save the measurements for later planning and reporting
    output_path = Path("logs/04_training_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")

    print("\nBenchmark results")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run_benchmark()
