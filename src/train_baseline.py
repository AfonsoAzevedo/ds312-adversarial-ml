import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import create_data_loaders
from src.model import CompactCNN


RANDOM_SEED = 42
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")


def set_seed(seed: int) -> None:
    #Set the random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    #Enable training mode
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        #Move the images and labels to the selected device
        images = images.to(device)
        labels = labels.to(device)

        #Clear the gradients from the previous batch
        optimizer.zero_grad(set_to_none=True)

        #Compute the predictions and classification loss
        outputs = model(images)
        loss = criterion(outputs, labels)

        #Calculate gradients and update model weights
        loss.backward()
        optimizer.step()

        #Record loss and the number of correct predictions
        batch_size = labels.size(0)
        total_loss += loss.detach().item() * batch_size
        total_correct += (
            outputs.detach().argmax(dim=1) == labels
        ).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    if not np.isfinite(average_loss):
        raise RuntimeError("Non-finite training loss detected")

    return average_loss, accuracy


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    #Switch to evaluation mode for validation
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    #Disable gradient calculations during validation
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    if not np.isfinite(average_loss):
        raise RuntimeError("Non-finite validation loss detected")

    return average_loss, accuracy


def main() -> None:
    #Create directories for checkpoints and training logs
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    best_path = CHECKPOINT_DIR / "baseline_best.pt"
    last_path = CHECKPOINT_DIR / "baseline_last.pt"
    history_path = LOG_DIR / "baseline_history.csv"
    config_path = LOG_DIR / "baseline_config.json"

    #Avoid overwriting results from an earlier training run
    output_paths = [
        best_path,
        last_path,
        history_path,
        config_path,
    ]

    if any(path.exists() for path in output_paths):
        raise FileExistsError(
            "Baseline output files already exist"
        )

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS acceleration is unavailable")

    set_seed(RANDOM_SEED)
    device = torch.device("mps")

    #Load the saved training and validation subsets
    train_loader, val_loader = create_data_loaders(
        batch_size=BATCH_SIZE,
        seed=RANDOM_SEED,
    )

    #Create the same CNN architecture used in the benchmark
    model = CompactCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    #Decrease the learning rate gradually across epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    #Save the experiment settings for reproducibility
    config = {
        "experiment": "baseline",
        "dataset": "CIFAR-10",
        "model": "CompactCNN",
        "parameters": parameter_count,
        "seed": RANDOM_SEED,
        "device": str(device),
        "training_images": len(train_loader.dataset),
        "validation_images": len(val_loader.dataset),
        "test_set_used": False,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR",
        "loss_function": "CrossEntropyLoss",
        "training_augmentation": [
            "RandomCrop(32, padding=4)",
            "RandomHorizontalFlip()",
        ],
        "validation_transform": "ToTensor()",
        "pixel_range": [0.0, 1.0],
        "checkpoint_selection": "best validation accuracy",
        "torch_version": torch.__version__,
    }

    config_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_accuracy_pct",
        "val_loss",
        "val_accuracy_pct",
        "epoch_seconds",
        "cumulative_seconds",
    ]

    best_accuracy = -1.0
    best_epoch = 0
    total_seconds = 0.0

    print("Starting baseline CNN training", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Parameters: {parameter_count:,}", flush=True)
    print(f"Training images: {len(train_loader.dataset)}", flush=True)
    print(f"Validation images: {len(val_loader.dataset)}", flush=True)
    print(f"Epochs: {EPOCHS}", flush=True)

    #Save training metrics after each completed epoch
    with history_path.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for epoch in range(1, EPOCHS + 1):
            torch.mps.synchronize()
            start_time = time.perf_counter()

            current_lr = optimizer.param_groups[0]["lr"]

            #Train the CNN using the training subset
            train_loss, train_accuracy = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )

            #Measure clean accuracy on the validation subset
            val_loss, val_accuracy = validate(
                model,
                val_loader,
                criterion,
                device,
            )

            torch.mps.synchronize()
            epoch_seconds = time.perf_counter() - start_time
            total_seconds += epoch_seconds

            scheduler.step()

            #Record the metrics for this epoch
            row = {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss": round(train_loss, 6),
                "train_accuracy_pct": round(train_accuracy, 4),
                "val_loss": round(val_loss, 6),
                "val_accuracy_pct": round(val_accuracy, 4),
                "epoch_seconds": round(epoch_seconds, 3),
                "cumulative_seconds": round(total_seconds, 3),
            }

            writer.writerow(row)
            csv_file.flush()

            #Save the best model based on validation accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_epoch = epoch

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_accuracy_pct": val_accuracy,
                        "config": config,
                    },
                    best_path,
                )

            #Save the latest model and optimizer state
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_accuracy_pct": val_accuracy,
                    "config": config,
                },
                last_path,
            )

            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.2f}% | "
                f"Time: {epoch_seconds:.2f}s",
                flush=True,
            )

    print("\nBaseline training completed", flush=True)
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Best epoch: {best_epoch}")
    print(f"Total training and validation time: {total_seconds:.2f}s")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Training history: {history_path}")
    print(f"Configuration: {config_path}")


if __name__ == "__main__":
    main()
