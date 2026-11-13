import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.attacks import pgd_attack
from src.data import create_data_loaders
from src.model import CompactCNN


SEED = 42
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
TRAIN_EPSILON = 8 / 255
TRAIN_STEP_SIZE = 10 / 255
MONITOR_EPSILON = 8 / 255
MONITOR_STEP_SIZE = 2 / 255
MONITOR_PGD_STEPS = 10
MONITOR_PER_CLASS = 50
CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")
SPLIT_DIR = Path("splits")


def set_seed(seed: int) -> None:
    #Seed Python, NumPy, and PyTorch before creating the model
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_monitor_loader(val_loader: DataLoader) -> DataLoader:
    #Choose fixed validation examples without using the test set
    val_subset = val_loader.dataset
    original_targets = np.asarray(val_subset.dataset.targets)
    original_indices = np.asarray(val_subset.indices)
    labels = original_targets[original_indices]
    rng = np.random.default_rng(SEED)

    chosen = []
    for class_id in range(10):
        class_positions = np.flatnonzero(labels == class_id)
        if len(class_positions) < MONITOR_PER_CLASS:
            raise RuntimeError("Not enough validation examples per class")
        selected = rng.choice(
            class_positions, size=MONITOR_PER_CLASS, replace=False
        )
        chosen.append(selected)

    positions = np.sort(np.concatenate(chosen)).astype(np.int64)
    monitor_path = SPLIT_DIR / "adv_monitor_indices.npy"

    #Keep the same monitoring images if the script runs again
    if monitor_path.exists():
        saved = np.load(monitor_path, allow_pickle=False)
        if not np.array_equal(saved, positions):
            raise RuntimeError("Existing monitoring indices do not match")
    else:
        np.save(monitor_path, positions)

    return DataLoader(
        Subset(val_subset, positions.tolist()),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


def make_training_examples(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    #Use a random start followed by one FGSM gradient-sign step
    #Project back to the allowed pixel neighborhood after the step
    return pgd_attack(
        model=model,
        images=images,
        labels=labels,
        epsilon=TRAIN_EPSILON,
        step_size=TRAIN_STEP_SIZE,
        steps=1,
        random_start=True,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    #Train on adversarial images instead of clean images
    model.train()
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device, dtype=torch.long)
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        #The attack uses input gradients without updating model weights
        adversarial_images = make_training_examples(model, images, labels)

        #Update the model once using the adversarial batch
        logits = model(adversarial_images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite adversarial training loss")
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            total_loss += loss.detach() * labels.size(0)
            total_correct += (logits.detach().argmax(1) == labels).sum()
        total_samples += labels.size(0)

    return (
        total_loss.item() / total_samples,
        100.0 * total_correct.item() / total_samples,
    )


def evaluate_clean(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    #Measure clean validation performance without gradients
    model.eval()
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device, dtype=torch.long)
    total_samples = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum()
            total_samples += labels.size(0)

    return (
        total_loss.item() / total_samples,
        100.0 * total_correct.item() / total_samples,
    )


def evaluate_pgd_monitor(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    #Use a fixed validation subset to detect weak PGD robustness
    #This diagnostic does not replace Chat 03's final evaluation
    model.eval()
    total_correct = torch.zeros((), device=device, dtype=torch.long)
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        #Generate PGD outside inference mode because it needs gradients
        adversarial_images = pgd_attack(
            model=model,
            images=images,
            labels=labels,
            epsilon=MONITOR_EPSILON,
            step_size=MONITOR_STEP_SIZE,
            steps=MONITOR_PGD_STEPS,
            random_start=False,
        )

        with torch.inference_mode():
            predictions = model(adversarial_images).argmax(1)
            total_correct += (predictions == labels).sum()
        total_samples += labels.size(0)

    return 100.0 * total_correct.item() / total_samples


def smoke_test(device: torch.device) -> None:
    #Test one actual training batch without writing checkpoints
    set_seed(SEED)
    train_loader, _ = create_data_loaders(BATCH_SIZE, SEED)
    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)
    model = CompactCNN().to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    #Snapshot parameters and BatchNorm statistics before the attack
    original_state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    optimizer.zero_grad(set_to_none=True)
    original_images = images.detach().clone()
    adversarial_images = make_training_examples(model, images, labels)

    if not torch.equal(images, original_images):
        raise RuntimeError("Attack modified the original images")
    if not model.training:
        raise RuntimeError("Attack did not restore training mode")
    if adversarial_images.requires_grad:
        raise RuntimeError("Attack returned an attached gradient graph")
    if not torch.isfinite(adversarial_images).all().item():
        raise RuntimeError("Attack produced non-finite images")
    if adversarial_images.min().item() < 0.0:
        raise RuntimeError("Attack produced negative pixel values")
    if adversarial_images.max().item() > 1.0:
        raise RuntimeError("Attack produced pixel values above one")

    max_change = (adversarial_images - images).abs().max().item()
    if max_change > TRAIN_EPSILON + 1e-6:
        raise RuntimeError("Attack exceeded the epsilon budget")
    if max_change <= 0.0:
        raise RuntimeError("Attack did not change any images")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Attack populated model parameter gradients")

    for name, original in original_state.items():
        if not torch.equal(model.state_dict()[name], original):
            raise RuntimeError(f"Attack modified model state: {name}")

    #A fixed seed should reproduce the random-start attack
    torch.manual_seed(123)
    first_attack = make_training_examples(model, images, labels)
    torch.manual_seed(123)
    second_attack = make_training_examples(model, images, labels)
    if not torch.allclose(first_attack, second_attack, atol=1e-6, rtol=0):
        raise RuntimeError("Random-start attack was not reproducible")

    #Confirm that a separate training pass really updates model weights
    logits = model(adversarial_images)
    loss = nn.functional.cross_entropy(logits, labels)
    if not torch.isfinite(loss).item():
        raise RuntimeError("Training loss was not finite")
    loss.backward()
    optimizer.step()

    updated = any(
        not torch.equal(parameter.detach(), original_state[name])
        for name, parameter in model.named_parameters()
    )
    if not updated:
        raise RuntimeError("Optimizer did not update the model weights")

    result = {
        "verification": "PASSED",
        "device": str(device),
        "training_batch_size": int(labels.size(0)),
        "training_epsilon": TRAIN_EPSILON,
        "training_step_size": TRAIN_STEP_SIZE,
        "max_abs_pixel_change": max_change,
        "training_loss": float(loss.item()),
        "input_constraints_passed": True,
        "random_start_seed_reproducible": True,
        "model_state_unchanged_during_attack": True,
        "parameter_gradients_unchanged_during_attack": True,
        "optimizer_updated_weights": True,
        "checkpoints_written": False,
        "test_set_used": False,
    }
    print(json.dumps(result, indent=2), flush=True)


def train(device: torch.device) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    best_path = CHECKPOINT_DIR / "adversarial_best.pt"
    last_path = CHECKPOINT_DIR / "adversarial_last.pt"
    history_path = LOG_DIR / "adversarial_history.csv"
    config_path = LOG_DIR / "adversarial_config.json"

    #Avoid silently overwriting earlier experiments
    if any(
        path.exists()
        for path in (best_path, last_path, history_path, config_path)
    ):
        raise FileExistsError("Adversarial training outputs already exist")

    set_seed(SEED)
    train_loader, val_loader = create_data_loaders(BATCH_SIZE, SEED)
    monitor_loader = make_monitor_loader(val_loader)
    model = CompactCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    if parameter_count != 288746:
        raise RuntimeError("Model differs from the locked baseline architecture")

    config = {
        "experiment": "fgsm_random_start_adversarial_training",
        "dataset": "CIFAR-10",
        "model": "CompactCNN",
        "trainable_parameters": parameter_count,
        "seed": SEED,
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
        "training_attack": "random_start_FGSM_one_step_projected",
        "training_epsilon": TRAIN_EPSILON,
        "training_step_size": TRAIN_STEP_SIZE,
        "training_attack_steps": 1,
        "training_random_start": True,
        "monitor_source": "fixed stratified validation subset",
        "monitor_images": len(monitor_loader.dataset),
        "monitor_images_per_class": MONITOR_PER_CLASS,
        "monitor_epsilon": MONITOR_EPSILON,
        "monitor_step_size": MONITOR_STEP_SIZE,
        "monitor_pgd_steps": MONITOR_PGD_STEPS,
        "monitor_random_start": False,
        "checkpoint_selection": "highest validation monitor PGD accuracy, then clean validation accuracy",
        "torch_version": str(torch.__version__),
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    fieldnames = [
        "epoch", "learning_rate", "adv_train_loss", "adv_train_accuracy_pct",
        "clean_val_loss", "clean_val_accuracy_pct",
        "pgd10_monitor_accuracy_pct", "epoch_seconds", "cumulative_seconds",
    ]
    best_score = (-1.0, -1.0)
    best_epoch = 0
    elapsed_total = 0.0

    print("Starting random-start FGSM adversarial training", flush=True)
    print(f"Device: {device} | Parameters: {parameter_count:,}", flush=True)
    print(
        f"Train: {len(train_loader.dataset)} | "
        f"Validation: {len(val_loader.dataset)} | "
        f"PGD monitor: {len(monitor_loader.dataset)}",
        flush=True,
    )

    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, EPOCHS + 1):
            torch.mps.synchronize()
            start = time.perf_counter()
            current_lr = optimizer.param_groups[0]["lr"]

            train_loss, train_accuracy = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            clean_loss, clean_accuracy = evaluate_clean(
                model, val_loader, criterion, device
            )
            monitor_accuracy = evaluate_pgd_monitor(
                model, monitor_loader, device
            )
            if not all(
                np.isfinite(value)
                for value in (
                    train_loss, train_accuracy, clean_loss,
                    clean_accuracy, monitor_accuracy,
                )
            ):
                raise RuntimeError("Non-finite training or validation metrics")

            torch.mps.synchronize()
            elapsed = time.perf_counter() - start
            elapsed_total += elapsed
            scheduler.step()

            row = {
                "epoch": epoch,
                "learning_rate": current_lr,
                "adv_train_loss": round(train_loss, 6),
                "adv_train_accuracy_pct": round(train_accuracy, 4),
                "clean_val_loss": round(clean_loss, 6),
                "clean_val_accuracy_pct": round(clean_accuracy, 4),
                "pgd10_monitor_accuracy_pct": round(monitor_accuracy, 4),
                "epoch_seconds": round(elapsed, 3),
                "cumulative_seconds": round(elapsed_total, 3),
            }
            writer.writerow(row)
            file.flush()

            #Prefer stronger PGD validation accuracy when saving best weights
            score = (monitor_accuracy, clean_accuracy)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "clean_val_accuracy_pct": clean_accuracy,
                        "pgd10_monitor_accuracy_pct": monitor_accuracy,
                        "config": config,
                    },
                    best_path,
                )

            #Keep the most recent weights and optimizer for inspection
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "clean_val_accuracy_pct": clean_accuracy,
                    "pgd10_monitor_accuracy_pct": monitor_accuracy,
                    "config": config,
                },
                last_path,
            )

            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Adv Train: {train_accuracy:.2f}% | "
                f"Clean Val: {clean_accuracy:.2f}% | "
                f"PGD-10 Monitor: {monitor_accuracy:.2f}% | "
                f"Time: {elapsed:.2f}s",
                flush=True,
            )

    print("\nAdversarial training completed", flush=True)
    print(f"Best checkpoint epoch: {best_epoch}", flush=True)
    print(f"Best PGD-10 monitor accuracy: {best_score[0]:.2f}%")
    print(f"Clean accuracy at best epoch: {best_score[1]:.2f}%")
    print(f"Total training and validation time: {elapsed_total:.2f}s")
    print(f"Best checkpoint: {best_path}")
    print(f"Latest checkpoint: {last_path}")
    print(f"History: {history_path}")
    print(f"Configuration: {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Verify one training update without saving model checkpoints",
    )
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("This run requires the verified Apple MPS device")
    device = torch.device("mps")

    if args.smoke_test:
        smoke_test(device)
    else:
        train(device)


if __name__ == "__main__":
    main()
