import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.attacks import pgd_attack
from src.data import create_data_loaders
from src.model import CompactCNN


CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")
SPLIT_DIR = Path("splits")


def read_training_history(path: Path) -> list[dict[str, str]]:
    #Read the training measurements saved after each epoch
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def evaluate_accuracy(
    model: CompactCNN,
    loader: DataLoader,
    device: torch.device,
    attack_settings: dict | None = None,
) -> tuple[int, int]:
    #Evaluate predictions independently of the training script
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        #PGD needs input gradients even when the model is in eval mode
        if attack_settings is not None:
            images = pgd_attack(
                model=model,
                images=images,
                labels=labels,
                **attack_settings,
            )

        with torch.inference_mode():
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())

    return correct, total


def main() -> None:
    best_path = CHECKPOINT_DIR / "adversarial_best.pt"
    last_path = CHECKPOINT_DIR / "adversarial_last.pt"
    history_path = LOG_DIR / "adversarial_history.csv"
    config_path = LOG_DIR / "adversarial_config.json"
    monitor_path = SPLIT_DIR / "adv_monitor_indices.npy"

    #Load only checkpoints created by our local training script
    best_checkpoint = torch.load(
        best_path, map_location="cpu", weights_only=True
    )
    last_checkpoint = torch.load(
        last_path, map_location="cpu", weights_only=True
    )
    history = read_training_history(history_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    #Check the recorded training history and checkpoint selection
    if len(history) != config["epochs"]:
        raise RuntimeError("Training history has the wrong number of epochs")
    if int(last_checkpoint["epoch"]) != config["epochs"]:
        raise RuntimeError("Latest checkpoint does not match the final epoch")
    if best_checkpoint["config"] != config:
        raise RuntimeError("Best checkpoint configuration does not match")
    if last_checkpoint["config"] != config:
        raise RuntimeError("Latest checkpoint configuration does not match")

    best_row = max(
        history,
        key=lambda row: (
            float(row["pgd10_monitor_accuracy_pct"]),
            float(row["clean_val_accuracy_pct"]),
        ),
    )
    best_epoch = int(best_row["epoch"])
    if int(best_checkpoint["epoch"]) != best_epoch:
        raise RuntimeError("Best checkpoint does not match selection rule")

    recorded_clean = float(best_row["clean_val_accuracy_pct"])
    recorded_pgd = float(best_row["pgd10_monitor_accuracy_pct"])
    if abs(best_checkpoint["clean_val_accuracy_pct"] - recorded_clean) > 1e-4:
        raise RuntimeError("Checkpoint clean accuracy differs from history")
    if abs(best_checkpoint["pgd10_monitor_accuracy_pct"] - recorded_pgd) > 1e-4:
        raise RuntimeError("Checkpoint PGD accuracy differs from history")

    #Check all model weights and BatchNorm buffers for invalid values
    for name, tensor in best_checkpoint["model_state_dict"].items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise RuntimeError(f"Invalid checkpoint tensor: {name}")

    if not torch.backends.mps.is_available():
        raise RuntimeError("Expected the verified MPS training device")
    device = torch.device("mps")

    #Load validation data only, without random augmentations
    _, val_loader = create_data_loaders(
        batch_size=config["batch_size"], seed=config["seed"]
    )
    val_subset = val_loader.dataset
    positions = np.load(monitor_path, allow_pickle=False)

    #Monitor positions refer to the fixed 5,000-image validation subset
    if positions.ndim != 1 or len(positions) != config["monitor_images"]:
        raise RuntimeError("Unexpected monitor subset size")
    if not np.issubdtype(positions.dtype, np.integer):
        raise RuntimeError("Monitor indices must be integers")
    if len(np.unique(positions)) != len(positions):
        raise RuntimeError("Duplicate monitoring positions")
    if positions.min() < 0 or positions.max() >= len(val_subset):
        raise RuntimeError("Monitoring positions are out of range")

    #Verify that all ten classes have equal monitoring representation
    original_labels = np.asarray(val_subset.dataset.targets)
    validation_labels = original_labels[np.asarray(val_subset.indices)]
    class_counts = np.bincount(validation_labels[positions], minlength=10)
    if not np.all(class_counts == config["monitor_images_per_class"]):
        raise RuntimeError("PGD monitor is not stratified correctly")

    monitor_loader = DataLoader(
        Subset(val_subset, positions.tolist()),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    #Restore the selected model from a new Python model instance
    model = CompactCNN().to(device)
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    if sum(p.numel() for p in model.parameters()) != 288746:
        raise RuntimeError("The model architecture was unexpectedly changed")

    clean_correct, clean_total = evaluate_accuracy(
        model, val_loader, device
    )
    pgd_correct, pgd_total = evaluate_accuracy(
        model,
        monitor_loader,
        device,
        attack_settings={
            "epsilon": config["monitor_epsilon"],
            "step_size": config["monitor_step_size"],
            "steps": config["monitor_pgd_steps"],
            "random_start": config["monitor_random_start"],
        },
    )
    clean_accuracy = 100.0 * clean_correct / clean_total
    pgd_accuracy = 100.0 * pgd_correct / pgd_total

    #Require the saved checkpoint to reproduce both recorded metrics
    if abs(clean_accuracy - recorded_clean) > 1e-4:
        raise RuntimeError("Independent clean accuracy does not match")
    if abs(pgd_accuracy - recorded_pgd) > 1e-4:
        raise RuntimeError("Independent PGD monitor accuracy does not match")

    results = {
        "verification": "PASSED",
        "device": str(device),
        "best_epoch": best_epoch,
        "last_epoch": int(last_checkpoint["epoch"]),
        "clean_val_correct": clean_correct,
        "clean_val_images": clean_total,
        "clean_val_accuracy_pct": round(clean_accuracy, 4),
        "pgd_monitor_correct": pgd_correct,
        "pgd_monitor_images": pgd_total,
        "pgd_monitor_accuracy_pct": round(pgd_accuracy, 4),
        "monitor_epsilon": config["monitor_epsilon"],
        "monitor_step_size": config["monitor_step_size"],
        "monitor_steps": config["monitor_pgd_steps"],
        "monitor_random_start": config["monitor_random_start"],
        "monitor_class_counts": class_counts.tolist(),
        "checkpoint_matches_history": True,
        "test_set_used": False,
    }

    #Save independent verification without changing either checkpoint
    output_path = LOG_DIR / "adversarial_verification.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
