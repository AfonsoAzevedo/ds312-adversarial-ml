import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.torch_version import TorchVersion
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from src.model import CompactCNN


BATCH_SIZE = 128


def main() -> None:
    #Load the saved model checkpoints and training history
    best_path = Path("checkpoints/baseline_best.pt")
    last_path = Path("checkpoints/baseline_last.pt")
    history_path = Path("logs/baseline_history.csv")
    config_path = Path("logs/baseline_config.json")

    #Allow the version metadata in our own checkpoint
    torch.serialization.add_safe_globals([TorchVersion])

    best_checkpoint = torch.load(
        best_path,
        map_location="cpu",
        weights_only=True,
    )

    last_checkpoint = torch.load(
        last_path,
        map_location="cpu",
        weights_only=True,
    )

    with history_path.open(newline="", encoding="utf-8") as file:
        history = list(csv.DictReader(file))

    config = json.loads(config_path.read_text(encoding="utf-8"))

    #Verify that the training run completed all planned epochs
    if len(history) != config["epochs"]:
        raise RuntimeError("Training history is incomplete")

    if last_checkpoint["epoch"] != config["epochs"]:
        raise RuntimeError("Latest checkpoint has the wrong epoch")

    #Identify the best epoch using the recorded validation accuracy
    best_row = max(
        history,
        key=lambda row: float(row["val_accuracy_pct"]),
    )

    best_epoch = int(best_row["epoch"])
    recorded_accuracy = float(best_row["val_accuracy_pct"])

    if best_checkpoint["epoch"] != best_epoch:
        raise RuntimeError("Best checkpoint has the wrong epoch")

    if abs(
        best_checkpoint["val_accuracy_pct"] - recorded_accuracy
    ) > 0.0001:
        raise RuntimeError("Checkpoint accuracy does not match the log")

    if best_checkpoint["config"] != config:
        raise RuntimeError("Checkpoint configuration does not match")

    #Check that the saved model parameters contain finite values
    for name, tensor in best_checkpoint["model_state_dict"].items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise RuntimeError(f"Non-finite model tensor: {name}")

    #Load the original validation images without augmentation
    dataset = CIFAR10(
        root="data",
        train=True,
        download=False,
        transform=transforms.ToTensor(),
    )

    val_indices = np.load(
        "splits/val_indices.npy",
        allow_pickle=False,
    )

    val_loader = DataLoader(
        Subset(dataset, val_indices.tolist()),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    #Create a new CNN and restore the best saved parameters
    model = CompactCNN().to(device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    correct = 0
    total = 0

    #Recompute clean validation accuracy independently
    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images).argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    verified_accuracy = 100.0 * correct / total

    #Allow a small tolerance for numerical device differences
    if abs(verified_accuracy - recorded_accuracy) > 0.1:
        raise RuntimeError(
            "Recomputed accuracy does not match training history"
        )

    results = {
        "verification": "PASSED",
        "device": str(device),
        "best_epoch": best_epoch,
        "last_epoch": last_checkpoint["epoch"],
        "recorded_validation_accuracy_pct": recorded_accuracy,
        "recomputed_validation_accuracy_pct": verified_accuracy,
        "correct_predictions": correct,
        "validation_images": total,
        "test_set_used": False,
    }

    #Save the verification results for future reference
    output_path = Path("logs/baseline_verification.json")
    output_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
