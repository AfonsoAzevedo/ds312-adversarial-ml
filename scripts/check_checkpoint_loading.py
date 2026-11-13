"""Load frozen checkpoints on CPU without running model evaluation"""

import json
from pathlib import Path

import torch
from torch.torch_version import TorchVersion

from src.model import CompactCNN


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "baseline": ("baseline_best.pt", 29),
    "fgsm_trained": ("adversarial_best.pt", 26),
}


def main() -> None:
    #Load only our verified local checkpoints with restricted deserialization
    results = {}

    for name, (filename, expected_epoch) in CHECKPOINTS.items():
        path = ROOT / "checkpoints" / filename

        with torch.serialization.safe_globals([TorchVersion]):
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )

        if checkpoint["epoch"] != expected_epoch:
            raise RuntimeError(f"Unexpected epoch in {filename}")

        #Require every parameter and buffer to match the locked architecture
        model = CompactCNN()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        for parameter in model.parameters():
            if not torch.isfinite(parameter).all():
                raise RuntimeError(f"Invalid model parameters in {filename}")

        for buffer in model.buffers():
            if buffer.is_floating_point() and not torch.isfinite(buffer).all():
                raise RuntimeError(f"Invalid model buffers in {filename}")

        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        if parameter_count != 288746:
            raise RuntimeError(f"Unexpected model size in {filename}")

        results[name] = {
            "checkpoint": filename,
            "epoch": checkpoint["epoch"],
            "parameters": parameter_count,
            "cpu_state_dict_loading": "PASSED",
        }

    print(json.dumps({"verification": "PASSED", "models": results}, indent=2))


if __name__ == "__main__":
    main()
