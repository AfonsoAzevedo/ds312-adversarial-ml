"""Reject inappropriate or missing files before creating the public repository"""

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    ".gitignore",
    "README.md",
    "requirements.txt",
    "src/attacks.py",
    "src/model.py",
    "src/train_baseline.py",
    "src/train_adversarial.py",
    "src/evaluate_final.py",
    "scripts/verify_frozen_artifacts.py",
    "scripts/check_checkpoint_loading.py",
    "splits/train_indices.npy",
    "splits/val_indices.npy",
    "checkpoints/baseline_best.pt",
    "checkpoints/adversarial_best.pt",
    "logs/baseline_history.csv",
    "logs/adversarial_history.csv",
    "results/final_evaluation/protocol_locked.json",
    "results/final_evaluation/test_results.json",
    "results/final_evaluation/test_results.csv",
}


def main() -> None:
    #Inspect the actual Git index rather than guessing what will be published
    output = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only", "-z"],
        cwd=ROOT,
    )
    paths = {value.decode() for value in output.split(b"\0") if value}

    missing = REQUIRED - paths
    if missing:
        raise RuntimeError(f"Required files not staged: {sorted(missing)}")

    #Reject datasets, virtual environments and intermediate checkpoints
    forbidden = []
    for name in paths:
        parts = Path(name).parts
        if any(part in ("data", "datasets", ".venv", "__pycache__") for part in parts):
            forbidden.append(name)
        if name.endswith((".zip", ".DS_Store", ".pem", ".env")):
            forbidden.append(name)
        if name.startswith("checkpoints/") and name not in {
            "checkpoints/baseline_best.pt",
            "checkpoints/adversarial_best.pt",
        }:
            forbidden.append(name)
        path = ROOT / name
        if path.is_file() and path.stat().st_size > 50 * 1024 * 1024:
            forbidden.append(name)

    if forbidden:
        raise RuntimeError(f"Unsafe files staged: {sorted(set(forbidden))}")

    print(f"Git staging check PASSED ({len(paths)} files)")
    for name in sorted(paths):
        print(name)


if __name__ == "__main__":
    main()
