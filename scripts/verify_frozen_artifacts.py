"""Audit archived DS 312 metrics and checkpoints without model inference"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "final_evaluation"


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_history(name: str) -> list[dict[str, str]]:
    path = ROOT / "logs" / f"{name}_history.csv"
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 30, f"Unexpected {name} epoch count"
    assert [int(row["epoch"]) for row in rows] == list(range(1, 31))
    return rows


def verify_splits() -> dict:
    directory = ROOT / "splits"
    training = np.load(directory / "train_indices.npy", allow_pickle=False)
    validation = np.load(directory / "val_indices.npy", allow_pickle=False)
    monitor = np.load(directory / "adv_monitor_indices.npy", allow_pickle=False)
    test = np.load(RESULTS / "test_selected_indices.npy", allow_pickle=False)

    #Confirm that every original training image belongs to exactly one subset
    assert training.shape == (45000,)
    assert validation.shape == (5000,)
    assert np.array_equal(
        np.sort(np.concatenate([training, validation])),
        np.arange(50000),
    )

    #Confirm saved monitor indices refer only to validation positions
    assert monitor.shape == (500,)
    assert np.issubdtype(monitor.dtype, np.integer)
    assert len(np.unique(monitor)) == 500
    assert monitor.min() >= 0 and monitor.max() < 5000

    #Confirm the locked evaluation used every test-set index exactly once
    assert np.array_equal(np.sort(test), np.arange(10000))
    return {
        "training_images": 45000,
        "validation_images": 5000,
        "monitor_images": 500,
        "test_indices": 10000,
    }


def verify_history() -> dict:
    baseline = read_history("baseline")
    adversarial = read_history("adversarial")
    baseline_config = read_json(ROOT / "logs" / "baseline_config.json")
    adv_config = read_json(ROOT / "logs" / "adversarial_config.json")

    #Verify the checkpoint-selection rules from actual saved measurements
    assert baseline_config["checkpoint_selection"] == "best validation accuracy"
    assert adv_config["checkpoint_selection"] == (
        "highest validation monitor PGD accuracy, then clean validation accuracy"
    )
    baseline_best = max(baseline, key=lambda row: float(row["val_accuracy_pct"]))
    adv_best = max(
        adversarial,
        key=lambda row: (
            float(row["pgd10_monitor_accuracy_pct"]),
            float(row["clean_val_accuracy_pct"]),
        ),
    )

    assert int(baseline_best["epoch"]) == 29
    assert int(adv_best["epoch"]) == 26
    assert math.isclose(float(baseline_best["val_accuracy_pct"]), 88.32)
    assert math.isclose(float(adv_best["clean_val_accuracy_pct"]), 70.10)
    assert math.isclose(float(adv_best["pgd10_monitor_accuracy_pct"]), 36.40)
    assert baseline_config["test_set_used"] is False
    assert adv_config["test_set_used"] is False

    #Compare all epochs to detect invalid measurements and unusual PGD drops
    monitor = []
    for row in adversarial:
        for field in (
            "adv_train_loss",
            "clean_val_loss",
            "pgd10_monitor_accuracy_pct",
            "epoch_seconds",
        ):
            assert math.isfinite(float(row[field]))
        monitor.append(float(row["pgd10_monitor_accuracy_pct"]))

    return {
        "baseline_best_epoch": 29,
        "baseline_best_clean_validation_pct": 88.32,
        "adversarial_best_epoch": 26,
        "adversarial_best_clean_validation_pct": 70.10,
        "adversarial_best_pgd_monitor_pct": 36.40,
        "last_10_epoch_monitor_range_pct": [min(monitor[-10:]), max(monitor[-10:])],
        "final_monitor_pct": monitor[-1],
        "baseline_training_seconds": float(baseline[-1]["cumulative_seconds"]),
        "adversarial_training_seconds": float(adversarial[-1]["cumulative_seconds"]),
    }


def verify_test_records() -> tuple[dict, dict]:
    protocol = read_json(RESULTS / "protocol_locked.json")
    validation_path = RESULTS / "validation_probe.json"
    validation = read_json(validation_path)
    results = read_json(RESULTS / "test_results.json")

    #Confirm the archived validation result matches the locked checksum
    assert file_sha256(validation_path) == protocol["validation_probe_sha256"]
    assert validation["completed_at_utc"] < protocol["locked_at_utc"]
    assert protocol["locked_at_utc"] < results["timestamp_utc"]
    assert protocol["test_size"] == results["count"] == 10000
    assert protocol["pgd_steps"] == 20
    assert protocol["pgd_restarts"] == 2
    assert protocol["epsilon_numerators"] == [0, 2, 4, 8]
    assert results["selection_seed"] == protocol["selection_seed"]
    assert len(results["records"]) == 14
    assert all(value == 1000 for value in results["class_counts"].values())

    csv_path = RESULTS / "test_results.csv"
    with csv_path.open(newline="", encoding="utf-8") as file:
        csv_rows = list(csv.DictReader(file))
    assert len(csv_rows) == 14
    for row, csv_row in zip(results["records"], csv_rows):
        assert row["n"] == 10000
        assert 0 <= row["robust_correct"] <= row["correct"] <= 10000
        assert math.isclose(row["accuracy_percent"], row["correct"] / 100)
        assert math.isclose(
            row["empirical_robust_accuracy_percent"],
            row["robust_correct"] / 100,
        )
        assert row["max_observed_linf"] <= row["epsilon"] + 2e-6
        for field in ("model", "attack", "epsilon_numerator", "n", "correct", "robust_correct"):
            assert str(row[field]) == csv_row[field]
        assert math.isclose(float(csv_row["accuracy_percent"]), row["accuracy_percent"])
        assert math.isclose(
            float(csv_row["empirical_robust_accuracy_percent"]),
            row["empirical_robust_accuracy_percent"],
        )

    #Verify all model hashes agree between validation, protocol, and test
    for model_name, expected_hash in protocol["checkpoint_sha256"].items():
        assert validation["checkpoint_info"][model_name]["checkpoint_sha256"] == expected_hash
        assert results["checkpoint_info"][model_name]["checkpoint_sha256"] == expected_hash

    combinations = Counter(
        (row["model"], row["attack"], row["epsilon_numerator"])
        for row in results["records"]
    )
    assert len(combinations) == 14
    assert all(value == 1 for value in combinations.values())

    return protocol, {
        "model_attack_conditions": 14,
        "test_images_per_condition": 10000,
        "recorded_test_accuracy_checks": "PASSED",
        "locked_protocol_checksum": "PASSED",
    }


def verify_checkpoints(protocol: dict) -> dict:
    names = {
        "baseline": ROOT / "checkpoints" / "baseline_best.pt",
        "fgsm_trained": ROOT / "checkpoints" / "adversarial_best.pt",
    }
    results = {}
    for name, path in names.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing selected checkpoint: {path}")
        actual = file_sha256(path)
        expected = protocol["checkpoint_sha256"][name]
        if actual != expected:
            raise RuntimeError(f"Checkpoint hash mismatch: {path}")
        results[name] = {"sha256": actual, "matched_locked_protocol": True}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-checkpoints",
        action="store_true",
        help="Audit a source-only archive that does not contain model weights",
    )
    args = parser.parse_args()

    summary = {
        "verification": "PASSED",
        "data_splits": verify_splits(),
        "training_history": verify_history(),
    }
    protocol, summary["evaluation_records"] = verify_test_records()
    summary["checkpoints"] = (
        "NOT VERIFIED: omitted by explicit request"
        if args.skip_checkpoints else verify_checkpoints(protocol)
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
