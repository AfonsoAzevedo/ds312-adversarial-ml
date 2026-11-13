"""Validation-first, frozen-checkpoint CIFAR-10 adversarial evaluation.

Run from the existing ds312-adversarial-ml repository root:
    python -m src.evaluate_final validate
    python -m src.evaluate_final lock --test-size 2000
    python -m src.evaluate_final test

DO NOT run 'test' before reviewing validation and the locked protocol.
This module never trains, optimizes, or saves model parameters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from src.model import CompactCNN
from src.attacks import fgsm_attack  #Reuse the existing, tested FGSM implementation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final_evaluation"
CHECKPOINTS = {
    "baseline": ROOT / "checkpoints" / "baseline_best.pt",
    "fgsm_trained": ROOT / "checkpoints" / "adversarial_best.pt",
}
EPSILON_NUMERATORS = (0, 2, 4, 8)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")
SEED = 42
SELECTION_SEED = 20261112
VALIDATION_COUNT = 500  #50 per class, never test data
PGD_ALPHA_RULE = "max(epsilon/4, 1/255)"  #Frozen on validation only


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def setup_device():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_frozen_model(name: str, device: torch.device):
    ckpt_path = CHECKPOINTS[name]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Required checkpoint missing: {ckpt_path}")
    #Allow trusted PyTorch version metadata while keeping safe loading
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise RuntimeError(
            f"Unexpected checkpoint structure at {ckpt_path}. "
            "Expected a dict with 'model_state_dict'; inspect before adapting."
        )
    model = CompactCNN()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    n_parameters = sum(t.numel() for t in model.parameters())
    if n_parameters != 288746:
        raise RuntimeError(f"CompactCNN parameter mismatch: {n_parameters} != 288746")
    model = model.to(device).eval()
    #No parameter gradients are needed for white-box input-gradient attacks
    for param in model.parameters():
        param.requires_grad_(False)
    return model, {
        "checkpoint": str(ckpt_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "parameter_count": n_parameters,
        "checkpoint_epoch": ckpt.get("epoch") if isinstance(ckpt.get("epoch"), (int, float)) else None,
    }


def locate_cifar10() -> Path:
    """Find local data only; never silently change the dataset source."""
    options = [ROOT / "data", ROOT / "datasets", ROOT / "data" / "cifar10"]
    for path in options:
        if (path / "cifar-10-batches-py").is_dir():
            return path
    #torchvision CIFAR10 may be downloadable in a fresh clone, but no network
    #is used here: keep this experiment tied to the already-prepared dataset
    raise FileNotFoundError(
        "CIFAR-10 files not found in data/, datasets/, or data/cifar10/. "
        "Check the dataset location before running; do not change the split."
    )


def stratified_indices(base_indices, labels, count: int, seed: int) -> np.ndarray:
    """Select a fixed, balanced subset; labels are only used for stratification."""
    base_indices = np.asarray(base_indices, dtype=np.int64)
    if len(base_indices) != len(np.unique(base_indices)):
        raise ValueError("Repeated indices in source split")
    if count > len(base_indices) or count % 10 != 0:
        raise ValueError("Count must be <= split size and divisible by ten")
    generator = np.random.default_rng(seed)
    targets = np.asarray(labels, dtype=np.int64)
    result = []
    for klass in range(10):
        possible = base_indices[targets[base_indices] == klass]
        if len(possible) < count // 10:
            raise ValueError(f"Not enough examples of class {klass}")
        result.extend(generator.choice(possible, size=count // 10, replace=False).tolist())
    #Sorted makes saved adversarial example selection deterministic
    return np.asarray(sorted(result), dtype=np.int64)


def loader_for_stage(stage: str, n: int, batch_size: int):
    cifar_root = locate_cifar10()
    if stage == "validation":
        dataset = CIFAR10(root=str(cifar_root), train=True, download=False, transform=ToTensor())
        val_path = ROOT / "splits" / "val_indices.npy"
        train_path = ROOT / "splits" / "train_indices.npy"
        val_idx = np.load(val_path, allow_pickle=False).astype(np.int64)
        train_idx = np.load(train_path, allow_pickle=False).astype(np.int64)
        if len(val_idx) != 5000 or len(train_idx) != 45000:
            raise RuntimeError("Unexpected original training/validation split sizes")
        if len(np.unique(np.r_[val_idx, train_idx])) != 50000:
            raise RuntimeError("Training/validation indices overlap or are duplicated")
        if min(val_idx) < 0 or max(val_idx) >= 50000:
            raise RuntimeError("Invalid validation indices")
        indices = stratified_indices(val_idx, dataset.targets, n, SELECTION_SEED)
        split_hashes = {
            "val_indices_sha256": sha256_file(val_path),
            "train_indices_sha256": sha256_file(train_path),
        }
    elif stage == "test":
        #Critical: the test dataset is not even constructed in validation/lock
        dataset = CIFAR10(root=str(cifar_root), train=False, download=False, transform=ToTensor())
        if len(dataset) != 10000:
            raise RuntimeError(f"Unexpected CIFAR-10 test size: {len(dataset)}")
        indices = stratified_indices(np.arange(10000), dataset.targets, n, SELECTION_SEED)
        split_hashes = {}
    else:
        raise ValueError(stage)
    if len(indices) != n or not all(np.bincount(np.asarray(dataset.targets)[indices], minlength=10) == n // 10):
        raise AssertionError("Unbalanced evaluation sample")
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size,
                        shuffle=False, num_workers=0, pin_memory=False)
    return loader, indices, split_hashes


def assert_valid_perturbation(clean: torch.Tensor, attacked: torch.Tensor, epsilon: float):
    if clean.shape != attacked.shape:
        raise AssertionError("Image shape changed under attack")
    if not bool(torch.isfinite(attacked).all().item()):
        raise AssertionError("Non-finite adversarial image")
    if float(attacked.min().item()) < -1e-6 or float(attacked.max().item()) > 1 + 1e-6:
        raise AssertionError("Adversarial image outside [0,1]")
    worst = float((attacked - clean).abs().max().item())
    if worst > epsilon + 2e-6:
        raise AssertionError(f"Perturbation bound exceeded ({worst:.9f} > {epsilon:.9f})")
    return worst


def pgd_attack_evaluation(model: torch.nn.Module, images: torch.Tensor,
                          labels: torch.Tensor, epsilon: float,
                          steps: int, restarts: int, seed: int):
    """Evaluation-only untargeted L-infinity PGD with fresh random restarts.

    Tracks the strongest (successful first, then maximum CE) intermediate
    iterate for each image, not simply the last iterate of each restart.
    Existing training checkpoints and src/attacks.py are left unchanged.
    """
    if steps < 1 or restarts < 1 or epsilon <= 0 or epsilon > 1:
        raise ValueError("Invalid attack settings")
    if images.ndim != 4 or images.shape[1:] != (3, 32, 32):
        raise ValueError("PGD expected CIFAR-10 [B,3,32,32]")
    model.eval()
    x0 = images.detach()
    #CPU random generator provides predictable starts on MPS as well as CPU
    generator = torch.Generator(device="cpu").manual_seed(seed)
    alpha = max(epsilon / 4.0, 1.0 / 255.0)
    best_x = x0.clone()
    best_loss = torch.full((len(labels),), -float("inf"), device=x0.device)
    best_success = torch.zeros(len(labels), dtype=torch.bool, device=x0.device)

    def remember(candidate):
        nonlocal best_x, best_loss, best_success
        with torch.no_grad():
            scores = model(candidate)
            losses = F.cross_entropy(scores, labels, reduction="none")
            success = scores.argmax(dim=1).ne(labels)
            replace = (success & ~best_success) | ((success == best_success) & (losses > best_loss))
            best_x = torch.where(replace[:, None, None, None], candidate.detach(), best_x)
            best_loss = torch.where(replace, losses, best_loss)
            best_success = best_success | success

    for _ in range(restarts):
        #Uniform initialization inside the actual feasible intersection
        noise = (2 * torch.rand(x0.shape, generator=generator) - 1).to(x0.device) * epsilon
        adv = (x0 + noise).clamp(0.0, 1.0).detach()
        remember(adv)
        for _ in range(steps):
            with torch.enable_grad():
                x = adv.detach().requires_grad_(True)
                loss = F.cross_entropy(model(x), labels)
                gradient = torch.autograd.grad(loss, x, only_inputs=True)[0]
            if not bool(torch.isfinite(gradient).all().item()):
                raise RuntimeError("PGD encountered non-finite input gradients")
            with torch.no_grad():
                adv = x.detach() + alpha * gradient.sign()
                adv = torch.max(torch.min(adv, x0 + epsilon), x0 - epsilon)
                adv = adv.clamp(0.0, 1.0).detach()
            remember(adv)
    return best_x.detach()


def evaluate_condition(model, loader, device, attack, eps, steps, restarts,
                       model_id, stage, collect_examples=False):
    start = time.perf_counter()
    n = 0
    correct = 0
    robust_correct = 0
    clean_correct_count = 0
    max_delta = 0.0
    example_bank = []
    model.eval()
    for batch_number, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            clean_predictions = model(images).argmax(1)
        if attack == "clean":
            adversarial = images.detach()
        elif attack == "fgsm":
            with torch.enable_grad():
                adversarial = fgsm_attack(model, images, labels, eps).detach()
        elif attack == "pgd":
            #Deterministic separate random streams by stage, checkpoint and epsilon
            #Restarts within the call use distinct random starts
            stage_code = 1 if stage == "validation" else 2
            #Identical random initializations for both frozen models
            batch_seed = 1000000 * stage_code + 1000 * round(eps * 255) + batch_number
            adversarial = pgd_attack_evaluation(
                model, images, labels, eps, steps, restarts, batch_seed)
        else:
            raise ValueError(attack)
        max_delta = max(max_delta, assert_valid_perturbation(images, adversarial, eps))
        with torch.no_grad():
            attacked_predictions = model(adversarial).argmax(1)
        clean_good = clean_predictions.eq(labels)
        attack_good = attacked_predictions.eq(labels)
        n += len(labels)
        correct += int(attack_good.sum().item())
        clean_correct_count += int(clean_good.sum().item())
        robust_correct += int((clean_good & attack_good).sum().item())
        #Only sample images for a predefined illustrative criterion
        if collect_examples and eps == 8 / 255 and attack == "pgd" and len(example_bank) < 6:
            desired = torch.nonzero(clean_good & ~attack_good).flatten().tolist()
            for idx in desired[:6 - len(example_bank)]:
                example_bank.append({
                    "clean": images[idx].detach().cpu().numpy(),
                    "attacked": adversarial[idx].detach().cpu().numpy(),
                    "true_label": int(labels[idx].item()),
                    "clean_prediction": int(clean_predictions[idx].item()),
                    "attack_prediction": int(attacked_predictions[idx].item()),
                    "batch": batch_number,
                    "local_index": int(idx),
                })
    if n <= 0:
        raise RuntimeError("Evaluation loader was empty")
    if attack == "clean" and correct != clean_correct_count:
        raise AssertionError("Clean evaluation inconsistent")
    if robust_correct > clean_correct_count:
        raise AssertionError("Empirical robust count exceeds clean count")
    runtime = time.perf_counter() - start
    return {
        "model": model_id, "stage": stage, "attack": attack,
        "epsilon_numerator": round(eps * 255), "epsilon": eps,
        "pgd_steps": steps if attack == "pgd" else None,
        "pgd_restarts": restarts if attack == "pgd" else None,
        "pgd_alpha": max(eps / 4.0, 1.0 / 255.0) if attack == "pgd" else None,
        "n": n, "correct": correct, "accuracy_percent": round(100 * correct / n, 4),
        "robust_correct": robust_correct,
        "empirical_robust_accuracy_percent": round(100 * robust_correct / n, 4),
        "clean_correct_on_same_samples": clean_correct_count,
        "runtime_seconds": round(runtime, 4),
        "max_observed_linf": round(max_delta, 9),
        "selection_seed": SELECTION_SEED,
    }, example_bank


def conditions_for(stage: str, locked: dict | None = None):
    tests = [("clean", 0.0, 0, 0, "clean")]
    for value in EPSILON_NUMERATORS[1:]:
        eps = value / 255.0
        tests.append(("fgsm", eps, 0, 0, "fgsm"))
        if stage == "validation":
            tests.extend([
                ("pgd", eps, 10, 1, "pgd10_r1"),
                ("pgd", eps, 20, 2, "pgd20_r2"),
                ("pgd", eps, 40, 2, "pgd40_r2_diagnostic"),
            ])
        else:
            tests.append(("pgd", eps, locked["pgd_steps"], locked["pgd_restarts"], "pgd_final"))
    return tests


def summarize_warnings(records):
    indexed = {(r["model"], r["epsilon_numerator"], r["variant"]): r for r in records}
    warnings = []
    for model in CHECKPOINTS:
        clean = indexed[(model, 0, "clean")]["accuracy_percent"]
        for budget in (2, 4, 8):
            a = indexed[(model, budget, "fgsm")]["empirical_robust_accuracy_percent"]
            p10 = indexed[(model, budget, "pgd10_r1")]["empirical_robust_accuracy_percent"]
            p20 = indexed[(model, budget, "pgd20_r2")]["empirical_robust_accuracy_percent"]
            if p20 > a + 1.0:
                warnings.append(f"{model} epsilon={budget}/255: PGD20 appears weaker than FGSM by >1pp")
            if p20 > p10 + 1.0:
                warnings.append(f"{model} epsilon={budget}/255: PGD20-r2 appears weaker than PGD10-r1 by >1pp")
            if p20 > clean + 1e-6:
                warnings.append(f"{model} epsilon={budget}/255: impossible robust>clean comparison")
        for budget in (2, 4, 8):
            strong = indexed[(model, budget, "pgd40_r2_diagnostic")]["empirical_robust_accuracy_percent"]
            medium = indexed[(model, budget, "pgd20_r2")]["empirical_robust_accuracy_percent"]
            if medium - strong > 2.0:
                warnings.append(f"{model} epsilon={budget}/255: PGD40 improves over PGD20 by >2pp; consider PGD40 final")
    return warnings


def csv_from_records(path, records):
    if not records:
        return
    keys = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def run_evaluation(stage: str, count: int, batch_size: int, locked=None):
    device = setup_device()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Stage={stage} device={device} n={count} batch={batch_size}", flush=True)
    loader, selected_indices, split_hashes = loader_for_stage(stage, count, batch_size)
    np.save(OUT / f"{stage}_selected_indices.npy", selected_indices)
    meta = {
        "stage": stage, "timestamp_utc": utc_now(), "seed": SEED,
        "selection_seed": SELECTION_SEED, "count": count,
        "class_counts": dict(Counter(int(np.asarray(loader.dataset.dataset.targets)[i]) for i in selected_indices)),
        "selected_indices_sha256": hashlib.sha256(selected_indices.tobytes()).hexdigest(),
        "split_hashes": split_hashes, "device": str(device),
        "torch": torch.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "batch_size": batch_size,
        "checkpoint_info": {}, "pixel_preprocessing": "ToTensor() only; unnormalized [0,1]",
        "prior_training_times_seconds": {
            "baseline": 395.72, "fgsm_trained": 632.41,
        },
        "prior_training_times_provenance": "Chat 02 verified MAIN CHAT HANDOFF, not recomputed in evaluation",
        "linf_pixel_budget_numerators": list(EPSILON_NUMERATORS),
        "pgd_alpha_rule": PGD_ALPHA_RULE,
        "pgd_random_start": "uniform[-epsilon,+epsilon], clamped to image and box",
        "pgd_best_iterate": "prefer misclassification, then maximal CE; includes intermediate steps",
        "attack_objective": "untargeted cross entropy, true labels",
    }
    records = []
    examples = {}
    for model_id in CHECKPOINTS:
        model, checkpoint_meta = load_frozen_model(model_id, device)
        meta["checkpoint_info"][model_id] = checkpoint_meta
        for attack, eps, steps, restarts, variant in conditions_for(stage, locked):
            print(f"{model_id:12s} {variant:23s} eps={round(eps*255)}/255", flush=True)
            row, images = evaluate_condition(
                model, loader, device, attack, eps, steps, restarts,
                model_id, stage, collect_examples=(stage == "test"))
            row["variant"] = variant
            records.append(row)
            print(f"   {row['correct']}/{row['n']} = {row['accuracy_percent']:.2f}% "
                  f"(robust: {row['empirical_robust_accuracy_percent']:.2f}%) "
                  f"in {row['runtime_seconds']:.2f}s", flush=True)
            if images:
                examples[model_id] = images
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    meta["completed_at_utc"] = utc_now()
    meta["records"] = records
    meta["validation_warnings"] = summarize_warnings(records) if stage == "validation" else []
    path = OUT / ("validation_probe.json" if stage == "validation" else "test_results.json")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {path}")
    write_json(path, meta)
    csv_from_records(OUT / ("validation_probe.csv" if stage == "validation" else "test_results.csv"), records)
    if stage == "test":
        #Only correctly classified clean examples that PGD actually fools
        np.savez_compressed(OUT / "adversarial_examples.npz", **{
            f"{model_id}_{kind}": np.stack([item[kind] for item in bank])
            for model_id, bank in examples.items() for kind in ("clean", "attacked")
        }, **{
            f"{model_id}_{kind}": np.asarray([item[kind] for item in bank])
            for model_id, bank in examples.items()
            for kind in ("true_label", "clean_prediction", "attack_prediction")
        }) if examples else None
    print(f"Saved {path}")
    if stage == "validation":
        print("Diagnostics:", meta["validation_warnings"] or "None triggered")
        print("Review validation metrics and runtime BEFORE locking test settings.")


def lock(args):
    validation_path = OUT / "validation_probe.json"
    if not validation_path.exists():
        raise RuntimeError("Run validation first. The test protocol cannot be locked without validation data")
    validation = read_json(validation_path)
    if validation["count"] != VALIDATION_COUNT:
        raise RuntimeError("Validation probe count changed; review before locking")
    if validation["validation_warnings"] and not args.acknowledge_diagnostics:
        raise RuntimeError("Validation diagnostics require review:\n" +
                           "\n".join(validation["validation_warnings"]) +
                           "\nAfter investigating, rerun lock with --acknowledge-diagnostics and document the decision.")
    if args.test_size not in (1000, 2000, 5000, 10000):
        raise ValueError("Test size must be 1000, 2000, 5000, or 10000")
    if args.pgd_steps not in (20, 40) or args.pgd_restarts not in (1, 2):
        raise ValueError("PGD final settings limited to validated step/restart combinations")
    if (args.pgd_steps, args.pgd_restarts) not in [(20, 2), (40, 2)]:
        raise ValueError("This protocol validates PGD20-r2 and PGD40-r2; use one of those only")
    ckpts = validation["checkpoint_info"]
    for name in CHECKPOINTS:
        if sha256_file(CHECKPOINTS[name]) != ckpts[name]["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint changed since validation; revalidate")
    protocol = {
        "locked_at_utc": utc_now(), "validation_probe_sha256": sha256_file(validation_path),
        "checkpoint_sha256": {n: ckpts[n]["checkpoint_sha256"] for n in CHECKPOINTS},
        "test_size": args.test_size, "batch_size": args.batch_size,
        "pgd_steps": args.pgd_steps, "pgd_restarts": args.pgd_restarts,
        "epsilon_numerators": list(EPSILON_NUMERATORS),
        "selection_seed": SELECTION_SEED, "alpha_rule": PGD_ALPHA_RULE,
        "decision_rationale": args.rationale,
        "diagnostics_reviewed": bool(args.acknowledge_diagnostics or not validation["validation_warnings"]),
        "development_only": False,
    }
    lock_path = OUT / "protocol_locked.json"
    if lock_path.exists():
        raise FileExistsError("Protocol already locked; do not silently revise after test access")
    write_json(lock_path, protocol)
    print(json.dumps(protocol, indent=2))
    print("The test set remains unopened. You may now run the final test command once.")


def run_test():
    lock_path = OUT / "protocol_locked.json"
    if not lock_path.exists():
        raise RuntimeError("STOP: no locked protocol. Run validate and lock first")
    lock_data = read_json(lock_path)
    validation_path = OUT / "validation_probe.json"
    if sha256_file(validation_path) != lock_data["validation_probe_sha256"]:
        raise RuntimeError("Validation record changed after locking")
    for name in CHECKPOINTS:
        if sha256_file(CHECKPOINTS[name]) != lock_data["checkpoint_sha256"][name]:
            raise RuntimeError("Frozen checkpoint changed after locking")
    if (OUT / "test_results.json").exists():
        raise FileExistsError("Test results already exist. Do not perform repeated test-driven optimization")
    run_evaluation("test", lock_data["test_size"], lock_data["batch_size"], locked=lock_data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="stage", required=True)
    v = subs.add_parser("validate", help="Run validation-only PGD strength/runtime checks")
    v.add_argument("--batch-size", type=int, default=64)
    lock_parser = subs.add_parser("lock", help="Lock test parameters from validation, without opening test set")
    lock_parser.add_argument("--test-size", type=int, default=2000)
    lock_parser.add_argument("--batch-size", type=int, default=64)
    lock_parser.add_argument("--pgd-steps", type=int, default=20)
    lock_parser.add_argument("--pgd-restarts", type=int, default=2)
    lock_parser.add_argument("--rationale", type=str, required=True,
                   help="One sentence explaining chosen N and PGD settings using validation runtime/strength")
    lock_parser.add_argument("--acknowledge-diagnostics", action="store_true")
    subs.add_parser("test", help="Read test only after protocol lock; refuse reruns")
    args = parser.parse_args()
    if args.stage == "validate":
        run_evaluation("validation", VALIDATION_COUNT, args.batch_size)
    elif args.stage == "lock":
        lock(args)
    else:
        run_test()


if __name__ == "__main__":
    main()
