"""Create reproducible paper figures exclusively from frozen evaluation records.

Run only after 'python -m src.evaluate_final test' has saved test_results.json.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "final_evaluation"
FIGURES = RESULTS / "figures"
MODEL_NAMES = {"baseline": "Standard training", "fgsm_trained": "FGSM adversarial training"}
CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck")


def make_figures():
    path = RESULTS / "test_results.json"
    if not path.exists():
        raise FileNotFoundError("No final test results. Do not generate invented figures.")
    payload = json.loads(path.read_text())
    rows = payload["records"]
    n_values = {row["n"] for row in rows}
    if len(n_values) != 1 or n_values != {payload["count"]}:
        raise RuntimeError("Inconsistent denominators in final evaluation results")
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Paper-ready comparison table generated from actual saved measurements.
    table = []
    for budget in (0, 2, 4, 8):
        item = {"epsilon_over_255": budget, "n": payload["count"]}
        for model in MODEL_NAMES:
            if budget == 0:
                row = next(r for r in rows if r["model"] == model and r["attack"] == "clean")
                item[f"{model}_clean_accuracy_pct"] = row["accuracy_percent"]
            else:
                for attack in ("fgsm", "pgd"):
                    row = next(r for r in rows if r["model"] == model and r["attack"] == attack
                               and r["epsilon_numerator"] == budget)
                    item[f"{model}_{attack}_robust_pct"] = row["empirical_robust_accuracy_percent"]
        table.append(item)
    columns = ["epsilon_over_255", "n", "baseline_clean_accuracy_pct",
               "fgsm_trained_clean_accuracy_pct", "baseline_fgsm_robust_pct",
               "fgsm_trained_fgsm_robust_pct", "baseline_pgd_robust_pct",
               "fgsm_trained_pgd_robust_pct"]
    with (RESULTS / "paper_comparison_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)

    # Main comparison figure. Use empirical robust accuracy: correct on both
    # the clean original and the strongest observed perturbation.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), sharey=True, constrained_layout=True)
    for ax, attack, attack_label in zip(axes, ("fgsm", "pgd"), ("FGSM", "PGD")):
        for model, readable in MODEL_NAMES.items():
            x, y = [], []
            for budget in (0, 2, 4, 8):
                match = [r for r in rows if r["model"] == model and r["epsilon_numerator"] == budget
                         and r["attack"] == ("clean" if budget == 0 else attack)]
                if len(match) != 1:
                    raise ValueError(f"Missing / duplicated {model} {attack} epsilon={budget}")
                x.append(budget)
                y.append(match[0]["empirical_robust_accuracy_percent"])
            ax.plot(x, y, marker="o", linewidth=1.9, markersize=5, label=readable)
        ax.set_title(f"{attack_label} white-box attack")
        ax.set_xticks([0, 2, 4, 8], ["0", "2", "4", "8"])
        ax.set_xlabel(r"$L_\infty$ budget ($\epsilon$ / 255)")
        ax.grid(True, linewidth=0.5, alpha=0.35)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Empirical robust accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.045), ncol=2, frameon=False)
    fig.suptitle(f"CIFAR-10 | matched {payload['count']:,}-image test sample", fontsize=11)
    fig.savefig(FIGURES / "accuracy_vs_epsilon.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "accuracy_vs_epsilon.pdf", bbox_inches="tight")
    plt.close(fig)

    # The attack-specific inference cost includes attack generation and final
    # correctness checks. It does not double-count original training duration.
    runtime_rows = [r for r in rows if r["epsilon_numerator"] == 8
                    and r["attack"] in ("fgsm", "pgd")]
    fig, ax = plt.subplots(figsize=(7.5, 3.8), constrained_layout=True)
    x = np.arange(2)
    width = 0.36
    for j, model in enumerate(MODEL_NAMES):
        durations = []
        for attack in ("fgsm", "pgd"):
            durations.append(next(r["runtime_seconds"] for r in runtime_rows
                                  if r["model"] == model and r["attack"] == attack))
        bars = ax.bar(x + (j - 0.5) * width, durations, width, label=MODEL_NAMES[model])
        ax.bar_label(bars, fmt="%.1fs", padding=2, fontsize=8)
    ax.set_xticks(x, ["FGSM", "PGD"])
    ax.set_ylabel("Evaluation time (seconds)")
    ax.set_title(r"Attack evaluation cost at $\epsilon = 8/255$")
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)
    fig.savefig(FIGURES / "evaluation_runtime.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "evaluation_runtime.pdf", bbox_inches="tight")
    plt.close(fig)

    # Explicit selection criterion: first up to six examples that were
    # initially correctly predicted and are then misclassified by PGD.
    examples_path = RESULTS / "adversarial_examples.npz"
    if examples_path.exists():
        examples = np.load(examples_path, allow_pickle=False)
        for model, label in MODEL_NAMES.items():
            clean_key, adv_key = f"{model}_clean", f"{model}_attacked"
            if clean_key not in examples or adv_key not in examples:
                continue
            clean_images = examples[clean_key]
            adv_images = examples[adv_key]
            if not len(clean_images):
                continue
            n = len(clean_images)
            fig, axes = plt.subplots(n, 3, figsize=(7.5, 1.8 * n + 0.8), squeeze=False,
                                     constrained_layout=True)
            truth = examples[f"{model}_true_label"]
            pred_adv = examples[f"{model}_attack_prediction"]
            for i in range(n):
                first = np.moveaxis(clean_images[i], 0, -1)
                attacked = np.moveaxis(adv_images[i], 0, -1)
                visible_difference = np.clip(0.5 + (attacked - first) * 6, 0, 1)
                for j, data in enumerate((first, attacked, visible_difference)):
                    axes[i, j].imshow(data, interpolation="nearest")
                    axes[i, j].set_xticks([])
                    axes[i, j].set_yticks([])
                axes[i, 0].set_ylabel(CLASS_NAMES[int(truth[i])], fontsize=9)
                axes[i, 1].set_xlabel(f"Pred: {CLASS_NAMES[int(pred_adv[i])]}", fontsize=8)
            for ax, title in zip(axes[0], ("Clean original", "PGD adversarial", "6× amplified change")):
                ax.set_title(title, fontsize=10)
            fig.suptitle(f"{label}: clean-correct examples fooled by PGD (8/255)", fontsize=10)
            fig.savefig(FIGURES / f"pgd_examples_{model}.png", dpi=300, bbox_inches="tight")
            fig.savefig(FIGURES / f"pgd_examples_{model}.pdf", bbox_inches="tight")
            plt.close(fig)
    print("Saved figures to", FIGURES)


if __name__ == "__main__":
    make_figures()
