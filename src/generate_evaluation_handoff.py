"""Build the MAIN CHAT HANDOFF from actual saved DS312 test records only."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "results" / "final_evaluation"


def main():
    validation = FOLDER / "validation_probe.json"
    locked = FOLDER / "protocol_locked.json"
    test = FOLDER / "test_results.json"
    if not all(p.exists() for p in (validation, locked, test)):
        raise FileNotFoundError("Cannot generate final handoff before validation, protocol lock, and test are recorded")
    v, p, t = [json.loads(f.read_text()) for f in (validation, locked, test)]
    rows = t["records"]
    n = t["count"]
    out = [
        "# MAIN CHAT HANDOFF", "", "**Workstream:** 03 Experiments | Controlled Evaluation & Figures",
        f"**Completed (UTC):** {t['completed_at_utc']}",
        "", "## Completed and verified by saved evaluation records",
        f"- Frozen CompactCNN checkpoints evaluated on {n:,} identical CIFAR-10 test images ({n//10:,} per class).",
        "- No model training, fine-tuning, checkpoint updates, or architecture/defense changes in this workstream.",
        "- Validation-only attack-strength probe preceded the recorded protocol lock and test evaluation.",
        "- Untargeted model-specific white-box FGSM and PGD used 0, 2/255, 4/255, 8/255 pixel-space budgets.",
        "- Attack outputs checked for image bounds and L-infinity perturbation limits in every batch.",
        "- The result file contains observed counts, denominators, times and checkpoint hashes, not inferred accuracies.",
        "", "## Actual held-out test results", "",
        "| Model | Epsilon | Clean accuracy | FGSM robust accuracy | PGD robust accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in ("baseline", "fgsm_trained"):
        clean = next(r for r in rows if r["model"] == model and r["attack"] == "clean")
        for eps in (0, 2, 4, 8):
            if eps:
                fgsm = next(r for r in rows if r["model"] == model and r["attack"] == "fgsm"
                            and r["epsilon_numerator"] == eps)
                pgd = next(r for r in rows if r["model"] == model and r["attack"] == "pgd"
                           and r["epsilon_numerator"] == eps)
                fstr = f"{fgsm['robust_correct']}/{n} ({fgsm['empirical_robust_accuracy_percent']:.2f}%)"
                pstr = f"{pgd['robust_correct']}/{n} ({pgd['empirical_robust_accuracy_percent']:.2f}%)"
            else:
                fstr = pstr = "same as clean (zero perturbation)"
            out.append(f"| {model} | {eps}/255 | {clean['correct']}/{n} "
                       f"({clean['accuracy_percent']:.2f}%) | {fstr} | {pstr} |")
    total = sum(r["runtime_seconds"] for r in rows)
    out.extend([
        "", "## Experimental configuration and computational cost",
        f"- Test images: {n:,}; fixed stratified selection seed: {p['selection_seed']}; "
        f"batch size: {p['batch_size']}.",
        f"- Final PGD: {p['pgd_steps']} steps, {p['pgd_restarts']} random restarts; "
        f"step size: {p['alpha_rule']}; initializations uniform in the epsilon-ball.",
        f"- Local hardware: {t['device']}; Python: {t['python']}; PyTorch: {t['torch']}.",
        f"- Cumulative measured evaluation condition runtimes: {total:.2f} seconds "
        "(not the same as wall-clock time for the full workflow).",
        f"- Prior verified Chat 02 training durations: baseline "
        f"{t['prior_training_times_seconds']['baseline']:.2f}s; defended "
        f"{t['prior_training_times_seconds']['fgsm_trained']:.2f}s.",
        f"- Protocol selection rationale: {p['decision_rationale']}",
        f"- Validation diagnostic warnings recorded: {v['validation_warnings'] or 'none'}.",
        "", "## Artifacts",
    ])
    for item in ["validation_probe.json", "validation_probe.csv", "protocol_locked.json",
                 "test_results.json", "test_results.csv", "paper_comparison_table.csv",
                 "validation_selected_indices.npy", "test_selected_indices.npy",
                 "adversarial_examples.npz"]:
        path = FOLDER / item
        if path.exists():
            out.append(f"- `results/final_evaluation/{item}`")
    figures = FOLDER / "figures"
    if figures.exists():
        for path in sorted(figures.glob("*.png")):
            out.append(f"- `results/final_evaluation/figures/{path.name}`")
    out.extend([
        "", "## Integrity / limitations / unresolved risks",
        "- Results describe empirical resistance to the stated finite set of attacks, not certified robustness.",
        "- PGD validation strength does not prove absence of all gradient masking. "
        "Review training-era PGD monitoring history for evidence of catastrophic overfitting.",
        "- One fixed training seed and one frozen architecture limit generalizability.",
        "- If the test set is a subset, clearly label its size in tables and paper; no full-test claim.",
        "- Audit paper figure captions, raw counts, training logs and archived checkpoint hashes before submission.",
        "", "## Next recommended action and deadline impact",
        "- Send the final results and figures to Chat 05 QA for independent numerical and methodology review.",
        "- Give Chat 04 Writing the recorded comparison table and validated limitation statements.",
        "- Target November 19 technical freeze; Phase 1 due November 20; IEEE report due December 5.",
    ])
    output = FOLDER / "MAIN_CHAT_HANDOFF.md"
    if output.exists():
        raise FileExistsError("Handoff already exists; review before overwriting")
    output.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("Saved", output)


if __name__ == "__main__":
    main()
