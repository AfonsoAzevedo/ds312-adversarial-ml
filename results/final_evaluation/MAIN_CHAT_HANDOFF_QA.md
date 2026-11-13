# MAIN CHAT HANDOFF

**Workstream:** 03 Experiments | Controlled Evaluation, Figures and Independent QA  
**Date:** November 12, 2026 (America/Phoenix; final test artifacts timestamped November 13 UTC)  
**Status:** Core final evaluation complete; numerical/code/figure audit passed with one minor plotting fix. Review existing training history before closing the catastrophic-overfitting check.

## Completed and verified

- Reviewed the DS 312 project assignment and locked scope: CIFAR-10, one CompactCNN architecture (288,746 parameters), ordinary vs FGSM-adversarial training, white-box FGSM and PGD; no retraining or new model introduced.
- Ran (per recorded Mac output) a 500-image stratified validation probe first. Selected PGD-20 with two random starts based on validation attack strength and runtime; PGD-40 did not improve validation accuracy at these tested budgets.
- The protocol was locked before opening the test dataset; validation and checkpoint SHA-256 hashes are recorded in `protocol_locked.json`.
- Completed final evaluation (per archived logs) on **all 10,000 CIFAR-10 test images**, exactly once per model/attack/budget condition, with 1,000 per class and matching selected indices.
- Independently inspected `src/evaluate_final.py`, `src/attacks.py`, and `src/model.py`: true-label, untargeted, model-specific input gradients; FGSM and PGD projection/clamp; two PGD random starts, 20 iterations and strongest per-image intermediate; both checkpoints loaded with `strict=True` and eval mode, without training steps.
- Independently recomputed every recorded accuracy from integer counts; verified JSON/CSV/table consistency, sample counts, recorded perturbation maxima, selection indices, lock-file checksum, and SHA agreement between validation and test checkpoint metadata.
- Audited the four PNG figures and all 12 saved adversarial examples. Corrected the accuracy-plot x-axis label and legend in a plot-only source/figure update; test/validation records and models are unchanged.

## Actual test results (robust metric = clean-correct AND adversarial-correct)

| Attack | Epsilon | Standard CNN | FGSM-trained CNN |
|---|---:|---:|---:|
| Clean | 0 | 8,776/10,000 = **87.76%** | 7,003/10,000 = **70.03%** |
| FGSM | 2/255 | 1,451/10,000 = 14.51% | 6,293/10,000 = 62.93% |
| FGSM | 4/255 | 374/10,000 = 3.74% | 5,505/10,000 = 55.05% |
| FGSM | 8/255 | 202/10,000 = 2.02% | 3,943/10,000 = 39.43% |
| PGD-20, two restarts | 2/255 | 282/10,000 = 2.82% | 6,262/10,000 = 62.62% |
| PGD-20, two restarts | 4/255 | 0/10,000 = 0.00% | 5,377/10,000 = 53.77% |
| PGD-20, two restarts | 8/255 | 0/10,000 = 0.00% | 3,396/10,000 = 33.96% |

Ordinary attacked accuracy differs slightly from the strict robust metric for **two examples** in baseline FGSM at 8/255 (204 vs. 202) and **two examples** in defended FGSM at 2/255 (6,295 vs. 6,293). Use the strict robust metric in primary tables/plots.

## Observed trade-offs and configuration

- The adversarially trained model has **17.73 percentage points lower clean accuracy** but **33.96 percentage points higher PGD robust accuracy** at epsilon=8/255.
- CIFAR-10 `ToTensor()` pixels in `[0,1]`; budgets 0,2/255,4/255,8/255; PGD step size `max(epsilon/4, 1/255)`; 20 steps; 2 uniform random restarts; max-CE success-preferring intermediate selection; batch size 64.
- Sample-selection seed 20261112; underlying setup seed 42; MPS device on an Apple Mac; recorded Python 3.14.4 and PyTorch 2.11.0.
- Total measured evaluation-condition runtime **556.6614 s** (9.28 min), excluding separate plotting and other workflow time.
- **Prior Chat 02 reported training times** 395.72 s (baseline) and 632.41 s (FGSM-trained, +59.81%); not independently verified in this QA archive because historical logs were not provided.
- Checkpoint hashes: `baseline_best.pt` `e6967650f06f5154626273a3f717a0bca7a1d3a6a60e2568eff6d7fc26b045b7`; `adversarial_best.pt` `069bf0e491b2d7d599dd732cb3402e106e944b3e03871a01df5bb11fcc56406a`.

## Artifacts

- Original locked input/protocol: `results/final_evaluation/validation_probe.json`, `protocol_locked.json`, `test_results.json`, and corresponding CSV files and selected-index arrays.
- Source: `src/evaluate_final.py`, `src/attacks.py`, `src/model.py`, corrected `src/plot_final_results.py`.
- Tables: `results/final_evaluation/paper_comparison_table.csv`.
- Figures: `results/final_evaluation/figures/accuracy_vs_epsilon.{png,pdf}` (corrected label/legend), `evaluation_runtime.{png,pdf}`, `pgd_examples_baseline.{png,pdf}`, `pgd_examples_fgsm_trained.{png,pdf}`.
- Illustrative data: `results/final_evaluation/adversarial_examples.npz`.
- Independent audit: `results/final_evaluation/QA_REVIEW.md` (copy of audit report in QA package).

## Limitations, risks and unresolved issues

- **Not certified robustness:** Accuracy represents empirical resistance to the stated white-box finite attacks; stronger attacks could reduce it.
- **Catastrophic-overfitting/gradient-masking review pending:** Validation PGD strengthening and PGD-vs-FGSM checks are reassuring but cannot establish absence. Inspect pre-existing adversarial-training history across epochs. No more test-set tuning or rerun.
- **Model/training provenance:** Frozen checkpoint hash agreement is verified in records, but the actual checkpoint bytes and training logs were not included in the QA ZIP. Treat training times as previously reported, not newly independently measured.
- One architecture, one training seed, one evaluation dataset, finite steps/restarts. The adversarial examples are deliberately selected attack successes, not random representative test samples; caption accordingly.
- Dataset link/hosting arrangement under professor's instructions is a separate outstanding submission requirement.

## Recommended MAIN actions and deadlines

1. Record 03 Experiments as **core evaluation completed and numerically QA-checked**, with **existing training-log review** still open.
2. Transfer the frozen comparison table, corrected figures and verified limitations to Chat 04 Writing; do not add more architectures, defenses, training or test evaluations without MAIN approval.
3. In Chat 05 QA, check pre-existing adversarial-training history for catastrophic overfitting, then finish final repository/reproducibility/submission audit.
4. Protect November 19, 2026 technical freeze; submit Phase 1 November 20 and 4-6 page IEEE report plus code/dataset links by December 5.
