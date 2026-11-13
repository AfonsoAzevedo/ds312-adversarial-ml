# DS 312 | Independent Controlled-Evaluation QA Review

**Review basis:** User-provided `ds312_final_eval_qa.zip`, including Python source, validation/protocol/test JSON and CSV, saved indices, adversarial examples, and PNG/PDF figure exports. **Audit date:** November 12, 2026 (America/Phoenix), November 13 UTC.

## Overall judgment

**Numerical and recorded-methodology review: PASS, with minor figure corrections and one outstanding historical-training check.** The 10,000-image final evaluation does not need rerunning. The models were not supplied in the QA archive, so checkpoint inference was not independently repeated; code was inspected statically and recorded data were recomputed/checksummed.

## Independent integrity checks

- **Records:** 14 test conditions (7 per model), each evaluated on `n=10,000`; 26 validation conditions (13 per model), each on `n=500`.
- **Test coverage:** `test_selected_indices.npy` contains each integer 0 through 9,999 exactly once, sorted; results record 1,000 images in each of the ten classes.
- **Validation chronology:** validation completed at `2026-11-13T03:11:04Z`, protocol locked at `03:13:48Z`, test began at `03:14:28Z`, and test ended at `03:23:47Z`. Validation JSON's SHA-256 exactly matches `protocol_locked.json`.
- **Checkpoint integrity record:** Both model SHA-256 values agree across validation, lock, and final results. The checkpoints themselves were not included in the ZIP, preventing independent verification of those files' contents.
- **Numerical validation:** Every `accuracy_percent` equals `100 * correct / n`, and every `empirical_robust_accuracy_percent` equals `100 * robust_correct / n`. CSV values agree with JSON, and the paper comparison table agrees with robust-accuracy values.
- **Perturbation constraints:** For all recorded conditions, maximum observed L-infinity perturbations do not exceed `epsilon + 2e-6` in `[0,1]` pixel space. The code asserts pixel range and budget for every evaluation batch.
- **Model-specific white-box attacks:** `evaluate_final.py` loads the two frozen model state dictionaries, uses eval mode, disables weight gradients, obtains input gradients with autograd, and computes FGSM/PGD separately for each model. No training/optimizer step or checkpoint-writing code is present in this evaluation module.
- **PGD evaluator:** Untargeted cross-entropy attack; 20 gradient-ascent steps per restart; two independent uniform-start samples per batch; projection to the L-infinity ball and valid pixel box each iteration; per-image selection prefers successful perturbations and then higher cross-entropy, including intermediate iterations. The PGD used for the **final evaluation** is `pgd_attack_evaluation()` in `src/evaluate_final.py`, not the shorter training/development PGD function in `src/attacks.py`.
- **Examples:** Six examples per model; original prediction equals true label and PGD final prediction differs; all stored pixels are in `[0,1]` and observed L-infinity changes respect 8/255.
- **Cumulative evaluation time:** The sum of the 14 recorded per-condition runtimes is **556.6614 seconds**, excluding unrelated setup, plotting and file management.

## Final held-out test results, empirical robust accuracy

| Condition | Standard CompactCNN | FGSM-trained CompactCNN |
|---|---:|---:|
| Clean, epsilon=0 | 8,776/10,000 = **87.76%** | 7,003/10,000 = **70.03%** |
| FGSM, epsilon=2/255 | 1,451/10,000 = 14.51% | 6,293/10,000 = 62.93% |
| FGSM, epsilon=4/255 | 374/10,000 = 3.74% | 5,505/10,000 = 55.05% |
| FGSM, epsilon=8/255 | 202/10,000 = 2.02% | 3,943/10,000 = 39.43% |
| PGD-20, two restarts, epsilon=2/255 | 282/10,000 = 2.82% | 6,262/10,000 = 62.62% |
| PGD-20, two restarts, epsilon=4/255 | 0/10,000 = 0.00% | 5,377/10,000 = 53.77% |
| PGD-20, two restarts, epsilon=8/255 | 0/10,000 = 0.00% | 3,396/10,000 = 33.96% |

**Metric convention:** Robust correctness requires a correct prediction both before and after the attack. Ordinary attacked-image accuracy differs by two images in baseline FGSM at 8/255 (204 vs. 202) and defended FGSM at 2/255 (6,295 vs. 6,293). Use empirical robust accuracy consistently in the paper's primary comparison table and curves.

**Observed trade-off:** FGSM adversarial training loses **17.73 percentage points** of clean test accuracy but increases PGD empirical robust accuracy at epsilon=8/255 by **33.96 percentage points**. The model's recorded training time rises from 395.72 s to 632.41 s (+59.81%), according to the previous Chat 02 training handoff; **these training-time values could not be independently revalidated without its training logs**.

## Visualization review

1. `accuracy_vs_epsilon` originally had a misleading `epsilon / 255` x-axis label and a legend overlapping the axis area. A corrected source script and regenerated plot are supplied in the companion fix package. The corrected x-axis explicitly states `epsilon=k/255` and tick labels show k=0,2,4,8.
2. `evaluation_runtime` shows correctly recorded time for each 10,000-image attack condition at epsilon=8/255. Interpret this as condition-level evaluation wall time on the reported Mac, **not as hardware-independent attack-generation complexity**.
3. Both `pgd_examples_*` files correctly show six clean-correct examples that become misclassified at epsilon=8/255. The third column displays **`clip(0.5 + 6 * (adversarial-clean), 0, 1)`** for visualization; it is not a valid original-range perturbation image and must be captioned as an amplified, recentered change.
4. PNG images were visually inspected. PDF exports are supplied as files but were not independently compared pixel-by-pixel with printed or embedded IEEE layouts.

## Remaining scientifically important limitations

- The saved validation probe found no heuristic warnings: PGD-20 was at least as strong as FGSM on its 500 validation images and PGD-40 did not improve **measured accuracy** beyond PGD-20 for two starts. That is evidence against a trivially weak PGD evaluation, **not proof of absent gradient masking**.
- Training-era PGD metrics across epochs and any monitored catastrophic-overfitting signals were not included. **Review the pre-existing adversarial-training history before asserting the absence of catastrophic overfitting**; no new training or final-test rerun is requested.
- The archive lacks the two checkpoint files and full training logs, so model weights, training seed and training-time provenance cannot all be independently confirmed here. Model state-dict loading, hashes, and metadata are recorded correctly by the evaluation code.
- Results apply to this architecture, saved training run, test set and stated finite attacks. They are **empirical**, not certified robustness; they do not support broad state-of-the-art claims or causal generalization to other models/seeds.
- A separate course-deliverable issue remains: confirm the dataset-link/hosting method with MAIN under the instructor's PDF instructions before final submission.

## Disposition and next actions

1. **Use corrected plotting code and accuracy figure**. Do not modify the checkpoints or rerun the held-out test.
2. Review the existing adversarial-training logs/checkpoint history for catastrophic overfitting, then close the one unresolved scientific QA item.
3. Transfer `MAIN_CHAT_HANDOFF_QA.md`, recorded results, corrected figures and this audit report to MAIN and Chat 04 Writing. Retain the original `test_results.json`, indices, protocol and checkpoint hashes unchanged.
4. Continue repository/documentation QA by the internal November 19 deadline and submit Phase 1 by November 20, 2026; final IEEE report remains due December 5, 2026.
