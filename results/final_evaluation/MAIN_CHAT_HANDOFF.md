# MAIN CHAT HANDOFF

**Workstream:** 03 Experiments | Controlled Evaluation & Figures
**Completed (UTC):** 2026-11-13T03:23:47.444598+00:00

## Completed and verified by saved evaluation records
- Frozen CompactCNN checkpoints evaluated on 10,000 identical CIFAR-10 test images (1,000 per class).
- No model training, fine-tuning, checkpoint updates, or architecture/defense changes in this workstream.
- Validation-only attack-strength probe preceded the recorded protocol lock and test evaluation.
- Untargeted model-specific white-box FGSM and PGD used 0, 2/255, 4/255, 8/255 pixel-space budgets.
- Attack outputs checked for image bounds and L-infinity perturbation limits in every batch.
- The result file contains observed counts, denominators, times and checkpoint hashes, not inferred accuracies.

## Actual held-out test results

| Model | Epsilon | Clean accuracy | FGSM robust accuracy | PGD robust accuracy |
|---|---:|---:|---:|---:|
| baseline | 0/255 | 8776/10000 (87.76%) | same as clean (zero perturbation) | same as clean (zero perturbation) |
| baseline | 2/255 | 8776/10000 (87.76%) | 1451/10000 (14.51%) | 282/10000 (2.82%) |
| baseline | 4/255 | 8776/10000 (87.76%) | 374/10000 (3.74%) | 0/10000 (0.00%) |
| baseline | 8/255 | 8776/10000 (87.76%) | 202/10000 (2.02%) | 0/10000 (0.00%) |
| fgsm_trained | 0/255 | 7003/10000 (70.03%) | same as clean (zero perturbation) | same as clean (zero perturbation) |
| fgsm_trained | 2/255 | 7003/10000 (70.03%) | 6293/10000 (62.93%) | 6262/10000 (62.62%) |
| fgsm_trained | 4/255 | 7003/10000 (70.03%) | 5505/10000 (55.05%) | 5377/10000 (53.77%) |
| fgsm_trained | 8/255 | 7003/10000 (70.03%) | 3943/10000 (39.43%) | 3396/10000 (33.96%) |

## Experimental configuration and computational cost
- Test images: 10,000; fixed stratified selection seed: 20261112; batch size: 64.
- Final PGD: 20 steps, 2 random restarts; step size: max(epsilon/4, 1/255); initializations uniform in the epsilon-ball.
- Local hardware: mps; Python: 3.14.4; PyTorch: 2.11.0.
- Cumulative measured evaluation condition runtimes: 556.66 seconds (not the same as wall-clock time for the full workflow).
- Prior verified Chat 02 training durations: baseline 395.72s; defended 632.41s.
- Protocol selection rationale: Validation on 500 stratified images showed identical accuracy for PGD20-r2 and PGD40-r2 at all budgets. PGD20-r2 was approximately twice as fast. Full CIFAR-10 test evaluation is computationally feasible.
- Validation diagnostic warnings recorded: none.

## Artifacts
- `results/final_evaluation/validation_probe.json`
- `results/final_evaluation/validation_probe.csv`
- `results/final_evaluation/protocol_locked.json`
- `results/final_evaluation/test_results.json`
- `results/final_evaluation/test_results.csv`
- `results/final_evaluation/paper_comparison_table.csv`
- `results/final_evaluation/validation_selected_indices.npy`
- `results/final_evaluation/test_selected_indices.npy`
- `results/final_evaluation/adversarial_examples.npz`
- `results/final_evaluation/figures/accuracy_vs_epsilon.png`
- `results/final_evaluation/figures/evaluation_runtime.png`
- `results/final_evaluation/figures/pgd_examples_baseline.png`
- `results/final_evaluation/figures/pgd_examples_fgsm_trained.png`

## Integrity / limitations / unresolved risks
- Results describe empirical resistance to the stated finite set of attacks, not certified robustness.
- PGD validation strength does not prove absence of all gradient masking. Review training-era PGD monitoring history for evidence of catastrophic overfitting.
- One fixed training seed and one frozen architecture limit generalizability.
- If the test set is a subset, clearly label its size in tables and paper; no full-test claim.
- Audit paper figure captions, raw counts, training logs and archived checkpoint hashes before submission.

## Next recommended action and deadline impact
- Send the final results and figures to Chat 05 QA for independent numerical and methodology review.
- Give Chat 04 Writing the recorded comparison table and validated limitation statements.
- Target November 19 technical freeze; Phase 1 due November 20; IEEE report due December 5.
