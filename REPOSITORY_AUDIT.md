# DS 312 | Source and Repository Audit

**Audit date:** November 12, 2026 (America/Phoenix)

**Basis:** User-supplied `ds312_code_audit.zip` containing 51 files, including 14 Python implementation scripts, training configurations, histories, validation and final test results, fixed split arrays, and final figures. Model checkpoints were **not included** in this archive.

## Verified checks

- Full adversarial training history contains 30 sequential epochs and selected epoch 26 under the documented lexicographic PGD-monitor, then clean-accuracy rule
- Adversarial training monitor reaches 36.40% at epoch 26 and ends at 35.80% at epoch 30; last ten epochs range from 34.60% to 36.40%
- No observed catastrophic collapse under this **500-image, PGD-10, no-random-start** monitoring protocol; this does not rule out catastrophic overfitting under all stronger evaluations
- Baseline selected epoch 29 at 88.32% clean validation accuracy; defended selected epoch 26 at 70.10% clean validation accuracy and 36.40% PGD-10 monitor accuracy
- Locked final protocol contains 10,000 test images, PGD-20, two random starts, and epsilon numerators 0, 2, 4 and 8
- The archive contains 14 final test conditions with integer counts that reconcile with the recorded accuracy metrics and CSV
- The archived validation-probe SHA-256 matches the SHA-256 recorded in the locked protocol
- Checkpoint hashes recorded in validation, locked protocol and final results agree with one another; actual model-file bytes were unavailable for independent hashing
- Saved training and validation arrays cover all original 50,000 training indices exactly once; final saved test indices cover all 10,000 positions exactly once
- Source comment conventions were normalized in `src/evaluate_final.py` and `src/plot_final_results.py` (24 comments changed). Their parsed Python ASTs remained identical, and all source files passed compilation checks
- A read-only verification script passed the source-only archive check with `--skip-checkpoints`, explicitly reporting the checkpoints as not verified

## Reproducibility and repository actions prepared

- `README.md` documents the dataset, environment, training protocols, exact frozen results, checkpoint handling, execution commands, and limitations
- `requirements.txt` pins five packages at the versions reported from the verified Mac environment
- `.gitignore` excludes data, virtual environments, operating system artifacts, archives and unneeded last-epoch optimizer checkpoints while retaining the two selected model checkpoints
- `scripts/verify_frozen_artifacts.py` verifies record integrity and compares actual model-file SHA-256 values to the locked protocol without using test inference
- `scripts/check_comment_style.py` checks the user's required comment conventions

## Pending checks

1. Run the artifact verifier on the user's Mac with **actual selected checkpoint files present**, without the `--skip-checkpoints` flag
2. Complete formatting of long Python lines where feasible (82 lines over 88 characters remain in the supplied source and scripts). This is a readability/style item, not evidence of an experimental defect
3. Initialize Git, inspect `git status` and `git diff --check`, commit only intended files and verify a GitHub remote push
4. Confirm the final dataset-link or ERAU hosting arrangement against the professor's assignment
5. Verify package installation on any alternate operating system if cross-platform execution is required; the original training scripts explicitly require Apple MPS

## Research integrity

No retraining or held-out test evaluation was performed as part of this audit. The existing `test_results.json`, locked protocol, checkpoint contents, training histories and figure data were not modified in the supplied patch package. The Python modifications are comment-only, with unchanged parsed program structures.
