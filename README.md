# Evaluating Adversarial Attacks and Defense Mechanisms in Deep Neural Networks

DS 312 Machine Learning final project, Embry-Riddle Aeronautical University (Fall 2026)

## Project overview

This project studies how untargeted FGSM and PGD attacks change CIFAR-10 image classification accuracy, and whether FGSM-based adversarial training improves resistance at a cost to clean accuracy and computation. It compares one 288,746-parameter `CompactCNN` architecture trained in two ways: standard supervised training and random-start one-step projected FGSM adversarial training. Gradient attacks are implemented directly in PyTorch.

The research results are empirical for the stated attacks. They **do not** provide certified robustness.

## Dataset

- **CIFAR-10:** 60,000 color images, 32 × 32 pixels, 10 classes
- Original training partition: 50,000 images, split into **45,000 training** and **5,000 validation** images using stratification and random seed 42
- Original test partition: **10,000 images**, reserved until the evaluation protocol was locked
- Input preparation: RGB tensors in **[0, 1]**, with no mean/std normalization
- Training-only augmentation: random crop with padding 4 and random horizontal flip
- Fixed splits: `splits/train_indices.npy`, `splits/val_indices.npy` and `splits/adv_monitor_indices.npy`

The original dataset is available from the [University of Toronto CIFAR-10 page](https://cave.cs.toronto.edu/kriz/cifar.html). Running `python -m src.prepare_data` downloads it using Torchvision if it is absent. **Do not commit the downloaded dataset to GitHub.** The instructor separately requests a dataset link hosted on the ERAU web account; confirm the acceptable hosting/link arrangement before final submission. The official dataset URL alone is not a verified substitute for that requirement.

## Environment

The original experiments were performed on an Apple M2 MacBook Pro with 8 GB unified memory, macOS 27.0.1, Python **3.14.4**, PyTorch **2.11.0**, Torchvision **0.26.0**, using PyTorch MPS acceleration. Dependency versions from the verified environment are pinned in `requirements.txt`.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -m src.prepare_data
```

Run commands from the repository root. On Windows, activate the virtual environment using the appropriate platform-specific command. Torchvision is used for the dataset download and transformations. The existing **training and checkpoint verification scripts require MPS** and were verified only on the recorded Mac. The final evaluation module supports MPS, CUDA, or CPU, subject to compatible PyTorch installations and hardware-specific numerical differences.

## Reproducing the archived results without re-running the test

The repository includes the selected model checkpoints and frozen evaluation records. From the repository root, after downloading CIFAR-10, use the read-only integrity check:

```bash
python -m scripts.verify_frozen_artifacts
python -m scripts.check_checkpoint_loading
python -m scripts.check_comment_style
python -m compileall -q src scripts
```

`check_checkpoint_loading` safely loads the verified local state dictionaries into fresh CPU models without dataset access or test inference. `verify_frozen_artifacts` checks the selected checkpoint SHA-256 values against the locked protocol, verifies training histories and selected epochs, reconciles final test metrics with CSV and JSON, and checks the fixed split indices. **It does not run neural-network inference, retrain, or re-evaluate the test set.** Checkpoints are loaded only as raw bytes for hashing in this audit.

Best checkpoints (kept in the repository):

- `checkpoints/baseline_best.pt`, epoch 29
- `checkpoints/adversarial_best.pt`, epoch 26

The `*_last.pt` files contain optimizer and scheduler state and are intentionally not included in Git because they are not needed for the reported comparisons. Model files must be treated as trusted local research artifacts, not arbitrary checkpoints obtained from unknown sources.

## Methods and training protocol

The same CompactCNN architecture, 45,000-image training split, 5,000-image validation split, seed 42, AdamW optimizer, initial learning rate 0.001, weight decay 0.0001, batch size 128, cosine annealing, and 30-epoch budget are used for both training conditions.

- **Standard model:** ordinary cross-entropy training; select the checkpoint with highest clean validation accuracy
- **Defended model:** randomly initialized one-step projected FGSM with training epsilon 8/255 and step size 10/255; select the checkpoint with the highest PGD-10 accuracy on a fixed, balanced **500-image validation monitor**, breaking ties by full clean validation accuracy
- **Monitor:** epsilon 8/255, PGD-10, step size 2/255, no random start; this was a development-time monitor, not the final robustness evaluation

Training scripts intentionally refuse to overwrite archived outputs. **Do not run training from this completed results directory.** For a separate reproduction in an isolated working copy that has no archived training outputs, the intended order is:

```bash
python -m src.prepare_data
python -m src.benchmark
python -m src.train_baseline
python -m src.verify_baseline
python -m src.test_fgsm
python -m src.test_pgd
python -m src.train_adversarial --smoke-test
python -m src.train_adversarial
python -m src.verify_adversarial
```

These training and verification commands require MPS as implemented. Stochastic training can differ slightly across PyTorch devices and environments even with fixed seeds. The submitted models and metrics refer to the **original archived run**, not to a guaranteed bit-for-bit retraining result. Do not delete the archived outputs merely to rerun experiments.

## Final evaluation and results

The evaluation protocol was chosen using validation data and saved to `results/final_evaluation/protocol_locked.json` **before** the held-out test evaluation. Both frozen models were evaluated using identical CIFAR-10 test images and input-space epsilon budgets 0, 2/255, 4/255, and 8/255.

The final white-box PGD uses **20 steps**, **two uniform random starts**, step size `max(epsilon/4, 1/255)`, and per-image selection of the strongest successful intermediate iterate. The final evaluation PGD is in `src/evaluate_final.py`, while training/development PGD is in `src/attacks.py`.

The primary table reports **empirical robust accuracy**, requiring the image to be classified correctly **both before and after the attack**. This can differ slightly from attacked-image accuracy.

| Condition | Standard CNN | FGSM-trained CNN |
|---|---:|---:|
| Clean | 87.76% | 70.03% |
| FGSM, 2/255 | 14.51% | 62.93% |
| FGSM, 4/255 | 3.74% | 55.05% |
| FGSM, 8/255 | 2.02% | 39.43% |
| PGD-20 × 2, 2/255 | 2.82% | 62.62% |
| PGD-20 × 2, 4/255 | 0.00% | 53.77% |
| PGD-20 × 2, 8/255 | 0.00% | 33.96% |

All values above are archived results from **10,000 test images**, not newly calculated model predictions. Clean test accuracy falls by **17.73 percentage points** with adversarial training, while PGD empirical robust accuracy at 8/255 rises by **33.96 percentage points**. Recorded training-and-validation times were **395.72 s** for the baseline and **632.41 s** for the defended CNN on the original MPS hardware.

The full raw evaluation records and provenance are stored in `results/final_evaluation/test_results.json`, `test_results.csv`, `validation_probe.json`, and `protocol_locked.json`. Tables and original figures are in the same directory.

**Public-release metadata note:** The training-time provenance wording in the archived validation and test JSON files was standardized after the final evaluation. The locked protocol records the original validation-file SHA-256 and the checksum of the publicly archived version. No predictions, measured accuracies, attack settings, model checkpoints, or dataset indices were changed. To regenerate figures **from the existing frozen JSON, without running model inference**, run `python -m src.plot_final_results` in a separate copy of the repository. This writes figures and the comparison CSV in that copy.

`src.evaluate_final` records the validation-first, locked protocol and one-time final evaluation workflow. Its `test` command intentionally refuses a repeated test evaluation when `test_results.json` exists. **Do not remove the locked records or re-run the frozen test from this submitted repository.** An independent evaluator can examine the source and original records without opening the test set again.

## Project structure

```text
src/                      Model, attacks, data loaders, training and evaluation
scripts/                  Read-only artifact and coding-style verification
splits/                   Fixed train/validation and monitoring indices
checkpoints/              Selected baseline and defended model weights
logs/                     Training configurations, histories and verification logs
results/final_evaluation/ Locked protocol, raw results, tables and figures
requirements.txt          Observed Python dependency versions
```

Key source entry points: `src/prepare_data.py`, `src/model.py`, `src/data.py`, `src/attacks.py`, `src/train_baseline.py`, `src/train_adversarial.py`, `src/evaluate_final.py`, and `src/plot_final_results.py`.

## Limitations and reproducibility notes

- PGD/FGSM results are **empirical**, not certified robustness and not a guarantee against stronger or different attacks
- The training-history check found **no observed PGD-10 monitoring collapse**, but its 500-image, no-random-start monitor cannot rule out catastrophic overfitting under all attacks
- Only one architecture, one training seed and one dataset were studied
- The model was evaluated once under the validation-locked test protocol, and the test set was not used for checkpoint selection
- Adversarial example figures deliberately depict clean-correct images fooled by PGD; the amplified difference panels are visualizations, not valid input-range perturbation images
- Selected checkpoints and evaluation metadata are checked with SHA-256, and Python package versions are pinned, but bitwise reproducibility is not guaranteed across hardware and libraries

## Course submission

The instructor requires an IEEE conference-style PDF report (4–6 pages excluding references), a GitHub code link, dataset-access link and execution instructions. The status report is due November 20, 2026; the final submission is due December 5, 2026. The dataset-hosting plan still requires confirmation with MAIN and, if necessary, the instructor.

## Optional source-quality checks and GitHub publication

For read-only lint and comment-style checks (after installing Ruff):

```bash
python -m pip install ruff
python -m ruff check --select E,F,W --ignore E501 src scripts
python -m scripts.check_comment_style
python -m compileall -q src scripts
```

The Ruff check excludes long-line rule E501 because some recorded scripts retain lines longer than 88 characters. **Do not run `ruff format` on this project:** it inserts spaces after `#` and conflicts with the required Python comment convention. All reported lint and comment checks passed when the repository was published.
