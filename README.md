# Evaluating Adversarial Attacks and Defense Mechanisms in Deep Neural Networks

DS 312 Machine Learning Final Project at Embry-Riddle Aeronautical University (Fall 2026)

## About the project

I wanted to see how much adversarial attacks could hurt a CNN's performance and whether adversarial training could make the model more resistant to those attacks.

I used CIFAR-10 and trained the same CNN architecture in two ways:

1. A standard CNN trained on clean images
2. A CNN trained with adversarial examples generated using FGSM with a random start

I then tested both models against FGSM and PGD using the same test images and perturbation limits. I implemented the attacks in PyTorch so I could control and check how the adversarial examples were generated.

The main tradeoff was clear: adversarial training lowered clean test accuracy from **87.76% to 70.03%**, but increased accuracy under the strongest PGD setting I tested (8/255) from **0.00% to 33.96%**.

## Dataset

I used [CIFAR-10](https://cave.cs.toronto.edu/kriz/cifar.html), which contains 60,000 RGB images of size 32 × 32 across 10 classes.

| Split | Images | How I used it |
|---|---:|---|
| Training | 45,000 | Train both models |
| Validation | 5,000 | Select checkpoints and set up the attack evaluation |
| Test | 10,000 | Evaluate both models after locking the evaluation settings |

I made a stratified training/validation split from CIFAR-10's original 50,000 training images using seed 42. This gave me 4,500 training and 500 validation images per class. I kept the original 10,000-image test set separate during development.

I used pixel values between 0 and 1, without mean/std normalization. For training, I applied random crops with padding 4 and random horizontal flips. I did not apply random augmentation to the validation or test images.

The split indices are saved in `splits/`. The dataset itself is not stored in this repository. Torchvision downloads it when I run:

```bash
python -m src.prepare_data
```

## Model and training

Both models use my `CompactCNN` with three convolutional blocks (32, 64, and 128 channels), batch normalization, ReLU, max pooling, global average pooling, and a final linear layer. The network has **288,746 trainable parameters**.

I kept the architecture and most training settings identical to make the comparison fair.

| Setting | Value |
|---|---|
| Epochs | 30 for each model |
| Batch size | 128 |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Weight decay | 0.0001 |
| Learning rate schedule | Cosine annealing |
| Loss | Cross-entropy |
| Random seed | 42 |

For the standard CNN, I selected the checkpoint with the highest clean validation accuracy (epoch 29).

For the defended CNN, I trained on adversarial images made with random-start, one-step projected FGSM. I used a training perturbation limit of **8/255** and step size of **10/255**. I selected the checkpoint using PGD-10 accuracy on the same balanced subset of 500 validation images, with clean validation accuracy as a tiebreaker (epoch 26).

The PGD training monitor used epsilon 8/255, step size 2/255, and no random start. I used it to monitor training, not as my final robustness result.

The selected model files are:

- `checkpoints/baseline_best.pt`
- `checkpoints/adversarial_best.pt`

## Attacks and final evaluation

I used untargeted, white-box attacks, meaning each attack calculated gradients from the exact model it was evaluating.

- **FGSM:** One gradient-sign step to increase classification loss
- **PGD:** Repeated gradient-sign steps, projecting the image back into the allowed perturbation range after each step

For the final evaluation, I used epsilon values **2/255, 4/255, and 8/255**. PGD used **20 steps**, **two random restarts**, and a step size of `max(epsilon/4, 1/255)`. The evaluation also kept track of the strongest successful intermediate PGD perturbation for each image.

I chose these settings using validation data before running the final test. Then I evaluated both frozen models on all **10,000 CIFAR-10 test images** using identical images and attack settings. The complete settings are saved in `results/final_evaluation/protocol_locked.json`.

## Results

The percentages below are **empirical robust accuracy**: an image counts as correct only if the model classified both the original clean image and the adversarial image correctly. This is slightly different from just measuring accuracy on the attacked images.

| Test condition | Standard CNN | FGSM-trained CNN |
|---|---:|---:|
| Clean | 87.76% | 70.03% |
| FGSM, 2/255 | 14.51% | 62.93% |
| FGSM, 4/255 | 3.74% | 55.05% |
| FGSM, 8/255 | 2.02% | 39.43% |
| PGD-20 (2 restarts), 2/255 | 2.82% | 62.62% |
| PGD-20 (2 restarts), 4/255 | 0.00% | 53.77% |
| PGD-20 (2 restarts), 8/255 | 0.00% | 33.96% |

At 8/255, the standard model was almost completely vulnerable to FGSM and PGD. The defended model performed much better against both attacks, although it lost **17.73 percentage points** of clean accuracy.

Adversarial training also took longer on my Mac: **632.41 seconds**, compared with **395.72 seconds** for standard training and validation (about **59.8% longer**).

I saved the complete test results, comparison table, evaluation settings, and figures in `results/final_evaluation/`.

## Running the project

I ran the experiments on an Apple M2 MacBook Pro with 8 GB unified memory, macOS 27.0.1, Python 3.14.4, PyTorch 2.11.0, and Torchvision 0.26.0 using the MPS backend. I saved the installed dependency versions in `requirements.txt`.

From the project directory, I can set up the environment with:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -m src.prepare_data
```

The training and checkpoint-verification scripts were tested on my Mac and currently require MPS. The final evaluation code supports MPS, CUDA, or CPU with compatible PyTorch installations, although results and runtimes can vary across hardware.

### Checking the saved results

I included the selected checkpoints and the original evaluation records so the reported results can be checked without running training or another test evaluation.

```bash
python -m scripts.verify_frozen_artifacts
python -m scripts.check_checkpoint_loading
python -m scripts.check_comment_style
python -m compileall -q src scripts
```

These checks compare checkpoint SHA-256 hashes with the locked evaluation protocol, validate the training histories and saved splits, reconcile the recorded results, and check that the model weights load into the correct architecture. They **do not** rerun inference on the test set.

The publicly archived evaluation JSON has a small descriptive-metadata update. I kept the original and archived validation-file checksums in the locked protocol. The predictions, accuracy values, attack settings, model checkpoints, and image indices did not change.

### Training from scratch

For someone who wants to reproduce training, these are the scripts I used, in order:

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

Run these in a **separate working copy**. I made the training scripts refuse to overwrite the saved experiment files, and I do not want to replace the checkpoints used for the results above. The included `*_best.pt` files are the selected models; I did not include the larger `*_last.pt` training-state files because they are unnecessary for the final comparison.

For safety, only load checkpoints from sources you trust. Even with a fixed seed, a fresh training run may not reproduce every number exactly on a different device or library version.

### Creating the figures

The figures in `results/final_evaluation/figures/` include the accuracy comparison, evaluation time, and PGD examples for each model.

To regenerate the plots from the **saved results**, without rerunning attacks, use a separate working copy and run:

```bash
python -m src.plot_final_results
```

The final evaluation code is in `src/evaluate_final.py`. I locked the settings before using the test set, and the script prevents repeating the test evaluation when the saved test record already exists. I would only repeat that evaluation if I found a real error in the original run.

## Repository structure

```text
src/                      Model, data preparation, attacks, training, and evaluation
scripts/                  Checks for saved results, model loading, and comment style
splits/                   Saved training, validation, and monitoring indices
checkpoints/              Best standard and adversarially trained models
logs/                     Training settings, histories, and verification results
results/final_evaluation/ Final numbers, evaluation settings, tables, and figures
requirements.txt          Python dependencies
```

## Limitations

I measured how well the models performed against the specific FGSM and PGD attacks I tested. **These numbers are not a guarantee of robustness** against every possible adversarial attack.

- I used one dataset, one CNN architecture, and one training seed
- The PGD-10 training monitor did not show a sudden collapse in robustness, but it cannot rule out all forms of catastrophic overfitting or gradient masking
- Stronger attacks could reduce the reported adversarial accuracy
- I picked the example images in the figures because PGD successfully fooled the model on them; they are illustrations, not a random sample
- The enlarged perturbation-difference figures are for visualization and do not show valid adversarial images by themselves
- I saved the exact models and settings, but identical retraining results cannot be guaranteed across devices and software versions

## Code checks

I also checked the Python files with Ruff, compilation, and a small script that checks my comment formatting:

```bash
python -m pip install ruff
python -m ruff check --select E,F,W --ignore E501 src scripts
python -m scripts.check_comment_style
python -m compileall -q src scripts
```

I exclude Ruff's E501 long-line rule because some of my scripts go over 88 characters. I also leave automated formatting off so it does not change how I write comments.
