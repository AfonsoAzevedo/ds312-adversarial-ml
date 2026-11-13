import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.torch_version import TorchVersion

from src.attacks import fgsm_attack, pgd_attack
from src.data import create_data_loaders
from src.model import CompactCNN


EPSILON_VALUES = (0.0, 2 / 255, 4 / 255, 8 / 255)
STEPS = 10
TOLERANCE = 1e-6


def check_synthetic_steps() -> dict[str, bool | float]:
    #Use a linear classifier with a known direction of attack
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(48, 2, bias=False, dtype=torch.float64),
    )
    with torch.no_grad():
        model[1].weight[0].fill_(0.02)
        model[1].weight[1].fill_(-0.02)

    images = torch.full((1, 3, 4, 4), 0.5, dtype=torch.float64)
    labels = torch.tensor([0], dtype=torch.long)
    epsilon = 2 / 255
    step_size = epsilon / 4

    #After six ascent steps the projected attack reaches the boundary
    adversarial = pgd_attack(
        model, images, labels, epsilon, step_size, steps=6,
        random_start=False,
    )
    expected = images - epsilon
    torch.testing.assert_close(adversarial, expected, atol=1e-12, rtol=0)

    #The linear example must have larger adversarial loss
    clean_loss = F.cross_entropy(model(images), labels).item()
    attacked_loss = F.cross_entropy(model(adversarial), labels).item()
    if attacked_loss <= clean_loss:
        raise AssertionError("Synthetic PGD did not increase the loss")

    #One step of PGD without a random start must equal FGSM
    one_step = pgd_attack(
        model, images, labels, epsilon, epsilon, steps=1,
        random_start=False,
    )
    fgsm = fgsm_attack(model, images, labels, epsilon)
    torch.testing.assert_close(one_step, fgsm, atol=1e-12, rtol=0)

    return {
        "projected_boundary_reached": True,
        "one_step_matches_fgsm": True,
        "clean_loss": round(clean_loss, 6),
        "adversarial_loss": round(attacked_loss, 6),
    }


def check_rejected_inputs(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    #Check invalid perturbation budgets and gradient steps
    for epsilon in (-0.1, 1.1, float("nan"), float("inf")):
        try:
            pgd_attack(model, images, labels, epsilon, 0.01, 2)
        except ValueError:
            pass
        else:
            raise AssertionError("PGD accepted an invalid epsilon")

    for step_size in (0.0, -0.01, float("nan"), float("inf")):
        try:
            pgd_attack(model, images, labels, 2 / 255, step_size, 2)
        except ValueError:
            pass
        else:
            raise AssertionError("PGD accepted an invalid step size")

    for steps in (0, -1, 1.5, True):
        try:
            pgd_attack(model, images, labels, 2 / 255, 0.01, steps)
        except ValueError:
            pass
        else:
            raise AssertionError("PGD accepted an invalid step count")

    bad_images = images[:2].clone()
    bad_images[0, 0, 0, 0] = 1.1
    try:
        pgd_attack(model, bad_images, labels[:2], 2 / 255, 0.01, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("PGD accepted out-of-range pixels")

    #Input gradients cannot be calculated inside inference mode
    with torch.inference_mode():
        try:
            pgd_attack(
                model, images[:2], labels[:2], 2 / 255,
                0.01, 2,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("PGD silently ran in inference mode")


def main() -> None:
    torch.manual_seed(42)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    #Load the trusted baseline checkpoint from the completed training run
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(
            "checkpoints/baseline_best.pt",
            map_location="cpu",
            weights_only=True,
        )
    if checkpoint["epoch"] != 29:
        raise AssertionError("Unexpected baseline checkpoint epoch")

    model = CompactCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    #Use the same first validation batch used for FGSM smoke tests
    _, validation_loader = create_data_loaders(batch_size=128, seed=42)
    images, labels = next(iter(validation_loader))
    images = images.to(device)
    labels = labels.to(device)
    original_images = images.clone()

    #Keep copies of all parameters and BatchNorm buffers
    original_state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }

    with torch.inference_mode():
        clean_outputs = model(images)
        clean_loss = F.cross_entropy(clean_outputs, labels).item()
        clean_correct = (
            clean_outputs.argmax(dim=1) == labels
        ).sum().item()

    epsilon_results = []

    for epsilon in EPSILON_VALUES:
        #Use ten projected ascent steps with a random start
        step_size = max(epsilon / 4, 1 / 255)
        adversarial = pgd_attack(
            model, images, labels, epsilon, step_size,
            steps=STEPS, random_start=True,
        )
        maximum_change = (adversarial - images).abs().max().item()

        #Verify perturbation budgets, valid pixels and detached outputs
        if maximum_change > epsilon + TOLERANCE:
            raise AssertionError("PGD exceeded the L-infinity budget")
        if adversarial.min().item() < 0.0 or adversarial.max().item() > 1.0:
            raise AssertionError("PGD produced out-of-range pixels")
        if not torch.isfinite(adversarial).all().item():
            raise AssertionError("PGD produced non-finite pixel values")
        if adversarial.requires_grad or adversarial.grad_fn is not None:
            raise AssertionError("PGD retained a gradient graph")
        if not torch.equal(original_images, images):
            raise AssertionError("PGD modified the input images")
        if epsilon == 0.0 and not torch.equal(adversarial, images):
            raise AssertionError("Zero-epsilon PGD changed the images")
        if epsilon > 0.0 and maximum_change <= 0.0:
            raise AssertionError("Positive-epsilon PGD did not change pixels")
        if model.training:
            raise AssertionError("PGD changed the model evaluation mode")

        #Record diagnostic accuracy and loss on validation data only
        with torch.inference_mode():
            logits = model(adversarial)
            attacked_loss = F.cross_entropy(logits, labels).item()
            attacked_correct = (
                logits.argmax(dim=1) == labels
            ).sum().item()
        if epsilon == 0.0 and attacked_correct != clean_correct:
            raise AssertionError("Zero-epsilon PGD accuracy differs from clean")

        epsilon_results.append(
            {
                "epsilon": round(epsilon, 9),
                "step_size": round(step_size, 9),
                "steps": STEPS,
                "random_start": True,
                "max_abs_pixel_change": round(maximum_change, 9),
                "loss": round(attacked_loss, 6),
                "correct_predictions": attacked_correct,
                "accuracy_pct": round(100 * attacked_correct / len(labels), 4),
                "constraints_passed": True,
            }
        )

    epsilon = 4 / 255
    step_size = epsilon / 4

    #Verify seeded random initialization produces the same output
    torch.manual_seed(2026)
    first_attack = pgd_attack(
        model, images[:4], labels[:4], epsilon, step_size,
        steps=3, random_start=True,
    )
    torch.manual_seed(2026)
    second_attack = pgd_attack(
        model, images[:4], labels[:4], epsilon, step_size,
        steps=3, random_start=True,
    )
    torch.testing.assert_close(
        first_attack, second_attack, atol=1e-6, rtol=0
    )

    #One PGD step from the clean image must reproduce FGSM
    one_step = pgd_attack(
        model, images[:4], labels[:4], epsilon, epsilon,
        steps=1, random_start=False,
    )
    fgsm = fgsm_attack(model, images[:4], labels[:4], epsilon)
    torch.testing.assert_close(one_step, fgsm, atol=TOLERANCE, rtol=0)

    #Check that the caller's input gradient is not filled by PGD
    gradient_input = images[:2].detach().clone().requires_grad_(True)
    pgd_attack(
        model, gradient_input, labels[:2], epsilon, step_size,
        steps=2, random_start=False,
    )
    if gradient_input.grad is not None:
        raise AssertionError("PGD populated an original input gradient")

    #PGD should run from no_grad while enabling gradients internally
    with torch.no_grad():
        no_grad_attack = pgd_attack(
            model, images[:2], labels[:2], epsilon, step_size,
            steps=2, random_start=False,
        )
    if not torch.isfinite(no_grad_attack).all().item():
        raise AssertionError("PGD failed inside no_grad")

    #PGD must restore training mode without updating BatchNorm buffers
    model.train()
    pgd_attack(
        model, images[:2], labels[:2], epsilon, step_size,
        steps=2, random_start=False,
    )
    if not all(module.training for module in model.modules()):
        raise AssertionError("PGD did not restore training mode")
    model.eval()

    #Preserve preexisting model gradients and avoid creating new ones
    first_parameter = next(model.parameters())
    first_parameter.grad = torch.full_like(first_parameter, 0.25)
    saved_gradient = first_parameter.grad.clone()
    pgd_attack(
        model, images[:2], labels[:2], epsilon, step_size,
        steps=2, random_start=False,
    )
    if not torch.equal(first_parameter.grad, saved_gradient):
        raise AssertionError("PGD modified a parameter gradient")
    first_parameter.grad = None
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("PGD populated parameter gradients")

    #Confirm that no saved model parameters or buffers changed
    for name, tensor in model.state_dict().items():
        if not torch.equal(tensor, original_state[name]):
            raise AssertionError(f"PGD changed the model state: {name}")

    check_rejected_inputs(model, images, labels)
    synthetic = check_synthetic_steps()

    summary = {
        "verification": "PASSED",
        "checkpoint_epoch": checkpoint["epoch"],
        "device": str(device),
        "dataset": "CIFAR-10 validation only",
        "batch_size": len(labels),
        "clean_batch_loss": round(clean_loss, 6),
        "clean_batch_correct": clean_correct,
        "clean_batch_accuracy_pct": round(100 * clean_correct / len(labels), 4),
        "attack": "Untargeted L-infinity PGD",
        "epsilon_results": epsilon_results,
        "synthetic_numerical_behavior": synthetic,
        "one_step_matches_fgsm": True,
        "seeded_random_start_reproducible": True,
        "pixel_bounds_and_epsilon_budgets": "PASSED",
        "model_weights_and_buffers_unchanged": True,
        "model_parameter_gradients_unchanged": True,
        "training_and_evaluation_modes_preserved": True,
        "invalid_input_rejection": "PASSED",
        "test_set_used": False,
    }

    #Save the diagnostic results and correctness checks
    output_path = Path("logs/pgd_validation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
