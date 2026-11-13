import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.torch_version import TorchVersion

from src.attacks import fgsm_attack
from src.data import create_data_loaders
from src.model import CompactCNN


EPSILON_VALUES = (0.0, 2 / 255, 4 / 255, 8 / 255)
TOLERANCE = 1e-6


def check_synthetic_gradient() -> dict[str, float | bool]:
    #Use a simple classifier with a known direction of attack
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

    #All input gradients should point toward smaller pixel values
    adversarial = fgsm_attack(model, images, labels, epsilon)
    expected = images - epsilon
    torch.testing.assert_close(adversarial, expected, atol=1e-12, rtol=0)

    #Compare the analytic gradient against a centered finite difference
    variable_images = images.clone().requires_grad_(True)
    loss = F.cross_entropy(model(variable_images), labels)
    analytical = torch.autograd.grad(loss, variable_images)[0]
    analytical_value = analytical.reshape(-1)[0].item()

    step = 1e-5
    positive = images.clone()
    negative = images.clone()
    positive.reshape(-1)[0] += step
    negative.reshape(-1)[0] -= step

    positive_loss = F.cross_entropy(model(positive), labels).item()
    negative_loss = F.cross_entropy(model(negative), labels).item()
    numerical_value = (positive_loss - negative_loss) / (2 * step)
    gradient_error = abs(analytical_value - numerical_value)

    if gradient_error >= 1e-8:
        raise AssertionError("Synthetic finite-difference gradient disagrees")

    #An untargeted step should increase loss for this linear example
    clean_loss = F.cross_entropy(model(images), labels).item()
    adversarial_loss = F.cross_entropy(model(adversarial), labels).item()
    if adversarial_loss <= clean_loss:
        raise AssertionError("Synthetic FGSM did not increase loss")

    return {
        "expected_gradient_sign_passed": True,
        "analytical_gradient": round(analytical_value, 10),
        "finite_difference_gradient": round(numerical_value, 10),
        "absolute_gradient_error": gradient_error,
        "clean_loss": round(clean_loss, 6),
        "adversarial_loss": round(adversarial_loss, 6),
    }


def check_rejected_inputs(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    #Reject invalid budgets and incorrectly scaled images
    for invalid_epsilon in (-0.01, 1.01, float("nan")):
        try:
            fgsm_attack(model, images, labels, invalid_epsilon)
        except ValueError:
            pass
        else:
            raise AssertionError("FGSM accepted an invalid epsilon")

    invalid_images = images[:2].clone()
    invalid_images[0, 0, 0, 0] = -0.1
    try:
        fgsm_attack(model, invalid_images, labels[:2], 2 / 255)
    except ValueError:
        pass
    else:
        raise AssertionError("FGSM accepted pixels below zero")

    #Inference mode prevents the input-gradient calculation needed by FGSM
    with torch.inference_mode():
        try:
            fgsm_attack(model, images[:2], labels[:2], 2 / 255)
        except RuntimeError:
            pass
        else:
            raise AssertionError("FGSM silently ran in inference mode")


def main() -> None:
    torch.manual_seed(42)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    #Load only our trusted and previously verified baseline checkpoint
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(
            "checkpoints/baseline_best.pt",
            map_location="cpu",
            weights_only=True,
        )

    if checkpoint["epoch"] != 29:
        raise AssertionError("Unexpected best checkpoint epoch")

    model = CompactCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    #Use only the reserved validation subset for these smoke tests
    _, validation_loader = create_data_loaders(batch_size=128, seed=42)
    images, labels = next(iter(validation_loader))
    images = images.to(device)
    labels = labels.to(device)
    original_images = images.clone()

    #Keep a copy of all weights and BatchNorm buffers
    original_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    with torch.inference_mode():
        clean_outputs = model(images)
        clean_loss = F.cross_entropy(clean_outputs, labels).item()
        clean_correct = (clean_outputs.argmax(dim=1) == labels).sum().item()

    epsilon_results = []

    for epsilon in EPSILON_VALUES:
        #Generate adversarial images against this exact baseline model
        adversarial = fgsm_attack(model, images, labels, epsilon)
        difference = adversarial - images
        maximum_change = difference.abs().max().item()

        #Check the norm budget and valid pixel interval
        if maximum_change > epsilon + TOLERANCE:
            raise AssertionError("FGSM exceeded the L-infinity budget")
        if adversarial.min().item() < 0.0 or adversarial.max().item() > 1.0:
            raise AssertionError("FGSM produced invalid image pixels")
        if not torch.isfinite(adversarial).all().item():
            raise AssertionError("FGSM produced non-finite image pixels")
        if adversarial.requires_grad or adversarial.grad_fn is not None:
            raise AssertionError("FGSM returned an attached gradient graph")
        if not torch.equal(original_images, images):
            raise AssertionError("FGSM modified the input batch")
        if epsilon == 0.0 and not torch.equal(adversarial, images):
            raise AssertionError("Zero-epsilon FGSM modified the images")
        if epsilon > 0.0 and maximum_change <= 0.0:
            raise AssertionError("Positive-epsilon FGSM made no pixel changes")
        if model.training:
            raise AssertionError("FGSM changed evaluation mode")

        #Diagnostic accuracy is not a final robustness result
        with torch.inference_mode():
            outputs = model(adversarial)
            attacked_loss = F.cross_entropy(outputs, labels).item()
            attacked_correct = (outputs.argmax(dim=1) == labels).sum().item()

        if epsilon == 0.0 and attacked_correct != clean_correct:
            raise AssertionError("Zero-epsilon accuracy does not match clean")

        epsilon_results.append(
            {
                "epsilon": round(epsilon, 9),
                "max_abs_pixel_change": round(maximum_change, 9),
                "loss": round(attacked_loss, 6),
                "correct_predictions": attacked_correct,
                "accuracy_pct": round(100 * attacked_correct / len(labels), 4),
                "constraints_passed": True,
            }
        )

    #Compare FGSM with an independent autograd calculation
    reference_images = images[:2].detach().clone().requires_grad_(True)
    reference_loss = F.cross_entropy(model(reference_images), labels[:2])
    reference_gradient = torch.autograd.grad(
        reference_loss, reference_images
    )[0]
    epsilon = 2 / 255
    expected = (
        reference_images.detach() + epsilon * reference_gradient.sign()
    ).clamp(0.0, 1.0)
    actual = fgsm_attack(model, images[:2], labels[:2], epsilon)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)

    #The caller's input gradient must remain untouched
    gradient_input = images[:2].detach().clone().requires_grad_(True)
    fgsm_attack(model, gradient_input, labels[:2], epsilon)
    if gradient_input.grad is not None:
        raise AssertionError("FGSM populated the original input gradient")

    #The attack should also work inside a no_grad context
    with torch.no_grad():
        no_grad_result = fgsm_attack(model, images[:2], labels[:2], epsilon)
    torch.testing.assert_close(no_grad_result, actual, atol=1e-6, rtol=0)

    #Verify that a temporary evaluation mode does not change BatchNorm data
    model.train()
    fgsm_attack(model, images[:2], labels[:2], epsilon)
    if not all(module.training for module in model.modules()):
        raise AssertionError("FGSM failed to restore the training modes")
    model.eval()

    #Check that preexisting parameter gradients are not overwritten
    first_parameter = next(model.parameters())
    first_parameter.grad = torch.full_like(first_parameter, 0.25)
    saved_gradient = first_parameter.grad.clone()
    fgsm_attack(model, images[:2], labels[:2], epsilon)
    if not torch.equal(first_parameter.grad, saved_gradient):
        raise AssertionError("FGSM changed a model parameter gradient")
    first_parameter.grad = None
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("FGSM populated model parameter gradients")

    #Confirm every weight and registered buffer is unchanged
    for name, value in model.state_dict().items():
        if not torch.equal(value, original_state[name]):
            raise AssertionError(f"FGSM changed model state: {name}")

    check_rejected_inputs(model, images, labels)
    synthetic = check_synthetic_gradient()

    summary = {
        "verification": "PASSED",
        "checkpoint_epoch": checkpoint["epoch"],
        "device": str(device),
        "dataset": "CIFAR-10 validation only",
        "batch_size": len(labels),
        "clean_batch_loss": round(clean_loss, 6),
        "clean_batch_correct": clean_correct,
        "clean_batch_accuracy_pct": round(100 * clean_correct / len(labels), 4),
        "epsilon_results": epsilon_results,
        "independent_gradient_formula": "PASSED",
        "synthetic_numerical_gradient": synthetic,
        "pixel_bounds_and_epsilon_budgets": "PASSED",
        "model_weights_and_buffers_unchanged": True,
        "model_parameter_gradients_unchanged": True,
        "training_and_evaluation_modes_preserved": True,
        "invalid_input_rejection": "PASSED",
        "test_set_used": False,
    }

    output_path = Path("logs/fgsm_validation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
