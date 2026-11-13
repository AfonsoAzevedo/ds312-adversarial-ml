import math

import torch
from torch import nn
from torch.nn import functional as F


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    #Reject invalid attack settings before calculating gradients
    if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
        raise ValueError("Epsilon must be between 0 and 1")
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Images must have shape [batch, 3, height, width]")
    if not images.is_floating_point():
        raise TypeError("Images must use floating-point pixel values")
    if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
        raise ValueError("Images and labels must have matching batch sizes")
    if labels.dtype != torch.long:
        raise TypeError("Class labels must have dtype torch.long")
    if images.device != labels.device:
        raise ValueError("Images and labels must be on the same device")
    first_parameter = next(model.parameters(), None)
    if first_parameter is not None and first_parameter.device != images.device:
        raise ValueError("Images and model must be on the same device")
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Generate attacks outside torch.inference_mode()")
    if not torch.isfinite(images).all().item():
        raise ValueError("Images contain non-finite pixels")
    if images.min().item() < 0.0 or images.max().item() > 1.0:
        raise ValueError("Images must use the [0, 1] pixel range")

    #Zero epsilon must leave the images unchanged
    if epsilon == 0.0:
        return images.detach().clone()

    #Save every module mode so BatchNorm behavior can be restored
    original_modes = [(module, module.training) for module in model.modules()]
    try:
        #Use running BatchNorm statistics when calculating the attack
        model.eval()
        with torch.enable_grad():
            attack_images = images.detach().clone().requires_grad_(True)
            outputs = model(attack_images)
            loss = F.cross_entropy(outputs, labels)

            #Calculate input gradients without filling parameter gradients
            input_gradients = torch.autograd.grad(loss, attack_images)[0]
            if not torch.isfinite(input_gradients).all().item():
                raise RuntimeError("FGSM produced non-finite input gradients")

            #Apply the untargeted sign step and enforce valid pixels
            adversarial_images = attack_images + epsilon * input_gradients.sign()
            adversarial_images = adversarial_images.clamp(0.0, 1.0)
            return adversarial_images.detach()
    finally:
        #Restore the original mode of each model submodule
        for module, was_training in original_modes:
            module.training = was_training


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    step_size: float,
    steps: int,
    random_start: bool = True,
) -> torch.Tensor:
    #Check the perturbation budget and step settings
    if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
        raise ValueError("Epsilon must be between 0 and 1")
    if not math.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("Step size must be finite and positive")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("PGD must use a positive integer number of steps")
    if not isinstance(random_start, bool):
        raise TypeError("Random start must be a Boolean value")
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Images must have shape [batch, 3, height, width]")
    if not images.is_floating_point():
        raise TypeError("Images must use floating-point pixel values")
    if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
        raise ValueError("Images and labels must have matching batch sizes")
    if labels.dtype != torch.long:
        raise TypeError("Class labels must have dtype torch.long")
    if images.device != labels.device:
        raise ValueError("Images and labels must be on the same device")
    first_parameter = next(model.parameters(), None)
    if first_parameter is not None and first_parameter.device != images.device:
        raise ValueError("Images and model must be on the same device")
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Generate attacks outside torch.inference_mode()")
    if not torch.isfinite(images).all().item():
        raise ValueError("Images contain non-finite pixels")
    if images.min().item() < 0.0 or images.max().item() > 1.0:
        raise ValueError("Images must use the [0, 1] pixel range")

    #Zero epsilon means that no perturbation is allowed
    if epsilon == 0.0:
        return images.detach().clone()

    #Keep the original pixels fixed for every PGD projection
    original_images = images.detach().clone()
    adversarial_images = original_images.clone()
    original_modes = [(module, module.training) for module in model.modules()]

    try:
        #Use running BatchNorm statistics for the entire attack
        model.eval()

        #Initialize inside the allowed L-infinity ball when requested
        if random_start:
            with torch.no_grad():
                noise = torch.empty_like(adversarial_images).uniform_(
                    -epsilon, epsilon
                )
                adversarial_images = (
                    original_images + noise
                ).clamp(0.0, 1.0)

        for _ in range(steps):
            #Start a fresh gradient graph for each PGD step
            with torch.enable_grad():
                attack_images = (
                    adversarial_images.detach().requires_grad_(True)
                )
                outputs = model(attack_images)
                loss = F.cross_entropy(outputs, labels)
                input_gradients = torch.autograd.grad(
                    loss, attack_images
                )[0]

            if not torch.isfinite(input_gradients).all().item():
                raise RuntimeError("PGD produced non-finite input gradients")

            #Move pixels to increase classification loss
            with torch.no_grad():
                adversarial_images = (
                    attack_images + step_size * input_gradients.sign()
                )

                #Project onto the original image's epsilon neighborhood
                lower_bound = original_images - epsilon
                upper_bound = original_images + epsilon
                adversarial_images = torch.maximum(
                    torch.minimum(adversarial_images, upper_bound),
                    lower_bound,
                )

                #Enforce the valid pixel interval after projection
                adversarial_images = adversarial_images.clamp(0.0, 1.0)

        return adversarial_images.detach()
    finally:
        #Restore the original state of every model submodule
        for module, was_training in original_modes:
            module.training = was_training
