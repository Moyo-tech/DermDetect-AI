"""DenseNet121 preparation; pretrained acquisition is deliberately external."""
import torch
from torch import nn
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.transforms import functional as TF

SEED = 20260729


def transform(image, *, training=False, generator=None):
    """Official V1 resize256/center224/bilinear/ImageNet normalization.

    Seeded train-only flips/right-angle rotations precede the same preprocessing.
    """
    image = image.convert("RGB")
    if training:
        if generator is None:
            raise ValueError("Explicit seeded generator required for augmentation")
        if torch.rand((), generator=generator).item() < .5:
            image = TF.hflip(image)
        if torch.rand((), generator=generator).item() < .5:
            image = TF.vflip(image)
        angle = 90 * torch.randint(4, (), generator=generator).item()
        image = TF.rotate(image, angle, expand=True)
    return DenseNet121_Weights.IMAGENET1K_V1.transforms()(image)


class DenseNetSystem(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("Invalid dropout")
        self.backbone = densenet121(weights=None)  # Never automatic downloads.
        self.backbone.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1024, 7))
        self.phase = "head"
        self.set_phase("head")

    def set_phase(self, phase):
        if phase not in ("head", "final_block"):
            raise ValueError("Only head warm-up and final-block adaptation allowed")
        self.phase = phase
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(name.startswith("classifier.") or
                (phase == "final_block" and name.startswith(("features.denseblock4.", "features.norm5."))))
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # Frozen BatchNorm running statistics must also stay frozen.
        for name, module in self.backbone.features.named_children():
            module.train(mode and self.phase == "final_block" and name in ("denseblock4", "norm5"))
        return self

    def forward(self, images):
        return self.backbone(images)

    def features(self, images):
        return torch.nn.functional.adaptive_avg_pool2d(
            torch.relu(self.backbone.features(images)), (1, 1)).flatten(1)


def weighted_loss(logits, targets, image_weights):
    """Batch mean of externally computed M2 weights; no batch renormalization.

    Uniform active-view sampling is required for the approved population objective.
    """
    weights = torch.as_tensor(image_weights, dtype=logits.dtype, device=logits.device)
    if weights.shape != targets.shape or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Finite nonnegative per-view weights required")
    return (torch.nn.functional.cross_entropy(logits, targets, reduction="none") * weights).mean()


def save_checkpoint(path, model, configuration):
    torch.save({"state_dict": model.state_dict(), "configuration": configuration,
                "phase": model.phase}, path)


def load_checkpoint(path, expected_configuration):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["configuration"] != expected_configuration:
        raise ValueError("Checkpoint configuration mismatch")
    model = DenseNetSystem(expected_configuration["dropout"])
    model.load_state_dict(saved["state_dict"], strict=True)
    model.set_phase(saved["phase"])
    return model.eval()
