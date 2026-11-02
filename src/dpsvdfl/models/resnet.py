"""Initial model W^(0): the public ImageNet ResNet-18 with a new, seeded classifier (paper Sec. V-A).

The backbone is ``torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)``. The 1000-class ImageNet
classifier is replaced by ``nn.Linear(512, num_classes)`` with PyTorch's default initialisation, drawn under the
classifier seed of the seed plan inside a scoped CPU RNG. W^(0) is therefore public and data-independent, and the same
experiment seed gives bitwise the same W^(0) for both variants on the same software stack.
"""

from torch import nn
from torchvision.models import ResNet, ResNet18_Weights, resnet18

from dpsvdfl.reproducibility import seeded_cpu_rng

ARCHITECTURE = "resnet18"
# Fixed explicitly (never "DEFAULT"): the checkpoint is part of the definition of W^(0).
RESNET18_WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
CHECKPOINT_NAME = "ResNet18_Weights.IMAGENET1K_V1"


def load_pretrained_resnet18(num_classes: int, classifier_seed: int) -> ResNet:
    """ResNet-18 with the IMAGENET1K_V1 weights and a new ``num_classes``-way classifier initialised under the seed."""
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError(f"num_classes must be a positive integer, got {num_classes!r}")
    model = resnet18(weights=RESNET18_WEIGHTS)
    replace_classifier(model, num_classes, classifier_seed)
    return model


def replace_classifier(model: ResNet, num_classes: int, seed: int) -> None:
    """Replace ``model.fc`` with a new ``nn.Linear(512, num_classes)`` (default init) under ``seed``."""
    in_features = model.fc.in_features
    with seeded_cpu_rng(seed):
        model.fc = nn.Linear(in_features, num_classes)
