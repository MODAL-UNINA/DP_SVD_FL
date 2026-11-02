"""Trainable, projected and frozen parameter blocks (paper Sec. III-A; Supplementary Sec. S6.3).

DP-SVD-FL partition of the torchvision ResNet-18:

    P (projected)        the ten convolution weights of layer3 and layer4, downsampling convolutions included
    B \\ P (unprojected)  fc.weight and fc.bias (trainable, transmitted in full dimension)
    frozen               stem, layer1, layer2 and EVERY BatchNorm parameter

Every BatchNorm module also stays in evaluation mode during local training (``set_batchnorm_eval``): its running
statistics remain those of the public pretrained model and never depend on client data. Setting
``requires_grad=False`` alone would not freeze them.
"""

from dataclasses import dataclass

from torch import nn

from dpsvdfl.spectral.matricization import matrix_shape

# P for ResNet-18, in architectural order (this order is the concatenation order of the reduced vector s_i).
RESNET18_PROJECTED_LAYERS = (
    "layer3.0.conv1",
    "layer3.0.conv2",
    "layer3.0.downsample.0",
    "layer3.1.conv1",
    "layer3.1.conv2",
    "layer4.0.conv1",
    "layer4.0.conv2",
    "layer4.0.downsample.0",
    "layer4.1.conv1",
    "layer4.1.conv2",
)
RESNET18_UNPROJECTED_PARAMETERS = ("fc.weight", "fc.bias")
PROJECTED_STAGES = ("layer3.", "layer4.")
BATCHNORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


@dataclass(frozen=True)
class ModelPartition:
    """Parameter names of the three disjoint groups of a model.

    projected:   P, blocks whose updates are expressed in the round's spectral basis.
    unprojected: B \\ P, trainable blocks transmitted in full dimension.
    frozen:      neither updated nor transmitted.
    """

    projected: tuple[str, ...]
    unprojected: tuple[str, ...]
    frozen: tuple[str, ...]

    @property
    def trainable(self) -> tuple[str, ...]:
        """B = P followed by B \\ P: the concatenation order of the reduced representation."""
        return self.projected + self.unprojected

    # Aliases used by the encoder, the reconstruction and local training (one block = one physical parameter).
    @property
    def projected_names(self) -> tuple[str, ...]:
        return tuple(self.projected)

    @property
    def unprojected_names(self) -> tuple[str, ...]:
        return tuple(self.unprojected)

    @property
    def trainable_names(self) -> tuple[str, ...]:
        return self.trainable

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        return self.trainable


def resnet18_partition(model: nn.Module) -> ModelPartition:
    """The DP-SVD-FL partition: layer3/layer4 convolutions projected, fc trainable, everything else frozen.

    Checks that every listed module is a bias-free Conv2d and that layer3 / layer4 hold no other convolution, so a model
    of another depth is refused instead of being silently partitioned.
    """
    modules = dict(model.named_modules())
    parameter_names = [name for name, _ in model.named_parameters()]
    for layer_name in RESNET18_PROJECTED_LAYERS:
        module = modules.get(layer_name)
        if not isinstance(module, nn.Conv2d):
            raise ValueError(f"{layer_name} is not a Conv2d layer of this model")
        if module.bias is not None:
            raise ValueError(f"{layer_name} has a bias, which the ResNet-18 partition does not expect")
    convolutions = tuple(name for name, module in model.named_modules()
                         if isinstance(module, nn.Conv2d) and name.startswith(PROJECTED_STAGES))
    if convolutions != RESNET18_PROJECTED_LAYERS:
        raise ValueError("the layer3 / layer4 convolutions of this model are not those of ResNet-18")
    for name in RESNET18_UNPROJECTED_PARAMETERS:
        if name not in parameter_names:
            raise ValueError(f"parameter {name} not found in the model")
    projected = tuple(f"{layer_name}.weight" for layer_name in RESNET18_PROJECTED_LAYERS)
    unprojected = RESNET18_UNPROJECTED_PARAMETERS
    frozen = tuple(name for name in parameter_names if name not in projected and name not in unprojected)
    return ModelPartition(projected=projected, unprojected=unprojected, frozen=frozen)


def apply_partition(model: nn.Module, partition: ModelPartition) -> None:
    """Set ``requires_grad=True`` exactly on the trainable blocks B and ``False`` on every other parameter."""
    trainable = set(partition.trainable_parameter_names)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in trainable)


def set_batchnorm_eval(model: nn.Module) -> None:
    """Put every BatchNorm layer in evaluation mode (must be called again after every ``model.train()``)."""
    for module in model.modules():
        if isinstance(module, BATCHNORM_TYPES):
            module.eval()


def matrix_shapes(model: nn.Module, parameter_names: tuple[str, ...]) -> dict[str, tuple[int, int]]:
    """{name: (m_l, n_l)} of the matricised blocks of the given parameters."""
    parameters = dict(model.named_parameters())
    return {name: matrix_shape(parameters[name].shape) for name in parameter_names}
