"""Server reconstruction (Decode) of the released vector and global-model update (Algorithm 1, lines 24-26).

The mechanism output is ALREADY normalised: (clipped sum + Gaussian noise) / (qN). With the SAME round state that
encoded the clients, the server splits it by the round layout and forms, for every trainable block l,

    projected l:    Delta_l = U_l Z_l V_l^T     (Z_l: the k_l x k_l block)
    unprojected l:  Delta_l = the full block

reshaped to the native parameter shape, and updates W^(t)_l = W^(t-1)_l + eta_s Delta_l (eta_s = 1 in the paper).
There is no further division by qN, no clipping, no noise and no SVD here. Frozen parameters and buffers are untouched.
"""

import math

import torch
from torch import nn

from dpsvdfl.spectral.projection import reconstruct_update
from dpsvdfl.spectral.representation import PROJECTED, RoundSpectralState, unpack_reduced_vector

SERVER_LEARNING_RATE = 1.0  # eta_s (paper, Table II)


@torch.no_grad()
def reconstruct_server_update(
    normalized_noisy_update: torch.Tensor,
    partition,
    spectral_state: RoundSpectralState,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Decode(z_t): map the normalised noisy reduced vector back to native-shaped trainable tensors (pure).

    Returns {parameter name: tensor} for exactly ``partition.trainable``, with the dtype of the vector, on ``device``
    (default: the device of the vector). Projected blocks are reconstructed on the device and in the dtype of their
    basis. Neither the vector nor the state is modified, and the returned tensors never alias the vector.
    """
    if spectral_state.layout.names != partition.trainable_names:
        raise ValueError("the round layout does not follow partition.trainable")
    if tuple(spectral_state.bases) != partition.projected_names:
        raise ValueError("the round bases do not match partition.projected")
    if not isinstance(normalized_noisy_update, torch.Tensor) or not normalized_noisy_update.is_floating_point():
        raise TypeError("the normalised noisy update must be a floating-point torch.Tensor")
    if not torch.isfinite(normalized_noisy_update).all():
        raise ValueError("the normalised noisy update contains non-finite values")
    output_dtype = normalized_noisy_update.dtype
    output_device = normalized_noisy_update.device if device is None else torch.device(device)

    blocks = unpack_reduced_vector(normalized_noisy_update, spectral_state.layout)
    update = {}
    for block in spectral_state.layout.blocks:
        values = blocks[block.name]
        if block.kind == PROJECTED:
            basis = spectral_state.bases[block.name]
            coefficients = values.to(device=basis.U_k.device, dtype=basis.U_k.dtype)
            matrix = reconstruct_update(coefficients, basis.U_k, basis.V_k)  # U_k Z V_k^T, [m, n]
            matrix = matrix.to(device=output_device, dtype=output_dtype)
        else:
            matrix = values.to(device=output_device, dtype=output_dtype, copy=True)
        update[block.name] = matrix.reshape(block.tensor_shape)  # native shape, e.g. fc.bias [K, 1] -> [K]
    return update


@torch.no_grad()
def apply_server_update(
    global_model: nn.Module,
    partition,
    reconstructed_update: dict[str, torch.Tensor],
    server_learning_rate: float = SERVER_LEARNING_RATE,
) -> None:
    """W_l <- W_l + eta_s Delta_l IN PLACE for every trainable block; everything is validated first."""
    if isinstance(server_learning_rate, bool) or not isinstance(server_learning_rate, (int, float)):
        raise TypeError(f"server_learning_rate must be a real number, got {server_learning_rate!r}")
    if not math.isfinite(server_learning_rate) or server_learning_rate < 0:
        raise ValueError(f"server_learning_rate must be finite and non-negative, got {server_learning_rate}")
    trainable = partition.trainable_parameter_names
    if set(reconstructed_update) != set(trainable):
        raise ValueError("update keys do not match the trainable blocks")
    parameters = dict(global_model.named_parameters())
    for name in trainable:
        value = reconstructed_update[name]
        if tuple(value.shape) != tuple(parameters[name].shape):
            raise ValueError(f"{name}: update shape {tuple(value.shape)} != parameter shape "
                             f"{tuple(parameters[name].shape)}")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{name}: the update must be finite floating-point values")

    if server_learning_rate == 0:
        return  # W + 0 * Delta = W; skipping keeps the model bitwise identical
    for name in trainable:
        parameter = parameters[name]
        step = reconstructed_update[name].to(device=parameter.device, dtype=parameter.dtype)
        parameter.add_(step, alpha=server_learning_rate)
