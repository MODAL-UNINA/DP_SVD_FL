"""Reduced representation of a client update: Encode and Decode layout (paper Sec. III-C, Algorithm 1).

At round t the server derives public bases from the released global weights W^(t-1) (one SVD and rank rule per
projected layer). Every sampled client then encodes its full update Delta W_i into

    s_i = ( vec(U_l^T Delta W_{i,l} V_l) )_{l in P}  followed by  ( vec(Delta W_{i,l}) )_{l in B \\ P}

in the order of ``partition.trainable``, where vec is the row-major flattening of the matricised block
(``tensor.reshape(-1)``). Its dimension is D = sum_{l in P} k_l^2 + sum_{l in B \\ P} m_l n_l.

This module stops at s_i: clipping, noise and aggregation are in ``dpsvdfl.dp.mechanism`` and the reconstruction in
``dpsvdfl.federated.server_update``.
"""

from dataclasses import dataclass

import torch
from torch import nn

from dpsvdfl.spectral.matricization import matricize, matrix_shape
from dpsvdfl.spectral.projection import (
    ProjectionBasis,
    compute_projection_basis,
    project_update,
    representation_dimension,
)
from dpsvdfl.spectral.rank_rule import GAMMA, RHO_MAX, TAU

PROJECTED = "projected"
UNPROJECTED = "unprojected"


@dataclass(frozen=True)
class ReducedBlockSpec:
    """Position and shapes of one trainable block inside s_i.

    kind:          "projected" or "unprojected".
    start, end:    s_i[start:end] holds this block.
    tensor_shape:  native parameter shape.
    matrix_shape:  (m_l, n_l) after matricisation.
    reduced_shape: (k_l, k_l) for a projected block, (m_l, n_l) for an unprojected one.
    """

    name: str
    kind: str
    start: int
    end: int
    tensor_shape: tuple[int, ...]
    matrix_shape: tuple[int, int]
    reduced_shape: tuple[int, int]

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ReducedLayout:
    """Ordered, contiguous blocks of s_i."""

    blocks: tuple[ReducedBlockSpec, ...]

    @property
    def dimension(self) -> int:
        """D: the end offset of the last block."""
        return self.blocks[-1].end if self.blocks else 0

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(block.name for block in self.blocks)


@dataclass(frozen=True)
class RoundSpectralState:
    """Everything public needed to encode and decode the updates of ONE round.

    bases:  {projected parameter name: ProjectionBasis} computed from the released W^(t-1) (never from client data).
    layout: the layout of s_i in this round (it changes when the ranks change).

    Built once per round and reused, unchanged, for every sampled client and for the server reconstruction.
    """

    bases: dict[str, ProjectionBasis]
    layout: ReducedLayout

    @property
    def ranks(self) -> dict[str, int]:
        return {name: basis.rank for name, basis in self.bases.items()}

    @property
    def dimension(self) -> int:
        return self.layout.dimension


def model_tensor_shapes(model: nn.Module, partition) -> dict[str, tuple[int, ...]]:
    """{trainable parameter name: native shape} of ``model``."""
    shapes = {name: tuple(parameter.shape) for name, parameter in model.named_parameters()}
    missing = [name for name in partition.trainable_names if name not in shapes]
    if missing:
        raise ValueError(f"the model has no parameters {missing}")
    return {name: shapes[name] for name in partition.trainable_names}


def build_reduced_layout(partition, tensor_shapes: dict[str, tuple[int, ...]], ranks: dict[str, int]) -> ReducedLayout:
    """Lay out the blocks in ``partition.trainable`` order: k_l^2 entries per projected block, m_l n_l otherwise."""
    if set(ranks) != set(partition.projected_names):
        raise ValueError("ranks must be given for exactly the projected blocks")

    blocks = []
    offset = 0
    for name in partition.trainable_names:
        tensor_shape = tuple(tensor_shapes[name])
        m, n = matrix_shape(tensor_shape)
        if name in ranks:
            kind = PROJECTED
            reduced_shape = (ranks[name], ranks[name])
        else:
            kind = UNPROJECTED
            reduced_shape = (m, n)
        length = reduced_shape[0] * reduced_shape[1]
        blocks.append(ReducedBlockSpec(name=name, kind=kind, start=offset, end=offset + length,
                                       tensor_shape=tensor_shape, matrix_shape=(m, n), reduced_shape=reduced_shape))
        offset += length

    layout = ReducedLayout(blocks=tuple(blocks))
    unprojected_shapes = {block.name: block.matrix_shape for block in layout.blocks if block.kind == UNPROJECTED}
    expected_dimension = representation_dimension(ranks, unprojected_shapes)
    if layout.dimension != expected_dimension:
        raise AssertionError(f"layout dimension {layout.dimension} != D = {expected_dimension}")
    return layout


@torch.no_grad()
def build_round_spectral_state(
    global_model: nn.Module,
    partition,
    tau: float = TAU,
    rho_max: float = RHO_MAX,
    gamma: float = GAMMA,
) -> RoundSpectralState:
    """Bases of this round from the released W^(t-1) (SVD + rank rule per projected layer) and the layout of s_i.

    The global model is only read. The bases live on the device of the corresponding global parameter.
    """
    parameters = dict(global_model.named_parameters())
    bases = {}
    for name in partition.projected_names:
        weight = parameters[name].detach()
        bases[name] = compute_projection_basis(matricize(weight), tau=tau, rho_max=rho_max, gamma=gamma)
    tensor_shapes = model_tensor_shapes(global_model, partition)
    ranks = {name: basis.rank for name, basis in bases.items()}
    layout = build_reduced_layout(partition, tensor_shapes, ranks)
    return RoundSpectralState(bases=bases, layout=layout)


@torch.no_grad()
def encode_client_update(
    delta: dict[str, torch.Tensor],
    partition,
    spectral_state: RoundSpectralState,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Encode(Delta W_i): the 1-D vector s_i of length D, on ``device`` (Algorithm 1, line 15).

    Projected block:   vec(U_l^T matricize(Delta W_{i,l}) V_l) with the round's public basis; the block is moved/cast to
                       the basis device and dtype only for the product.
    Unprojected block: Delta W_{i,l}.reshape(-1), complete.
    No SVD is computed here and ``delta`` is not modified; s_i has the dtype of the update. On CUDA the projection runs
    on the GPU and s_i is assembled there (concatenation and copies are exact, so values do not depend on the device).
    """
    layout = spectral_state.layout
    _check_update(delta, partition, layout)
    device = torch.device(device)

    pieces = []
    for block in layout.blocks:
        update = delta[block.name]
        if block.kind == PROJECTED:
            basis = spectral_state.bases[block.name]
            update_matrix = matricize(update).to(device=basis.U_k.device, dtype=basis.U_k.dtype)
            coefficients = project_update(update_matrix, basis.U_k, basis.V_k)  # U_k^T X V_k, [k, k]
            piece = coefficients.reshape(-1)
        else:
            piece = update.reshape(-1)  # native row-major order
        pieces.append(piece.to(dtype=update.dtype))

    if len({piece.device for piece in pieces}) > 1:  # mixed placement: bring every piece to `device` first
        pieces = [piece.to(device) for piece in pieces]
    reduced_vector = torch.cat(pieces).to(device)  # cat copies: s_i never aliases delta
    if reduced_vector.numel() != layout.dimension:
        raise AssertionError(f"encoded length {reduced_vector.numel()} != D = {layout.dimension}")
    return reduced_vector


def unpack_reduced_vector(reduced_vector: torch.Tensor, layout: ReducedLayout) -> dict[str, torch.Tensor]:
    """Split a reduced vector into its blocks, in layout order (views, no copy).

    Projected blocks come back as their [k_l, k_l] coefficient matrices and unprojected blocks as their matricised
    [m_l, n_l] values. This is not a reconstruction: U Z V^T is applied in ``dpsvdfl.federated.server_update``.
    """
    if reduced_vector.ndim != 1 or reduced_vector.numel() != layout.dimension:
        raise ValueError(f"expected a vector of length D = {layout.dimension}, got shape {tuple(reduced_vector.shape)}")
    return {block.name: reduced_vector[block.start: block.end].reshape(block.reduced_shape) for block in layout.blocks}


def _check_update(delta: dict[str, torch.Tensor], partition, layout: ReducedLayout) -> None:
    """Validate an update against the partition and the layout (keys and native shapes)."""
    if layout.names != partition.trainable_names:
        raise ValueError("the round layout does not follow partition.trainable")
    trainable = partition.trainable_parameter_names
    if set(delta) != set(trainable):
        missing = sorted(set(trainable) - set(delta))
        unexpected = sorted(set(delta) - set(trainable))
        raise ValueError(f"update keys do not match the trainable blocks (missing {missing}, unexpected {unexpected})")
    for block in layout.blocks:
        shape = tuple(delta[block.name].shape)
        if shape != block.tensor_shape:
            raise ValueError(f"{block.name}: update shape {shape} != parameter shape {block.tensor_shape}")
