"""Representation strategies of the projected blocks: adaptive DP-SVD-FL and the fixed-rank spectral ablation.

A strategy maps the released global model W^(t-1) and the partition to the round state (bases + layout) consumed
unchanged by the encoder, the Gaussian mechanism and the reconstruction:

    adaptive_spectral  DP-SVD-FL (paper Sec. III-B): SVD of W_l^(t-1) and the rank rule in every round.
    fixed_spectral     fixed-rank spectral ablation (paper Sec. VI-C): ranks k_l^(0) selected ONCE by the rank rule on
                       W^(0); the SVD of the current W_l^(t-1) is still recomputed every round and its top-k_l^(0)
                       singular vectors are used. It is not a frozen-basis experiment.

The strategy never touches clipping, noise, normalisation, reconstruction or accounting.
"""

import hashlib
import json
from dataclasses import dataclass

import torch
from torch import nn

from dpsvdfl.spectral.matricization import matricize
from dpsvdfl.spectral.projection import (
    check_fixed_rank,
    compute_projection_basis,
    compute_spectral_basis_at_fixed_rank,
)
from dpsvdfl.spectral.rank_rule import GAMMA, RHO_MAX, TAU
from dpsvdfl.spectral.representation import RoundSpectralState, build_reduced_layout, model_tensor_shapes

ADAPTIVE_SPECTRAL = "adaptive_spectral"
FIXED_SPECTRAL = "fixed_spectral"
STRATEGY_KINDS = (ADAPTIVE_SPECTRAL, FIXED_SPECTRAL)


@dataclass(frozen=True)
class InitialRankEntry:
    name: str
    matrix_shape: tuple[int, int]
    rank: int  # k_l^(0)
    branch: str  # rank-rule branch on W_l^(0)
    gap_ratio: float | None  # max consecutive singular-value ratio on W_l^(0)


@dataclass(frozen=True)
class InitialRankSnapshot:
    """k_l^(0) of every projected layer, from the rank rule applied ONCE to the public W^(0) (data-independent)."""

    layers: tuple[InitialRankEntry, ...]
    tau: float
    rho_max: float
    gamma: float

    @property
    def ranks(self) -> dict[str, int]:
        return {entry.name: entry.rank for entry in self.layers}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.layers)

    def record(self) -> dict:
        return {
            "tau": self.tau,
            "rho_max": self.rho_max,
            "gamma": self.gamma,
            "layers": [{"name": e.name, "matrix_shape": list(e.matrix_shape), "rank": e.rank, "branch": e.branch,
                        "gap_ratio": e.gap_ratio} for e in self.layers],
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.record(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@torch.no_grad()
def compute_initial_rank_snapshot(initial_model: nn.Module, partition, tau: float = TAU, rho_max: float = RHO_MAX,
                                  gamma: float = GAMMA) -> InitialRankSnapshot:
    """SVD(W_l^(0)) -> rank rule -> k_l^(0) for every projected layer (one SVD per layer, once, before training)."""
    parameters = dict(initial_model.named_parameters())
    layers = []
    for name in partition.projected_names:
        weight = matricize(parameters[name].detach())
        basis = compute_projection_basis(weight, tau=tau, rho_max=rho_max, gamma=gamma)
        layers.append(InitialRankEntry(name=name, matrix_shape=tuple(weight.shape), rank=basis.rank,
                                       branch=basis.rank_rule.branch, gap_ratio=basis.rank_rule.max_gap_ratio))
    return InitialRankSnapshot(layers=tuple(layers), tau=tau, rho_max=rho_max, gamma=gamma)


@dataclass(frozen=True)
class RepresentationStrategy:
    """How the projected blocks are represented (build it with the two constructor functions below)."""

    kind: str
    tau: float = TAU
    rho_max: float = RHO_MAX
    gamma: float = GAMMA
    initial_ranks: InitialRankSnapshot | None = None

    def __post_init__(self) -> None:
        if self.kind not in STRATEGY_KINDS:
            raise ValueError(f"strategy kind must be one of {STRATEGY_KINDS}, got {self.kind!r}")
        if self.kind == ADAPTIVE_SPECTRAL and self.initial_ranks is not None:
            raise ValueError("the adaptive strategy selects its ranks every round: no initial rank snapshot")
        if self.kind == FIXED_SPECTRAL:
            if self.initial_ranks is None:
                raise ValueError("fixed_spectral needs the initial rank snapshot k_l^(0)")
            snapshot = self.initial_ranks
            if (snapshot.tau, snapshot.rho_max, snapshot.gamma) != (self.tau, self.rho_max, self.gamma):
                raise ValueError("the strategy's rank-rule parameters differ from those of its snapshot")


def adaptive_spectral_strategy(tau: float = TAU, rho_max: float = RHO_MAX, gamma: float = GAMMA) -> RepresentationStrategy:
    return RepresentationStrategy(ADAPTIVE_SPECTRAL, tau, rho_max, gamma, None)


def fixed_spectral_strategy(snapshot: InitialRankSnapshot) -> RepresentationStrategy:
    return RepresentationStrategy(FIXED_SPECTRAL, snapshot.tau, snapshot.rho_max, snapshot.gamma, snapshot)


@torch.no_grad()
def build_round_state(global_model: nn.Module, partition, strategy: RepresentationStrategy) -> RoundSpectralState:
    """The representation state of the round from the released W^(t-1) (the model is only read)."""
    parameters = dict(global_model.named_parameters())
    fixed = strategy.initial_ranks.ranks if strategy.initial_ranks is not None else {}
    if strategy.kind == FIXED_SPECTRAL and strategy.initial_ranks.names != partition.projected_names:
        raise ValueError("the snapshot layers differ from the projected layers")
    bases = {}
    for name in partition.projected_names:
        weight = matricize(parameters[name].detach())
        if strategy.kind == FIXED_SPECTRAL:
            check_fixed_rank(fixed[name], tuple(weight.shape), name)
            bases[name] = compute_spectral_basis_at_fixed_rank(weight, fixed[name])
        else:
            bases[name] = compute_projection_basis(weight, tau=strategy.tau, rho_max=strategy.rho_max,
                                                   gamma=strategy.gamma)
    tensor_shapes = model_tensor_shapes(global_model, partition)
    layout = build_reduced_layout(partition, tensor_shapes, {name: basis.rank for name, basis in bases.items()})
    return RoundSpectralState(bases=bases, layout=layout)
