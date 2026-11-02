"""The two variants of DP-SVD-FL evaluated in the paper and how one round of each is built (paper Sec. III, VI-C).

Both variants share the same pipeline: local SGD from W^(t-1), Encode in the round's public spectral basis, one joint
clip per client at C, Gaussian noise N(0, sigma^2 C^2 I_D) added once to the sum, division by the public constant qN,
Decode and one server update with eta_s = 1, and the same accountant. They differ only in how the ranks are chosen and
in their independent DP-noise namespace:

    key              label                 ranks k_l^(t)
    adaptive         DP-SVD-FL             rank rule on the released W^(t-1) at every round
    fixed_spectral   Fixed-rank spectral   k_l^(0) chosen on W^(0) and kept fixed (ablation, Sec. VI-C)

The fixed-rank spectral ablation is a separate CIFAR-10 Dirichlet experiment.
"""

from dataclasses import dataclass, field

from torch import nn

from dpsvdfl.federated.runner import RoundResult, run_round
from dpsvdfl.federated.seeds import DPSVDFL_NOISE_NAMESPACE, FIXED_SPECTRAL_NOISE_NAMESPACE
from dpsvdfl.models.partitions import ModelPartition
from dpsvdfl.spectral.strategies import (
    InitialRankSnapshot,
    adaptive_spectral_strategy,
    build_round_state,
    fixed_spectral_strategy,
)

REPRESENTATION_ADAPTIVE = "adaptive_spectral"
REPRESENTATION_FIXED = "fixed_spectral"


def _canonical(partition: ModelPartition) -> ModelPartition:
    return partition


@dataclass(frozen=True)
class MethodSpec:
    """Static description of one variant.

    ``parameters`` lists the variant's hyperparameters (none for the two variants of the paper); they are recorded in
    every run summary.
    """

    key: str
    label: str
    noise_namespace: str
    representation: str
    training_partition: object  # Callable[[ModelPartition], ModelPartition]
    parameters: dict = field(default_factory=dict)


METHODS: dict[str, MethodSpec] = {
    spec.key: spec for spec in (
        MethodSpec("adaptive", "DP-SVD-FL", DPSVDFL_NOISE_NAMESPACE, REPRESENTATION_ADAPTIVE, _canonical),
        MethodSpec("fixed_spectral", "Fixed-rank spectral", FIXED_SPECTRAL_NOISE_NAMESPACE, REPRESENTATION_FIXED,
                   _canonical),
    )
}


def get_method(key: str) -> MethodSpec:
    if key not in METHODS:
        raise ValueError(f"unknown method {key!r}; expected one of {sorted(METHODS)}")
    return METHODS[key]


def method_parameters(key: str, overrides: dict | None = None) -> dict:
    """The variant's hyperparameters: the paper's values, updated by ``overrides`` (unknown keys are refused)."""
    spec = get_method(key)
    parameters = dict(spec.parameters)
    for name, value in (overrides or {}).items():
        if name not in parameters:
            raise ValueError(f"{key} has no parameter {name!r} (parameters: {sorted(parameters) or 'none'})")
        parameters[name] = value
    return parameters


def make_round_function(key: str, parameters: dict, spectral_config, initial_ranks: InitialRankSnapshot | None):
    """The round function of a variant: build the round state from W^(t-1), then run the shared round."""
    spec = get_method(key)
    if spec.representation == REPRESENTATION_ADAPTIVE:
        strategy = adaptive_spectral_strategy(spectral_config.tau, spectral_config.rho_max, spectral_config.gamma)
    else:
        if initial_ranks is None:
            raise ValueError("the fixed-rank spectral ablation needs the initial rank snapshot k_l^(0)")
        strategy = fixed_spectral_strategy(initial_ranks)

    def round_function(global_model: nn.Module, partition: ModelPartition, train_set, client_slots, sampled,
                       round_index: int, seed_plan, config, device=None, num_workers: int = 0,
                       multiprocessing_context=None) -> RoundResult:
        state = build_round_state(global_model, partition, strategy)
        return run_round(global_model, partition, state, train_set, client_slots, sampled, round_index, seed_plan,
                         config, device=device, num_workers=num_workers,
                         multiprocessing_context=multiprocessing_context)

    return round_function
