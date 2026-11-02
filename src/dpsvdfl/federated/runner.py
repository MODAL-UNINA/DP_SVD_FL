"""One communication round of DP-SVD-FL and the T-round loop (paper Algorithm 1).

One round t (round_index = t - 1), in this order:

    1. the round representation state (bases + layout) is built ONCE from the released W^(t-1) by the caller
       (adaptive or fixed-rank spectral bases);
    2. for every sampled slot i in S_t, in increasing slot order, from the SAME W^(t-1):
         local training -> Delta W_i -> Encode with the SAME round state -> s_i;
    3. Gaussian mechanism: clip each s_i once, sum, add N(0, sigma^2 C^2 I_D) once, divide once by qN;
    4. Decode with the SAME round state and apply ONE server update -> W^(t).

The global model is modified only in step 4. Clients are trained and encoded one at a time (a generator consumed by the
mechanism), so at most one client update is held in memory. On CUDA the whole reduced-update path stays on the GPU.

Diagnostics of a round:
    released (post-processing of released weights; publishable): D(t), D(t)/D_full, ranks, rank-rule branches and gap
        ratios, the public learning rate;
    benchmark-only (depend on S_t or on unprivatised updates; NOT DP releases): sampled / non-empty / clipped counts,
        mean / median / max of the raw ||s_i||, finiteness flags and norms of the server update and of the model.
Individual uploads, sampled slot IDs and seeds are never returned.

Privacy modes: with a ``RunSeedPlan`` (benchmark_reproducible) the Poisson schedule and the per-round noise seeds derive
from the experiment seed; with a ``PrivateReleaseSeedPlan`` (private_release) every S_t and every noise vector are drawn
from the OS CSPRNG at their round (``dpsvdfl.privacy.randomness``).
"""

import dataclasses
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from dpsvdfl.dp.mechanism import ClippingObservation, DPMechanismConfig, apply_reduced_dp_mechanism
from dpsvdfl.federated.sampling import generate_sampling_schedule, validate_sampling_schedule
from dpsvdfl.federated.seeds import RunSeedPlan
from dpsvdfl.federated.server_update import SERVER_LEARNING_RATE, apply_server_update, reconstruct_server_update
from dpsvdfl.models.partitions import ModelPartition
from dpsvdfl.privacy.accountant import PrivacyAccountantConfig
from dpsvdfl.privacy.randomness import is_private_release_plan
from dpsvdfl.spectral.projection import full_dimension
from dpsvdfl.spectral.rank_rule import GAMMA, RHO_MAX, TAU
from dpsvdfl.spectral.representation import RoundSpectralState, encode_client_update
from dpsvdfl.training.evaluation import evaluate_classification_accuracy
from dpsvdfl.training.local import LocalTrainingConfig, round_learning_rate, train_client

NUM_ROUNDS = 300


# ---------------------------------------------------------------- configuration


@dataclass(frozen=True)
class SpectralConfig:
    """Public rank-rule parameters (paper: tau = 2, rho_max = 0.05, gamma = 0.95)."""

    tau: float = TAU
    rho_max: float = RHO_MAX
    gamma: float = GAMMA


@dataclass(frozen=True)
class DPSVDFLConfig:
    """Configuration of one run (defaults: the main protocol of the paper, Table II).

    N, q, C and sigma live in ``dp``; the accountant is always derived from them, so the mechanism and the accountant
    cannot diverge. ``delta = None`` means 1 / N.
    """

    num_rounds: int = NUM_ROUNDS
    local: LocalTrainingConfig = field(default_factory=LocalTrainingConfig)
    dp: DPMechanismConfig = field(default_factory=DPMechanismConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    server_learning_rate: float = SERVER_LEARNING_RATE
    delta: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.num_rounds, bool) or not isinstance(self.num_rounds, int) or self.num_rounds < 1:
            raise ValueError(f"num_rounds must be a positive integer, got {self.num_rounds!r}")

    def accountant_config(self) -> PrivacyAccountantConfig:
        return PrivacyAccountantConfig.from_mechanism_config(self.dp, num_rounds=self.num_rounds, delta=self.delta)


# ---------------------------------------------------------------- round diagnostics


@dataclass(frozen=True)
class ReleasedRoundDiagnostics:
    """Post-processing of the released W^(t-1): publishable with the DP output (empty spectral fields for P = empty)."""

    round_number: int  # t = 1, ..., T
    dimension: int  # D(t)
    full_dimension: int  # D_full of the method's trainable set
    dimension_ratio: float  # D(t) / D_full
    ranks: dict[str, int]
    rank_branches: dict[str, str | None]  # rank-rule branch; None for a fixed rank
    gap_ratios: dict[str, float | None]  # max_j s_j / s_(j+1) of the rank rule; None for a fixed rank
    learning_rate: float  # public local learning-rate schedule lr_t
    rank_sources: dict[str, str] = field(default_factory=dict)  # "adaptive" or "initial_fixed"
    basis_sources: dict[str, str] = field(default_factory=dict)  # "spectral"

    @property
    def is_adaptive_spectral(self) -> bool:
        return all(source == "adaptive" for source in self.rank_sources.values())


@dataclass(frozen=True)
class BenchmarkRoundDiagnostics:
    """Benchmark-only diagnostics (depend on S_t and on unprivatised updates): NOT a DP release.

    clipping_fraction = num_clipped / num_sampled_slots (None for an empty round); a vector counts as clipped when
    ||s_i||_2 > C. raw_reduced_norm_*: mean / median / max of the raw pre-DP ||s_i||_2 over the non-empty sampled
    slots. The stability scalars are L2 norms / max-abs over the trainable parameters: of the reconstructed update
    (before eta_s) and of W^(t-1) / W^(t).
    """

    round_number: int
    num_sampled_slots: int
    num_nonempty_sampled_slots: int
    num_clipped: int
    clipping_fraction: float | None
    raw_reduced_norm_mean: float | None = None
    raw_reduced_norm_median: float | None = None
    raw_reduced_norm_max: float | None = None
    server_update_all_finite: bool | None = None
    model_all_finite: bool | None = None
    server_update_l2_norm: float | None = None
    server_update_max_abs: float | None = None
    trainable_model_l2_norm_before: float | None = None
    trainable_model_l2_norm_after: float | None = None
    trainable_model_max_abs_after: float | None = None
    relative_server_update_norm: float | None = None


STABILITY_DIAGNOSTIC_KEYS = ("server_update_l2_norm", "server_update_max_abs", "trainable_model_l2_norm_before",
                             "trainable_model_l2_norm_after", "trainable_model_max_abs_after",
                             "relative_server_update_norm")


@dataclass(frozen=True)
class RoundResult:
    released: ReleasedRoundDiagnostics
    benchmark: BenchmarkRoundDiagnostics
    devices: dict[str, str]  # where each stage ran
    timings: dict[str, float] = field(default_factory=dict, compare=False)  # observational wall-clock seconds


class _ClippingTally:
    """Scalar clipping diagnostics of the raw s_i, fed by the mechanism (no coordinate is stored)."""

    def __init__(self) -> None:
        self.num_sampled = 0
        self.num_nonempty = 0
        self.num_clipped = 0
        self.nonempty_norms: list[float] = []

    def record(self, original_norm: float, was_clipped: bool, nonempty: bool) -> None:
        self.num_sampled += 1
        self.num_nonempty += int(nonempty)
        self.num_clipped += int(was_clipped)
        if nonempty:
            self.nonempty_norms.append(original_norm)

    def norm_statistics(self) -> tuple[float | None, float | None, float | None]:
        if not self.nonempty_norms:
            return None, None, None
        norms = self.nonempty_norms
        return float(np.mean(norms)), float(np.median(norms)), max(norms)


@torch.no_grad()
def tensor_stability_statistics(tensors) -> tuple[float | None, float | None]:
    """(joint L2 norm, max |coordinate|) of ``tensors`` in float64; a non-finite result is returned as None."""
    norms, maxima = [], []
    for tensor in tensors:
        values = tensor.detach()
        if values.numel() == 0:
            continue
        norms.append(torch.linalg.vector_norm(values, dtype=torch.float64))
        maxima.append(torch.linalg.vector_norm(values, ord=float("inf"), dtype=torch.float64))
    if not norms:
        return None, None
    device = norms[0].device
    l2 = float(torch.linalg.vector_norm(torch.stack([norm.to(device) for norm in norms])))
    max_abs = float(torch.stack([value.to(device) for value in maxima]).max())
    return (l2 if math.isfinite(l2) else None), (max_abs if math.isfinite(max_abs) else None)


def relative_norm(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _released_diagnostics(state: RoundSpectralState, round_index: int,
                          local: LocalTrainingConfig) -> ReleasedRoundDiagnostics:
    D_full = full_dimension({block.name: block.matrix_shape for block in state.layout.blocks})
    return ReleasedRoundDiagnostics(
        round_number=round_index + 1,
        dimension=state.dimension,
        full_dimension=D_full,
        dimension_ratio=state.dimension / D_full,
        ranks={name: basis.rank for name, basis in state.bases.items()},
        rank_branches={name: basis.rank_rule.branch if basis.rank_rule is not None else None
                       for name, basis in state.bases.items()},
        gap_ratios={name: basis.rank_rule.max_gap_ratio if basis.rank_rule is not None else None
                    for name, basis in state.bases.items()},
        learning_rate=round_learning_rate(round_index, local.base_learning_rate, local.learning_rate_decay),
        rank_sources={name: basis.rank_source for name, basis in state.bases.items()},
        basis_sources={name: basis.basis_source for name, basis in state.bases.items()},
    )


# ---------------------------------------------------------------- one round


def _client_contributions(
    global_model: nn.Module,
    partition: ModelPartition,
    train_set: Dataset,
    client_slots: list[np.ndarray],
    sampled_slot_ids: list[int],
    round_index: int,
    seed_plan: RunSeedPlan,
    local_config: LocalTrainingConfig,
    round_state: RoundSpectralState,
    device: torch.device,
    num_workers: int,
    multiprocessing_context: str | None,
    timings: dict[str, float],
) -> Iterator[torch.Tensor]:
    """Yield the raw s_i of the sampled slots one at a time (train -> Delta W_i -> Encode), in slot order.

    An empty slot yields the encoding of an exact zero update. Delta W_i and s_i stay on the training device.
    """
    for slot_id in sampled_slot_ids:
        started = time.perf_counter()
        result = train_client(
            global_model, partition, train_set, client_slots[slot_id], round_index,
            seed_plan.local_training_seed(round_index, slot_id), local_config, device=device,
            num_workers=num_workers, multiprocessing_context=multiprocessing_context, keep_delta_on_device=True,
        )
        for key, seconds in result.timings.items():
            timings[key] = timings.get(key, 0.0) + seconds
        encoding_started = time.perf_counter()
        reduced_update = encode_client_update(result.delta, partition, round_state, device=device)  # s_i on device
        timings["encoding_s"] = timings.get("encoding_s", 0.0) + time.perf_counter() - encoding_started
        del result
        timings["client_work_s"] = timings.get("client_work_s", 0.0) + time.perf_counter() - started
        yield reduced_update


def run_round(
    global_model: nn.Module,
    partition: ModelPartition,
    round_state: RoundSpectralState,
    train_set: Dataset,
    client_slots: list[np.ndarray],
    sampled_slot_ids: list[int] | np.ndarray,
    round_index: int,
    seed_plan: RunSeedPlan,
    config: DPSVDFLConfig,
    device: str | torch.device | None = None,
    num_workers: int = 0,
    multiprocessing_context: str | None = None,
) -> RoundResult:
    """Steps 2-4 of one round with a given public representation state; updates ``global_model`` IN PLACE.

    ``round_state`` must have been built from the current W^(t-1) for this partition.
    """
    _check_round_inputs(client_slots, sampled_slot_ids, config)
    round_started = time.perf_counter()  # observational only
    timings = {"local_training_s": 0.0, "client_delta_s": 0.0, "encoding_s": 0.0, "client_work_s": 0.0}
    sampled_slot_ids = [int(slot_id) for slot_id in sampled_slot_ids]
    training_device = next(global_model.parameters()).device if device is None else torch.device(device)
    destination = dict(global_model.named_parameters())[partition.trainable_parameter_names[0]]
    dtype = destination.dtype

    tally = _ClippingTally()

    def observe_clip(observation: ClippingObservation) -> None:  # benchmark-only scalars of the index-th s_i
        slot_id = sampled_slot_ids[observation.index]
        tally.record(observation.original_norm, observation.was_clipped, nonempty=len(client_slots[slot_id]) > 0)

    contributions = _client_contributions(
        global_model, partition, train_set, client_slots, sampled_slot_ids, round_index, seed_plan, config.local,
        round_state, training_device, num_workers, multiprocessing_context, timings,
    )
    mechanism_started = time.perf_counter()
    if is_private_release_plan(seed_plan):  # private_release: noise from the OS CSPRNG, never from a seed
        released = apply_reduced_dp_mechanism(
            contributions, round_state.dimension, config.dp, noise_seed=None, dtype=dtype, device=training_device,
            on_clip=observe_clip, noise_sampler=seed_plan.gaussian_noise,
        )
    else:  # benchmark_reproducible: per-round seed of the method's noise namespace
        released = apply_reduced_dp_mechanism(
            contributions, round_state.dimension, config.dp, noise_seed=seed_plan.noise_seed(round_index),
            dtype=dtype, device=training_device, on_clip=observe_clip,
        )
    mechanism_and_clients_s = time.perf_counter() - mechanism_started
    update_started = time.perf_counter()
    reconstructed = reconstruct_server_update(released.normalized_noisy_update, partition, round_state,
                                              device=destination.device)
    parameters = dict(global_model.named_parameters())
    trainable = partition.trainable_parameter_names
    model_l2_before, _ = tensor_stability_statistics(parameters[name] for name in trainable)  # W^(t-1)
    apply_server_update(global_model, partition, reconstructed, config.server_learning_rate)  # W^(t)
    server_update_all_finite = all(bool(torch.isfinite(value).all()) for value in reconstructed.values())
    model_all_finite = all(bool(torch.isfinite(parameters[name]).all()) for name in trainable)
    update_l2, update_max_abs = tensor_stability_statistics(reconstructed.values())
    model_l2_after, model_max_abs_after = tensor_stability_statistics(parameters[name] for name in trainable)
    server_update_s = time.perf_counter() - update_started
    norm_mean, norm_median, norm_max = tally.norm_statistics()

    bases = list(round_state.bases.values())
    devices = {
        "global_model": str(next(global_model.parameters()).device),
        "local_training": str(training_device),
        "spectral_bases": str(bases[0].U_k.device) if bases else "not_applicable",
        "dp_mechanism": str(released.normalized_noisy_update.device),
        "server_update_destination": str(destination.device),
    }
    benchmark = BenchmarkRoundDiagnostics(
        round_number=round_index + 1,
        num_sampled_slots=tally.num_sampled,
        num_nonempty_sampled_slots=tally.num_nonempty,
        num_clipped=tally.num_clipped,
        clipping_fraction=None if tally.num_sampled == 0 else tally.num_clipped / tally.num_sampled,
        raw_reduced_norm_mean=norm_mean,
        raw_reduced_norm_median=norm_median,
        raw_reduced_norm_max=norm_max,
        server_update_all_finite=server_update_all_finite,
        model_all_finite=model_all_finite,
        server_update_l2_norm=update_l2,
        server_update_max_abs=update_max_abs,
        trainable_model_l2_norm_before=model_l2_before,
        trainable_model_l2_norm_after=model_l2_after,
        trainable_model_max_abs_after=model_max_abs_after,
        relative_server_update_norm=relative_norm(update_l2, model_l2_before),
    )
    client_work_s = timings.pop("client_work_s")
    timings.update(dp_mechanism_s=max(0.0, mechanism_and_clients_s - client_work_s), server_update_s=server_update_s,
                   round_core_s=time.perf_counter() - round_started)
    return RoundResult(released=_released_diagnostics(round_state, round_index, config.local), benchmark=benchmark,
                       devices=devices, timings=timings)


# ---------------------------------------------------------------- T rounds


@dataclass
class RunSummary:
    """Outcome of ``run_federated_rounds`` (the rounds run by this call only)."""

    rounds: list[RoundResult]
    evaluations: list[dict]  # [{"round": t, "test_accuracy": ...}]: post-processing of released models
    final_test_accuracy: float | None


RoundFunction = Callable[..., RoundResult]


def run_federated_rounds(
    round_function: RoundFunction,
    global_model: nn.Module,
    partition: ModelPartition,
    train_set: Dataset,
    client_slots: list[np.ndarray],
    seed_plan: RunSeedPlan,
    config: DPSVDFLConfig,
    test_set: Dataset | None = None,
    eval_every: int | None = None,
    eval_batch_size: int = 256,
    device: str | torch.device | None = None,
    num_workers: int = 0,
    multiprocessing_context: str | None = None,
    on_round_committed: Callable[[int, nn.Module, RoundResult, dict | None], None] | None = None,
    start_round_index: int = 0,
    sampling_schedule: list[np.ndarray] | None = None,
) -> RunSummary:
    """Run rounds ``start_round_index + 1, ..., T`` IN PLACE on ``global_model``.

    ``round_function(global_model, partition, train_set, client_slots, sampled, round_index, seed_plan, config,
    device=..., num_workers=..., multiprocessing_context=...)`` builds the round state and calls ``run_round``.

    Sampling: benchmark_reproducible uses ``sampling_schedule`` if given, otherwise the full T-round Poisson schedule
    generated once from ``seed_plan.sampling_seed``; private_release draws every S_t from the OS CSPRNG at its round.
    Resume: with ``start_round_index = r`` the model must be W^(r); absolute round indices give the same learning
    rates, seeds and sampled sets as an uninterrupted run. Evaluation: after every ``eval_every``-th round and after
    round T. ``on_round_committed(round_index, model, result, evaluation)`` is called last in every round.
    """
    if len(client_slots) != config.dp.num_client_slots:
        raise ValueError(f"got {len(client_slots)} client slots, but the mechanism uses N = {config.dp.num_client_slots}")
    if eval_every is not None and (isinstance(eval_every, bool) or not isinstance(eval_every, int) or eval_every < 1):
        raise ValueError(f"eval_every must be None or a positive integer, got {eval_every!r}")
    if (isinstance(start_round_index, bool) or not isinstance(start_round_index, int)
            or not 0 <= start_round_index <= config.num_rounds):
        raise ValueError(f"start_round_index must be an integer in [0, T = {config.num_rounds}], "
                         f"got {start_round_index!r}")
    config.accountant_config()  # validates (q, sigma, T, delta) before round 1

    private = is_private_release_plan(seed_plan)
    if private:
        if sampling_schedule is not None:
            raise ValueError("private_release never uses a stored or seed-derived sampling schedule")
        schedule = None
    elif sampling_schedule is None:
        schedule = generate_sampling_schedule(config.dp.num_client_slots, config.dp.sampling_rate, config.num_rounds,
                                              seed_plan.sampling_seed)
    else:
        validate_sampling_schedule(list(sampling_schedule), config.dp.num_client_slots, config.num_rounds)
        schedule = [np.asarray(sampled, dtype=np.int64) for sampled in sampling_schedule]

    rounds, evaluations = [], []
    for round_index in range(start_round_index, config.num_rounds):
        round_started = time.perf_counter()
        if private:
            sampled = seed_plan.sample_round(round_index, config.dp.num_client_slots, config.dp.sampling_rate)
        else:
            sampled = schedule[round_index]
        function_started = time.perf_counter()
        result = round_function(
            global_model, partition, train_set, client_slots, sampled, round_index, seed_plan, config,
            device=device, num_workers=num_workers, multiprocessing_context=multiprocessing_context,
        )
        function_s = time.perf_counter() - function_started
        round_number = round_index + 1
        evaluation = None
        evaluation_s = 0.0
        is_final = round_number == config.num_rounds
        if test_set is not None and (is_final or (eval_every is not None and round_number % eval_every == 0)):
            evaluation_started = time.perf_counter()
            accuracy = evaluate_classification_accuracy(global_model, test_set, batch_size=eval_batch_size,
                                                        num_workers=num_workers,
                                                        multiprocessing_context=multiprocessing_context)
            evaluation_s = time.perf_counter() - evaluation_started
            evaluation = {"round": round_number, "test_accuracy": accuracy}
            evaluations.append(evaluation)
        timings = dict(result.timings)
        core_s = timings.pop("round_core_s", function_s)
        timings.update(spectral_state_s=max(0.0, function_s - core_s), evaluation_s=evaluation_s,
                       round_total_s=time.perf_counter() - round_started)
        result = dataclasses.replace(result, timings=timings)
        rounds.append(result)
        if on_round_committed is not None:
            on_round_committed(round_index, global_model, result, evaluation)

    return RunSummary(rounds=rounds, evaluations=evaluations,
                      final_test_accuracy=evaluations[-1]["test_accuracy"] if evaluations else None)


# ---------------------------------------------------------------- records for logging


def released_round_record(result: RoundResult) -> dict:
    """JSON record of the released diagnostics of one round (``released_round_diagnostics.jsonl``).

    Every round records D, D_full, the ranks, rank-rule branches and gap ratios and the learning rate; the fixed-rank
    ablation also records the rank and basis source of every projected layer.
    """
    released = result.released
    record = {"round": released.round_number, "D": released.dimension, "D_full": released.full_dimension,
              "D_ratio": released.dimension_ratio}
    if released.ranks:
        record.update(ranks=released.ranks, rank_branches=released.rank_branches, gap_ratios=released.gap_ratios)
    record["learning_rate"] = released.learning_rate
    if released.ranks and not released.is_adaptive_spectral:
        record["rank_sources"] = released.rank_sources
        record["basis_sources"] = released.basis_sources
    return record


def benchmark_round_record(result: RoundResult) -> dict:
    """JSON record of the benchmark-only diagnostics of one round (NOT a DP release)."""
    benchmark = result.benchmark
    return {
        "round": benchmark.round_number,
        "num_sampled_slots": benchmark.num_sampled_slots,
        "num_nonempty_sampled_slots": benchmark.num_nonempty_sampled_slots,
        "num_clipped": benchmark.num_clipped,
        "clipping_fraction": benchmark.clipping_fraction,
        "raw_reduced_norm_mean": benchmark.raw_reduced_norm_mean,
        "raw_reduced_norm_median": benchmark.raw_reduced_norm_median,
        "raw_reduced_norm_max": benchmark.raw_reduced_norm_max,
        "server_update_all_finite": benchmark.server_update_all_finite,
        "model_all_finite": benchmark.model_all_finite,
        **{key: getattr(benchmark, key) for key in STABILITY_DIAGNOSTIC_KEYS},
        "devices": result.devices,
    }


def format_round_line(result: RoundResult, num_rounds: int) -> str:
    """Compact console line (the sampled / clipped counts are benchmark-only diagnostics)."""
    released, benchmark = result.released, result.benchmark
    return (f"round {released.round_number:03d}/{num_rounds:03d} | sampled={benchmark.num_sampled_slots} "
            f"nonempty={benchmark.num_nonempty_sampled_slots} | D={released.dimension} "
            f"({100 * released.dimension_ratio:.2f}%) | clipped={benchmark.num_clipped}/{benchmark.num_sampled_slots} "
            f"| lr={released.learning_rate:.6f}")


def _check_round_inputs(client_slots, sampled_slot_ids, config: DPSVDFLConfig) -> None:
    if len(client_slots) != config.dp.num_client_slots:
        raise ValueError(f"got {len(client_slots)} client slots, but the mechanism uses N = {config.dp.num_client_slots}")
    ids = [int(slot_id) for slot_id in sampled_slot_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("sampled slot IDs must be distinct")
    if any(slot_id < 0 or slot_id >= len(client_slots) for slot_id in ids):
        raise ValueError(f"sampled slot IDs must lie in [0, {len(client_slots)})")
