"""Experiment driver: configuration files, one run of one variant of the method, checkpoint and resume.

A *run* is one (dataset, partition, experiment seed, method[, noise multiplier]) cell of a configuration grid, where
``method`` is ``adaptive`` (DP-SVD-FL) or ``fixed_spectral`` (fixed-rank ablation). Its inputs are pure functions of the
experiment seed (``dpsvdfl.federated.seeds``): the client partition, the Poisson schedule, W^(0) and the local-training
streams are identical for both variants at the same seed, while each variant draws its own DP noise.

Layout of a run directory::

    run_config.json                      public configuration + fingerprints of W^(0), partition and schedule
    released_round_diagnostics.jsonl     released per-round diagnostics (D, ranks, branches, learning rate, accuracy)
    benchmark_diagnostics_internal.jsonl benchmark-only diagnostics (clipping counts, norms): NOT a DP release
    checkpoints/latest.json, round_XXXX.pt   model checkpoints (resume)
    round_attempts_internal.jsonl        private_release only: write-ahead ledger of the round attempts
    final_model.pt, public_summary.json, status.json   written when the run completes

Resume: running the same command again continues after the last checkpoint with absolute round indices, so the
learning rates, sampled sets and seeds are those of an uninterrupted run. On the same device type and software stack
a resumed benchmark_reproducible run is expected to be bitwise identical to an uninterrupted one.
"""

import json
import os
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from dpsvdfl.data.datasets import DATASET_NAMES, dataset_labels, load_benchmark, num_classes
from dpsvdfl.data.partitioning import PARTITION_SCHEMES, make_partition, validate_partition
from dpsvdfl.dp.mechanism import DPMechanismConfig
from dpsvdfl.federated.io import (
    append_jsonl,
    arrays_fingerprint,
    cpu_state_dict,
    read_json,
    read_jsonl,
    state_dict_fingerprint,
    truncate_jsonl,
    write_bytes_atomic,
    write_json_atomic,
)
from dpsvdfl.federated.methods import get_method, make_round_function, method_parameters
from dpsvdfl.federated.runner import (
    DPSVDFLConfig,
    SpectralConfig,
    benchmark_round_record,
    format_round_line,
    released_round_record,
    run_federated_rounds,
)
from dpsvdfl.federated.sampling import generate_sampling_schedule
from dpsvdfl.federated.seeds import RunSeedPlan, contextual_noise_namespace, privacy_sweep_noise_context
from dpsvdfl.models.partitions import apply_partition, resnet18_partition
from dpsvdfl.models.resnet import ARCHITECTURE, CHECKPOINT_NAME, load_pretrained_resnet18
from dpsvdfl.privacy.modes import PrivacyConfiguration, public_privacy_record
from dpsvdfl.privacy.randomness import OSRandomness, PrivateReleaseSeedPlan, check_private_backend
from dpsvdfl.spectral.strategies import compute_initial_rank_snapshot
from dpsvdfl.training.local import AUGMENTATION_BACKEND, LocalTrainingConfig

RUN_CONFIG = "run_config.json"
RELEASED_LOG = "released_round_diagnostics.jsonl"
BENCHMARK_LOG = "benchmark_diagnostics_internal.jsonl"
ATTEMPTS_LOG = "round_attempts_internal.jsonl"
CHECKPOINTS_DIR = "checkpoints"
FINAL_MODEL = "final_model.pt"
PUBLIC_SUMMARY = "public_summary.json"
STATUS_FILE = "status.json"

# The shared protocol of the paper (Table II). A configuration file lists these keys explicitly.
PROTOCOL_KEYS = (
    "num_client_slots", "sampling_rate", "num_rounds", "local_epochs", "batch_size", "learning_rate",
    "learning_rate_decay", "momentum", "weight_decay", "batching_policy", "clipping_norm", "noise_multiplier",
    "delta", "server_learning_rate", "tau", "rho_max", "gamma", "dirichlet_alpha", "eval_every", "privacy_mode",
)
NOISE_CONTEXTS = ("none", "privacy_sweep")


# ---------------------------------------------------------------- run specification


@dataclass(frozen=True)
class RunSpec:
    """Everything that defines one run (its public identity)."""

    dataset: str
    partition: str  # "iid" or "dirichlet"
    dirichlet_alpha: float | None
    seed: int
    method: str
    method_parameters: dict
    config: DPSVDFLConfig
    eval_every: int | None
    noise_context: str | None = None
    privacy_mode: str = "benchmark_reproducible"
    experiment_name: str = "experiment"

    def __post_init__(self) -> None:
        if self.dataset not in DATASET_NAMES:
            raise ValueError(f"unknown dataset {self.dataset!r}; expected one of {DATASET_NAMES}")
        if self.partition not in PARTITION_SCHEMES:
            raise ValueError(f"unknown partition {self.partition!r}; expected one of {PARTITION_SCHEMES}")
        if (self.partition == "dirichlet") != (self.dirichlet_alpha is not None):
            raise ValueError("dirichlet_alpha is required for the Dirichlet partition and only for it")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"the experiment seed must be a non-negative integer, got {self.seed!r}")
        get_method(self.method)
        PrivacyConfiguration(self.privacy_mode)

    @property
    def noise_namespace(self) -> str:
        return contextual_noise_namespace(get_method(self.method).noise_namespace, self.noise_context)

    @property
    def label(self) -> str:
        return get_method(self.method).label

    def configuration_record(self) -> dict:
        """The public configuration (same keys as the run summaries of the paper's experiments)."""
        config = self.config
        return {
            "dataset": self.dataset,
            "partition": self.partition,
            "dirichlet_alpha": self.dirichlet_alpha,
            "architecture": ARCHITECTURE,
            "checkpoint": CHECKPOINT_NAME,
            "N": config.dp.num_client_slots,
            "q": config.dp.sampling_rate,
            "C": config.dp.clipping_norm,
            "sigma": config.dp.noise_multiplier,
            "T": config.num_rounds,
            "delta": config.accountant_config().target_delta,
            "local_epochs": config.local.local_epochs,
            "batch_size": config.local.batch_size,
            "learning_rate": config.local.base_learning_rate,
            "learning_rate_decay": config.local.learning_rate_decay,
            "momentum": config.local.momentum,
            "weight_decay": config.local.weight_decay,
            "batching_policy": config.local.batching_policy,
            "augmentation_backend": AUGMENTATION_BACKEND,
            "server_learning_rate": config.server_learning_rate,
            "tau": config.spectral.tau,
            "rho_max": config.spectral.rho_max,
            "gamma": config.spectral.gamma,
            "eval_every": self.eval_every,
        }

    def record(self) -> dict:
        return json.loads(json.dumps({
            "experiment_name": self.experiment_name,
            "method": self.method,
            "label": self.label,
            "method_parameters": self.method_parameters,
            "experiment_seed": self.seed,
            "noise_namespace": self.noise_namespace,
            "noise_context": self.noise_context,
            "privacy": PrivacyConfiguration(self.privacy_mode).record(),
            "configuration": self.configuration_record(),
        }))


def protocol_to_config(protocol: dict, noise_multiplier: float | None = None) -> DPSVDFLConfig:
    """DPSVDFLConfig of a protocol dictionary (all ``PROTOCOL_KEYS`` must be present)."""
    missing = [key for key in PROTOCOL_KEYS if key not in protocol]
    unknown = [key for key in protocol if key not in PROTOCOL_KEYS]
    if missing or unknown:
        raise ValueError(f"protocol: missing keys {missing}, unknown keys {unknown}")
    sigma = protocol["noise_multiplier"] if noise_multiplier is None else noise_multiplier
    return DPSVDFLConfig(
        num_rounds=int(protocol["num_rounds"]),
        local=LocalTrainingConfig(
            local_epochs=int(protocol["local_epochs"]), batch_size=int(protocol["batch_size"]),
            base_learning_rate=float(protocol["learning_rate"]),
            learning_rate_decay=float(protocol["learning_rate_decay"]), momentum=float(protocol["momentum"]),
            weight_decay=float(protocol["weight_decay"]), batching_policy=protocol["batching_policy"],
        ),
        dp=DPMechanismConfig(
            clipping_norm=float(protocol["clipping_norm"]), noise_multiplier=float(sigma),
            sampling_rate=float(protocol["sampling_rate"]), num_client_slots=int(protocol["num_client_slots"]),
        ),
        spectral=SpectralConfig(tau=float(protocol["tau"]), rho_max=float(protocol["rho_max"]),
                                gamma=float(protocol["gamma"])),
        server_learning_rate=float(protocol["server_learning_rate"]),
        delta=None if protocol["delta"] is None else float(protocol["delta"]),
    )


# ---------------------------------------------------------------- configuration files


def load_experiment_config(path: str | Path) -> dict:
    """Read and check a JSON experiment configuration (see ``configs/``)."""
    config = read_json(path)
    for key in ("name", "protocol", "grid"):
        if key not in config:
            raise ValueError(f"{path}: missing top-level key {key!r}")
    grid = config["grid"]
    for key in ("datasets", "partitions", "seeds", "methods"):
        if key not in grid:
            raise ValueError(f"{path}: grid without {key!r}")
    if grid.get("noise_context", "none") not in NOISE_CONTEXTS:
        raise ValueError(f"{path}: grid.noise_context must be one of {NOISE_CONTEXTS}")
    protocol_to_config(config["protocol"])  # validates the protocol
    for method in grid["methods"]:
        method_parameters(method, config.get("method_parameters", {}).get(method))
    return config


def apply_overrides(protocol: dict, overrides: list[str]) -> dict:
    """``key=value`` overrides of protocol entries (JSON values), e.g. ``num_rounds=2`` for a smoke test."""
    protocol = dict(protocol)
    for item in overrides:
        key, separator, value = item.partition("=")
        if not separator or key not in PROTOCOL_KEYS:
            raise ValueError(f"invalid override {item!r}; expected key=value with key in {PROTOCOL_KEYS}")
        protocol[key] = json.loads(value)
    return protocol


def expand_runs(config: dict, datasets=None, partitions=None, seeds=None, methods=None, noise_multipliers=None,
                overrides: list[str] | None = None) -> list[RunSpec]:
    """Every run of a configuration grid, optionally restricted to the given values (``None`` = all)."""
    grid = config["grid"]
    protocol = apply_overrides(config["protocol"], overrides or [])

    def select(values, chosen, what):
        if chosen is None:
            return list(values)
        unknown = [value for value in chosen if value not in values]
        if unknown:
            raise ValueError(f"{what} {unknown} not in the configuration grid {list(values)}")
        return list(chosen)

    sigmas = grid.get("noise_multipliers") or [None]
    runs = []
    for dataset in select(grid["datasets"], datasets, "datasets"):
        for partition in select(grid["partitions"], partitions, "partitions"):
            for sigma in select(sigmas, noise_multipliers, "noise multipliers"):
                run_config = protocol_to_config(protocol, sigma)
                noise_context = None
                if grid.get("noise_context", "none") == "privacy_sweep":
                    noise_context = privacy_sweep_noise_context(run_config.dp.noise_multiplier)
                for seed in select(grid["seeds"], seeds, "seeds"):
                    for method in select(grid["methods"], methods, "methods"):
                        runs.append(RunSpec(
                            dataset=dataset, partition=partition,
                            dirichlet_alpha=float(protocol["dirichlet_alpha"]) if partition == "dirichlet" else None,
                            seed=int(seed), method=method,
                            method_parameters=method_parameters(method, config.get("method_parameters", {}).get(method)),
                            config=run_config, eval_every=protocol["eval_every"], noise_context=noise_context,
                            privacy_mode=protocol["privacy_mode"], experiment_name=config["name"],
                        ))
    return runs


def run_directory(output_root: str | Path, spec: RunSpec, sweep: bool = False) -> Path:
    """``<output_root>/<experiment>/<dataset>_<partition>[_sigma<1000 sigma>]/seed<s>/<method>``."""
    setting = f"{spec.dataset}_{spec.partition}"
    if spec.partition == "dirichlet":
        setting += f"{spec.dirichlet_alpha:g}"
    if sweep:
        setting += f"_sigma{round(spec.config.dp.noise_multiplier * 1000):04d}"
    return Path(output_root) / spec.experiment_name / setting / f"seed{spec.seed}" / spec.method


# ---------------------------------------------------------------- federation inputs


@dataclass
class Federation:
    """The common inputs of a seed: client partition, Poisson schedule (benchmark mode) and W^(0) builder."""

    client_slots: list[np.ndarray]
    schedule: list[np.ndarray] | None
    num_classes: int
    classifier_seed: int
    fingerprints: dict = field(default_factory=dict)

    def initial_model(self) -> torch.nn.Module:
        return load_pretrained_resnet18(self.num_classes, self.classifier_seed)


def build_federation(spec: RunSpec, train_set) -> Federation:
    """Partition the training set and draw the sampling schedule from the common seed streams."""
    plan = RunSeedPlan.create(spec.seed)
    labels = dataset_labels(train_set)
    N = spec.config.dp.num_client_slots
    client_slots = make_partition(labels, spec.partition, N, plan.partition_seed, alpha=spec.dirichlet_alpha)
    validate_partition(client_slots, num_examples=len(labels), num_clients=N)
    client_slots = [np.asarray(slot, dtype=np.int64) for slot in client_slots]
    schedule = None
    fingerprints = {"partition": arrays_fingerprint(client_slots)}
    if not PrivacyConfiguration(spec.privacy_mode).is_private:  # private_release never draws a schedule from a seed
        schedule = generate_sampling_schedule(N, spec.config.dp.sampling_rate, spec.config.num_rounds,
                                              plan.sampling_seed)
        fingerprints["sampling_schedule"] = arrays_fingerprint(schedule)
    return Federation(client_slots, schedule, num_classes(spec.dataset), plan.classifier_seed, fingerprints)


def environment_record(device: torch.device) -> dict:
    cuda = device.type == "cuda"
    return {
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "device_type": device.type, "device_name": torch.cuda.get_device_name(device) if cuda else None,
        "cuda": torch.version.cuda if cuda else None,
        "cudnn": torch.backends.cudnn.version() if cuda else None, "torch_threads": torch.get_num_threads(),
    }


# ---------------------------------------------------------------- checkpoints


def _latest_checkpoint(run_dir: Path) -> dict | None:
    pointer = run_dir / CHECKPOINTS_DIR / "latest.json"
    return read_json(pointer) if pointer.exists() else None


def _save_checkpoint(run_dir: Path, model: torch.nn.Module, completed_round: int, device_type: str) -> None:
    """Write round_XXXX.pt, then atomically move the latest pointer to it, then remove older checkpoints."""
    directory = run_dir / CHECKPOINTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    file_name = f"round_{completed_round:04d}.pt"
    state = cpu_state_dict(model)
    write_bytes_atomic(directory / file_name, lambda file: torch.save(state, file))
    write_json_atomic(directory / "latest.json", {"completed_round": completed_round, "file": file_name,
                                                  "device_type": device_type})
    for old in directory.glob("round_*.pt"):
        if old.name != file_name:
            old.unlink()


class RoundAttemptLedger:
    """private_release: write-ahead ledger of the round attempts (one fsynced record BEFORE any draw of a round).

    A round attempt whose noisy aggregate may have left the process before a crash is always on record, and the final
    accounting composes EVERY recorded attempt (epsilon for T' >= T mechanism invocations). A redone round draws fresh
    randomness: nothing is ever re-released.
    """

    def __init__(self, path: Path, num_rounds: int, completed_round: int, fresh: bool) -> None:
        self.path = path
        self.num_rounds = num_rounds
        self.counts: dict[int, int] = {}
        for record in read_jsonl(path):
            self.counts[record["round"]] = record["attempt"]
        if fresh and self.counts:
            raise RuntimeError(f"{path}: round attempts are recorded but the run has no checkpoint")
        missing = [t for t in range(1, completed_round + 1) if t not in self.counts]
        if missing:
            raise RuntimeError(f"{path}: committed rounds {missing} have no recorded attempt; the privacy cost of "
                               "this run cannot be accounted, so it is not resumed")

    def record(self, round_index: int) -> None:
        round_number = round_index + 1
        attempt = self.counts.get(round_number, 0) + 1
        append_jsonl(self.path, {"round": round_number, "attempt": attempt})
        self.counts[round_number] = attempt

    def total_attempts(self) -> int:
        missing = [t for t in range(1, self.num_rounds + 1) if t not in self.counts]
        if missing:
            raise RuntimeError(f"{self.path}: rounds {missing} have no recorded attempt")
        return sum(self.counts.values())


# ---------------------------------------------------------------- one run


@dataclass
class RunResources:
    """Runtime options that are not part of the scientific identity of a run."""

    data_root: Path
    device: torch.device
    num_workers: int = 0
    eval_batch_size: int = 256
    checkpoint_every: int = 1
    allow_device_change: bool = False
    log: Callable[[str], None] = print


def resolve_device(spec: str) -> torch.device:
    """'auto' -> cuda if available else cpu; 'cpu'; 'cuda' / 'cuda:N' (must be available)."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"device {spec!r} requested but CUDA is not available")
    if device.type not in ("cpu", "cuda"):
        raise ValueError(f"unsupported device {spec!r}; use auto, cpu, cuda or cuda:N")
    return device


def run_experiment(spec: RunSpec, run_dir: str | Path, resources: RunResources, train_set=None, test_set=None) -> dict:
    """Run, resume or recognise as complete one run; returns its public summary.

    ``train_set`` / ``test_set`` may be passed to avoid reloading the dataset (tests); otherwise they are loaded from
    ``resources.data_root`` (never downloaded here: see ``scripts/prepare_data.py``).
    """
    run_dir = Path(run_dir)
    log, device = resources.log, resources.device
    privacy = PrivacyConfiguration(spec.privacy_mode)
    if (run_dir / STATUS_FILE).exists() and read_json(run_dir / STATUS_FILE).get("status") == "complete":
        log(f"{run_dir}: complete; not rerun")
        return read_json(run_dir / PUBLIC_SUMMARY)
    if privacy.is_private:
        if resources.checkpoint_every != 1:
            raise ValueError("private_release commits every round: use --checkpoint-every 1")
        check_private_backend()
        os.umask(0o077)  # every file of a private run is owner-only
    run_dir.mkdir(parents=True, exist_ok=True)

    if train_set is None or test_set is None:
        train_set, test_set = load_benchmark(spec.dataset, resources.data_root, download=False)
    federation = build_federation(spec, train_set)
    model = federation.initial_model()  # W^(0) on CPU
    canonical = resnet18_partition(model)
    method = get_method(spec.method)
    training_partition = method.training_partition(canonical)
    snapshot = compute_initial_rank_snapshot(model, canonical, spec.config.spectral.tau, spec.config.spectral.rho_max,
                                             spec.config.spectral.gamma)
    identity = {
        **spec.record(),
        "fingerprints": {"initial_model": state_dict_fingerprint(cpu_state_dict(model)), **federation.fingerprints,
                         "initial_rank_snapshot": snapshot.fingerprint()},
        "training_partition": {"projected": list(training_partition.projected),
                               "unprojected": list(training_partition.unprojected)},
        "initial_rank_snapshot": snapshot.record(),
    }

    latest = _latest_checkpoint(run_dir)
    if latest is None:
        if (run_dir / RELEASED_LOG).exists() and read_jsonl(run_dir / RELEASED_LOG):
            raise RuntimeError(f"{run_dir}: round records exist but no checkpoint; refusing to restart from scratch")
        write_json_atomic(run_dir / RUN_CONFIG, {**identity, "created_at": datetime.now(timezone.utc).isoformat(),
                                                 "environment": environment_record(device)})
        _save_checkpoint(run_dir, model, 0, device.type)  # W^(0) as round 0
        completed = 0
    else:
        saved = read_json(run_dir / RUN_CONFIG)
        differing = sorted(key for key in identity if saved.get(key) != identity[key])
        if differing:
            raise RuntimeError(f"{run_dir}: the existing run differs from this specification in {differing}")
        if latest["device_type"] != device.type and not resources.allow_device_change:
            raise RuntimeError(f"{run_dir}: saved on {latest['device_type']}, now on {device.type}; pass "
                               "--allow-device-change to accept a run that is no longer bitwise reproducible")
        model.load_state_dict(torch.load(run_dir / CHECKPOINTS_DIR / latest["file"], map_location="cpu",
                                         weights_only=True))
        completed = int(latest["completed_round"])
    T = spec.config.num_rounds
    truncate_jsonl(run_dir / RELEASED_LOG, completed)
    truncate_jsonl(run_dir / BENCHMARK_LOG, completed)

    seed_plan = RunSeedPlan.create(spec.seed, spec.noise_namespace)
    ledger = None
    if privacy.is_private:
        ledger = RoundAttemptLedger(run_dir / ATTEMPTS_LOG, T, completed, fresh=latest is None)
        seed_plan = PrivateReleaseSeedPlan(seed_plan, OSRandomness(), attempt_recorder=ledger.record)

    apply_partition(model, training_partition)
    model = model.to(device)
    round_function = make_round_function(spec.method, spec.method_parameters, spec.config.spectral,
                                         snapshot if spec.method == "fixed_spectral" else None)
    started = time.perf_counter()

    def on_round_committed(round_index, global_model, result, evaluation) -> None:
        round_number = round_index + 1
        record = released_round_record(result)
        if evaluation is not None:
            record["test_accuracy"] = evaluation["test_accuracy"]
        append_jsonl(run_dir / RELEASED_LOG, record)
        append_jsonl(run_dir / BENCHMARK_LOG, benchmark_round_record(result))
        if round_number % resources.checkpoint_every == 0 or round_number == T:
            _save_checkpoint(run_dir, global_model, round_number, device.type)
        line = f"{spec.label} | " + format_round_line(result, T)
        if evaluation is not None:
            line += f" | test_acc={evaluation['test_accuracy']:.4f}"
        log(line)

    if completed < T:
        log(f"{spec.label}: {'starting' if completed == 0 else f'resuming after round {completed}'} on {device} "
            f"({run_dir})")
        run_federated_rounds(
            round_function, model, training_partition, train_set, federation.client_slots, seed_plan, spec.config,
            test_set=test_set, eval_every=spec.eval_every, eval_batch_size=resources.eval_batch_size, device=device,
            num_workers=resources.num_workers, on_round_committed=on_round_committed, start_round_index=completed,
            sampling_schedule=federation.schedule,
        )
    return _finalize(run_dir, spec, model, privacy, ledger.total_attempts() if ledger is not None else None,
                     time.perf_counter() - started, log)


def _finalize(run_dir: Path, spec: RunSpec, model, privacy: PrivacyConfiguration, attempts: int | None,
              elapsed: float, log) -> dict:
    """Write final_model.pt, public_summary.json (released values only) and the completion marker."""
    state = cpu_state_dict(model)
    write_bytes_atomic(run_dir / FINAL_MODEL, lambda file: torch.save(state, file))
    released = read_jsonl(run_dir / RELEASED_LOG)
    if [record["round"] for record in released] != list(range(1, spec.config.num_rounds + 1)):
        raise RuntimeError(f"{run_dir}: the released round log is incomplete")
    evaluations = [{"round": r["round"], "test_accuracy": r["test_accuracy"]} for r in released if "test_accuracy" in r]
    summary = {
        "note": "Released outputs only: no seeds, sampled sets or client updates.",
        "method": spec.label,
        "method_key": spec.method,
        "method_parameters": spec.method_parameters,
        "noise_context": spec.noise_context,
        "configuration": spec.configuration_record(),
        "privacy": public_privacy_record(spec.config.accountant_config(), privacy, attempts),
        "final_test_accuracy": evaluations[-1]["test_accuracy"] if evaluations else None,
        "evaluations": evaluations,
        "final_round_released_diagnostics": released[-1],
        "mean_dimension_ratio": float(np.mean([record["D_ratio"] for record in released])),
        "final_model": FINAL_MODEL,
    }
    write_json_atomic(run_dir / PUBLIC_SUMMARY, summary)
    write_json_atomic(run_dir / STATUS_FILE, {"status": "complete", "completed_round": spec.config.num_rounds})
    accuracy = summary["final_test_accuracy"]
    log(f"{spec.label}: COMPLETE | final accuracy {'n/a' if accuracy is None else f'{100 * accuracy:.2f}%'} | "
        f"epsilon {summary['privacy']['epsilon']:.4f} (delta {summary['privacy']['delta']:g}, "
        f"{summary['privacy']['privacy_mode']}) | {elapsed / 3600:.2f} h this session")
    return summary
