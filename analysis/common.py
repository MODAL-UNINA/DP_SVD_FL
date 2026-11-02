"""Shared run discovery, validation and aggregation for the table and figure scripts.

A *run directory* is a directory that contains at least one of ``public_summary.json``,
``released_round_diagnostics.jsonl`` or ``status.json``. Roots are walked recursively; folders named
``internal`` or ``checkpoints`` are not entered. A run is *included* only when

* ``status.json`` says ``"status": "complete"``,
* ``public_summary.json`` is present and readable,
* ``released_round_diagnostics.jsonl`` holds exactly the rounds 1..T (T = 300) in order,
* the ``test_accuracy`` recorded at round T equals ``final_test_accuracy`` of the summary,
* the experiment seed can be read from the path (see ``SEED_PATTERNS``),
* no other complete run of the same campaign, setting and method has the same seed.

Everything else is reported as *excluded* with the reason. Runs of any method other than DP-SVD-FL
(``adaptive``) and its fixed-rank ablation (``fixed_spectral``) are listed separately as *ignored*.

Aggregation conventions (identical to the paper): the final accuracy is the test accuracy recorded at
round 300 (never the best checkpoint); a cell is mean +/- sample standard deviation (ddof = 1) over the
required seeds (0, 1, 2), and a cell is reported only when every required seed is complete. Runs are
grouped by (campaign, dataset, partition, Dirichlet alpha, sigma, method); runs of different campaign
options or different sigma are never pooled.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

EXPECTED_ROUNDS = 300
DEFAULT_SEEDS = (0, 1, 2)

# Method keys of this repository, in table order, with their display names.
METHOD_LABELS = {
    "adaptive": "DP-SVD-FL",
    "fixed_spectral": "Fixed-rank spectral",
}
METHOD_ORDER = list(METHOD_LABELS)

DATASET_LABELS = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "emnist_balanced": "EMNIST Bal."}
DATASET_ORDER = ["cifar10", "cifar100", "emnist_balanced"]
PARTITION_LABELS = {"dirichlet": "Dir.", "iid": "IID"}
PARTITION_ORDER = ["dirichlet", "iid"]

# Campaign options of the command-line interface (one per experiment family of the paper).
CAMPAIGNS = {
    "main": "main experiment (3 datasets x 2 partitions)",
    "privacy_sweep": "privacy sweep (CIFAR-10 Dirichlet)",
    "fixed_rank": "fixed-rank spectral ablation (CIFAR-10 Dirichlet)",
}

RUN_FILES = ("public_summary.json", "released_round_diagnostics.jsonl", "status.json")
SKIP_DIRS = {"internal", "checkpoints"}
# The experiment seed is taken from the nearest ancestor (or the run directory itself) whose name matches.
SEED_PATTERNS = (re.compile(r"^seed(\d+)$"), re.compile(r"^sigma\d+_seed(\d+)$"), re.compile(r"__s(\d+)$"))
# Hints used only to describe runs whose public_summary.json is missing (e.g. runs that failed).
SIGMA_DIR_PATTERN = re.compile(r"(?:^|_)sigma(\d+)(?:_|$)")
SETTING_DIR_PATTERN = re.compile(r"^(cifar100|cifar10|emnist_balanced)_(iid|dirichlet)")
# Configuration entries that must agree inside a group (otherwise the runs are kept in separate groups).
SIGNATURE_KEYS = ("architecture", "T", "delta", "local_epochs", "N", "q", "C")


# ----------------------------------------------------------------------------- data classes
@dataclass
class Run:
    campaign: str
    root_index: int
    root_name: str
    rel_path: str
    method: str
    seed: int
    dataset: str
    partition: str
    alpha: float | None
    sigma: float
    epsilon: float
    delta: float
    signature: tuple
    final_accuracy: float
    records: list = field(repr=False)

    @property
    def setting(self) -> tuple:
        return (self.dataset, self.partition, self.alpha, self.sigma, self.signature)

    @property
    def key(self) -> tuple:
        return (self.campaign,) + self.setting + (self.method,)

    def location(self) -> dict:
        return {"root_index": self.root_index, "root": self.root_name, "path": self.rel_path}

    def evaluations(self) -> list[tuple[int, float]]:
        return [(r["round"], r["test_accuracy"]) for r in self.records if r.get("test_accuracy") is not None]


@dataclass
class Group:
    """All included runs of one (campaign, setting, method)."""

    campaign: str
    dataset: str
    partition: str
    alpha: float | None
    sigma: float
    signature: tuple
    method: str
    runs: dict = field(default_factory=dict)  # seed -> Run

    def complete(self, seeds=DEFAULT_SEEDS) -> bool:
        return all(s in self.runs for s in seeds)

    def selected(self, seeds=DEFAULT_SEEDS) -> list[Run]:
        return [self.runs[s] for s in seeds if s in self.runs]

    def accuracies(self, seeds=DEFAULT_SEEDS) -> dict[int, float]:
        return {s: self.runs[s].final_accuracy for s in seeds if s in self.runs}

    def epsilon(self) -> float | None:
        values = {r.epsilon for r in self.runs.values()}
        return values.pop() if len(values) == 1 else None

    def describe(self) -> dict:
        return {"campaign": self.campaign, "dataset": self.dataset, "partition": self.partition,
                "dirichlet_alpha": self.alpha, "sigma": self.sigma, "method": self.method,
                "seeds_complete": sorted(self.runs)}


@dataclass
class Campaign:
    name: str
    roots: list
    runs: list = field(default_factory=list)
    excluded: list = field(default_factory=list)
    ignored: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    groups: dict = field(default_factory=dict)  # Run.key -> Group

    def find(self, dataset, partition, method, sigma=None) -> list[Group]:
        """Groups of a setting/method, optionally restricted to one sigma."""
        out = [g for g in self.groups.values() if g.dataset == dataset and g.partition == partition
               and g.method == method and (sigma is None or math.isclose(g.sigma, sigma))]
        return sorted(out, key=lambda g: (g.sigma, g.alpha if g.alpha is not None else -1))

    def one(self, dataset, partition, method, sigma=None) -> Group | None:
        groups = self.find(dataset, partition, method, sigma)
        if len(groups) > 1:
            self.warnings.append(f"several groups for {dataset}/{partition}/{method}"
                                 f"{'' if sigma is None else f'/sigma={sigma}'}: "
                                 "they differ in Dirichlet alpha, sigma or configuration and are not pooled; "
                                 "the first one is used")
        return groups[0] if groups else None

    def report(self) -> dict:
        return {
            "roots": [{"root_index": i, "root": Path(r).name} for i, r in enumerate(self.roots)],
            "included_runs": [dict(r.location(), method=r.method, seed=r.seed, dataset=r.dataset,
                                   partition=r.partition, dirichlet_alpha=r.alpha, sigma=r.sigma,
                                   epsilon=r.epsilon, final_test_accuracy=r.final_accuracy)
                              for r in sorted(self.runs, key=lambda r: (r.root_index, r.rel_path))],
            "excluded_runs": sorted(self.excluded, key=lambda e: (e["root_index"], e["path"])),
            "ignored_runs": sorted(self.ignored, key=lambda e: (e["root_index"], e["path"])),
            "groups": [dict(g.describe(), n_seeds=len(g.runs))
                       for g in sorted(self.groups.values(), key=group_sort_key)],
            "warnings": self.warnings,
        }


# ----------------------------------------------------------------------------- discovery
def _read_json(path: Path):
    with path.open() as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _seed_from_path(run_dir: Path, root: Path) -> int | None:
    current = run_dir
    while True:
        for pattern in SEED_PATTERNS:
            m = pattern.search(current.name)
            if m:
                return int(m.group(1))
        if current == root or current.parent == current:
            return None
        current = current.parent


def _path_hints(run_dir: Path, root: Path) -> dict:
    """Best-effort description (dataset, partition, sigma) from directory names, for excluded runs."""
    hints = {}
    current = run_dir
    while True:
        m = SIGMA_DIR_PATTERN.search(current.name)
        if m and "sigma" not in hints:
            hints["sigma"] = int(m.group(1)) / 1000.0
        m = SETTING_DIR_PATTERN.match(current.name)
        if m and "dataset" not in hints:
            hints["dataset"], hints["partition"] = m.group(1), m.group(2)
        if current == root or current.parent == current:
            break
        current = current.parent
    return hints


def iter_run_dirs(root: Path):
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        if any(name in filenames for name in RUN_FILES):
            yield Path(dirpath)


def _norm_alpha(config: dict):
    if config.get("partition") != "dirichlet":
        return None
    alpha = config.get("dirichlet_alpha")
    return None if alpha is None else round(float(alpha), 6)


def load_campaign(name: str, roots, expected_rounds: int = EXPECTED_ROUNDS) -> Campaign:
    """Discover, validate and group every run under ``roots`` (one campaign option)."""
    campaign = Campaign(name=name, roots=[str(r) for r in roots])
    candidates = []
    for index, root in enumerate(roots):
        root = Path(root)
        if not root.is_dir():
            campaign.warnings.append(f"root {index} ({root.name}) is not a directory; skipped")
            continue
        for run_dir in iter_run_dirs(root):
            candidates.append((index, root, run_dir))

    by_key = defaultdict(list)
    for index, root, run_dir in candidates:
        rel = run_dir.relative_to(root).as_posix() or "."
        entry = {"root_index": index, "root": root.name, "path": rel}
        summary = None
        summary_path = run_dir / "public_summary.json"
        if summary_path.is_file():
            try:
                summary = _read_json(summary_path)
            except (OSError, ValueError) as exc:
                entry["reason"] = f"public_summary.json unreadable: {exc.__class__.__name__}"
        method = (summary or {}).get("method_key") or run_dir.name
        entry["method"] = method
        if method not in METHOD_LABELS:
            campaign.ignored.append(dict(entry, reason="not a DP-SVD-FL variant of this repository"))
            continue
        seed = _seed_from_path(run_dir, root)
        entry["seed"] = seed
        config = (summary or {}).get("configuration") or {}
        hints = _path_hints(run_dir, root)
        entry["dataset"] = config.get("dataset", hints.get("dataset"))
        entry["partition"] = config.get("partition", hints.get("partition"))
        entry["sigma"] = config.get("sigma", hints.get("sigma"))
        if not config:
            entry["description_source"] = "directory names (public_summary.json missing)"

        def exclude(reason):
            entry.setdefault("reason", reason)
            campaign.excluded.append(entry)

        records = []
        diag_path = run_dir / "released_round_diagnostics.jsonl"
        if diag_path.is_file():
            try:
                records = _read_jsonl(diag_path)
            except (OSError, ValueError) as exc:
                exclude(f"released_round_diagnostics.jsonl unreadable: {exc.__class__.__name__}")
                continue
        entry["rounds_recorded"] = len(records)
        status_path = run_dir / "status.json"
        if not status_path.is_file():
            exclude("status.json missing (run not completed)")
            continue
        try:
            status = _read_json(status_path)
        except (OSError, ValueError) as exc:
            exclude(f"status.json unreadable: {exc.__class__.__name__}")
            continue
        if status.get("status") != "complete":
            exclude(f"status is {status.get('status')!r}, not 'complete'")
            continue
        if summary is None:
            exclude("public_summary.json missing")
            continue
        if seed is None:
            exclude("experiment seed not found in the directory names")
            continue
        T = config.get("T")
        if T != expected_rounds:
            exclude(f"configured T = {T}, expected {expected_rounds}")
            continue
        if [r.get("round") for r in records] != list(range(1, expected_rounds + 1)):
            exclude(f"round records are not exactly 1..{expected_rounds}")
            continue
        final_eval = records[-1].get("test_accuracy")
        if final_eval is None or final_eval != summary.get("final_test_accuracy"):
            exclude("test accuracy at the final round differs from final_test_accuracy")
            continue
        privacy = summary.get("privacy") or {}
        run = Run(campaign=name, root_index=index, root_name=root.name, rel_path=rel, method=method, seed=seed,
                  dataset=config["dataset"], partition=config["partition"], alpha=_norm_alpha(config),
                  sigma=round(float(config["sigma"]), 6), epsilon=float(privacy.get("epsilon", float("nan"))),
                  delta=float(privacy.get("delta", config.get("delta", float("nan")))),
                  signature=tuple((k, config.get(k)) for k in SIGNATURE_KEYS),
                  final_accuracy=float(final_eval), records=records)
        by_key[(run.key, seed)].append((run, entry))

    for (key, seed), items in by_key.items():
        if len(items) > 1:
            paths = [f"{e['root']}/{e['path']}" for _, e in items]
            for _, e in items:
                e["reason"] = f"duplicate complete runs for the same seed and setting: {paths}"
                campaign.excluded.append(e)
            continue
        run = items[0][0]
        campaign.runs.append(run)
        group = campaign.groups.get(key)
        if group is None:
            group = campaign.groups[key] = Group(campaign=name, dataset=run.dataset, partition=run.partition,
                                                 alpha=run.alpha, sigma=run.sigma, signature=run.signature,
                                                 method=run.method)
        group.runs[seed] = run

    # Settings that appear with more than one configuration signature are kept apart; say so.
    signatures = defaultdict(set)
    for g in campaign.groups.values():
        signatures[(g.dataset, g.partition, g.alpha, g.sigma)].add(g.signature)
    for setting, sigs in signatures.items():
        if len(sigs) > 1:
            campaign.warnings.append(f"setting {setting} has {len(sigs)} different configurations "
                                     f"({list(SIGNATURE_KEYS)}); they are kept in separate groups")
    for g in campaign.groups.values():
        if g.epsilon() is None:
            campaign.warnings.append(f"group {g.describe()} has different recorded epsilon across seeds")
    return campaign


def group_sort_key(g: Group):
    return (DATASET_ORDER.index(g.dataset) if g.dataset in DATASET_ORDER else 99,
            PARTITION_ORDER.index(g.partition) if g.partition in PARTITION_ORDER else 99,
            g.alpha if g.alpha is not None else -1, g.sigma,
            METHOD_ORDER.index(g.method) if g.method in METHOD_ORDER else 99)


# ----------------------------------------------------------------------------- statistics
def mean_std(values) -> tuple[float | None, float | None]:
    """Mean and sample standard deviation (ddof = 1); std is None for fewer than two values."""
    values = list(values)
    if not values:
        return None, None
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else None)


def summarize_group(group: Group, seeds=DEFAULT_SEEDS) -> dict:
    """Final-accuracy statistics of a group; ``complete`` is False when a required seed is missing."""
    accs = group.accuracies(seeds)
    complete = group.complete(seeds)
    mean, std = mean_std(accs.values()) if complete else (None, None)
    return {**group.describe(), "epsilon": group.epsilon(), "required_seeds": list(seeds),
            "complete": complete, "n_seeds": len(accs), "per_seed_accuracy": accs,
            "accuracy_mean": mean, "accuracy_std": std,
            "runs": [group.runs[s].location() for s in seeds if s in group.runs]}


def evaluation_curve(runs: list[Run]) -> dict:
    """Mean and sample std of the recorded test accuracy at every evaluation round (no smoothing).

    Every run must have been evaluated at the same rounds.
    """
    evals = [dict(r.evaluations()) for r in runs]
    rounds = sorted(evals[0]) if evals else []
    if any(sorted(e) != rounds for e in evals):
        raise ValueError("runs were evaluated at different rounds")
    means, stds = [], []
    for rnd in rounds:
        m, s = mean_std(e[rnd] for e in evals)
        means.append(m)
        stds.append(s)
    return {"rounds": rounds, "mean": means, "std": stds}


def dimension_series(runs: list[Run]) -> dict:
    """Per-round privatized dimension D_t and full dimension D_full of a set of runs (one per seed)."""
    D = [[rec["D"] for rec in r.records] for r in runs]
    D_full = sorted({rec["D_full"] for r in runs for rec in r.records})
    return {"D": D, "D_full": D_full}


# ----------------------------------------------------------------------------- formatting
def pct(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{100.0 * x:.{digits}f}"


def pm(mean: float | None, std: float | None, digits: int = 2, signed: bool = False, sep: str = " ± ") -> str:
    if mean is None:
        return "n/a"
    m = f"{100.0 * mean:+.{digits}f}".replace("-", "\u2212") if signed else f"{100.0 * mean:.{digits}f}"
    return m if std is None else f"{m}{sep}{100.0 * std:.{digits}f}"


def setting_label(dataset: str, partition: str) -> str:
    return f"{DATASET_LABELS.get(dataset, dataset)} {PARTITION_LABELS.get(partition, partition)}"
