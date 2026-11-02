#!/usr/bin/env python
"""Empirical audit of the adaptive rank rule (Supplementary Sec. S10.1-S10.4). Read-only on its inputs, CPU only.

Inputs are adaptive DP-SVD-FL run directories, found by recursive discovery under one or more roots (``--runs``). A run
directory holds ``public_summary.json`` with ``method_key == "adaptive"``, ``released_round_diagnostics.jsonl``,
``benchmark_diagnostics_internal.jsonl`` and ``status.json``. The experiment seed is read from ``run_config.json``
(``experiment_seed``) when present, otherwise from the nearest ancestor directory named ``seed<k>``.

Everything reported is either read from the released per-round log (rank k_l(t), branch, maximal consecutive
singular-value ratio, D(t)) or an exact arithmetic identity of the method:

    D(t) = U + sum_l k_l(t)^2,   U = D_full - sum_l m_l n_l   (the non-projected coordinates: fc weight and bias)

The layer geometry (m_l x n_l) and the initial ranks k_l^(0) are NOT read from the run directories: they are
recomputed from the public initial weights W0 (torchvision ``resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)``,
which honours ``TORCH_HOME``), with the same code path as the training pipeline (matricise -> float64 SVD ->
``select_rank``). Round 1 of every released log is the rule applied to W0, so the recomputation must reproduce the
logged first-round rank, branch and maximal gap ratio exactly (ratio to a relative tolerance of 1e-9).

With ``--final-models`` the full rule outcome (k*, k95, retained energy, ...) is also evaluated on each run's
``final_model.pt`` (W^(T), the model after the last round; never used by a logged round).

Outputs (``--out DIR``, must be empty or absent):
    checks.json                     every consistency check of every run (PASS / FAIL)
    tables/*.csv                    inventory, layer geometry, branch usage, rank statistics, D(t) verification,
                                    energy-branch share of D, decomposition of the decrease of D, rank-change events,
                                    gap-ratio margins, noise vs signal scale of the released update
    tables/per_round_layer_long.csv one row per (run, round, layer)
    endpoints/rule_at_W0.csv        rule outcome at W0 and its comparison with round 1 of every log
    endpoints/rule_at_WT.csv        (only with --final-models)
    summary.json                    the headline numbers of Supplementary S10.1-S10.4
    summary.md                      the same, as a Markdown report

Example:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src \\
        python analysis/rank_audit.py --runs outputs/runs --out results/rank_audit [--final-models]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only, before torch is imported

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # without `pip install -e .`
from dpsvdfl.models.partitions import RESNET18_PROJECTED_LAYERS  # noqa: E402
from dpsvdfl.spectral.matricization import matricize  # noqa: E402
from dpsvdfl.spectral.projection import SVD_DTYPE, reduced_svd  # noqa: E402
from dpsvdfl.spectral.rank_rule import ENERGY_BRANCH, FALLBACK_BRANCH, GAMMA, GAP_BRANCH, RHO_MAX, TAU  # noqa: E402
from dpsvdfl.spectral.rank_rule import energy_rank, select_rank  # noqa: E402

PROJECTED = tuple(f"{name}.weight" for name in RESNET18_PROJECTED_LAYERS)
BRANCHES = (GAP_BRANCH, ENERGY_BRANCH, FALLBACK_BRANCH)
DATASET_ORDER = ("cifar10", "cifar100", "emnist_balanced")
PARTITION_ORDER = ("dirichlet0.6", "iid")
DATASET_LABEL = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "emnist_balanced": "EMNIST-Balanced"}
FC_IN_PLUS_BIAS = 513  # fc: 512 inputs + 1 bias per class
GAP_RATIO_RTOL = 1e-9  # relative tolerance of the W0 recomputation of the logged first-round gap ratio
SEED_DIR = re.compile(r"^seed(\d+)$")


def short_name(layer: str) -> str:
    """Compact label: layer4.0.downsample.0.weight -> L4.0.ds, layer3.1.conv2.weight -> L3.1.c2."""
    return layer.replace("layer", "L").replace(".weight", "").replace("downsample.0", "ds").replace("conv", "c")


class AuditError(RuntimeError):
    pass


# ---------------------------------------------------------------- discovery


def partition_label(config: dict) -> str:
    if config["partition"] == "dirichlet":
        return f"dirichlet{config['dirichlet_alpha']}"
    return config["partition"]


def seed_of(run_dir: Path) -> int:
    """experiment_seed of run_config.json if present, else the nearest ancestor directory named seed<k>."""
    from_dir = None
    for parent in (run_dir, *run_dir.parents):
        match = SEED_DIR.match(parent.name)
        if match:
            from_dir = int(match.group(1))
            break
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        seed = json.loads(config_path.read_text()).get("experiment_seed")
        if seed is not None:
            if from_dir is not None and from_dir != int(seed):
                raise AuditError(f"{run_dir}: run_config.json seed {seed} != directory seed{from_dir}")
            return int(seed)
    if from_dir is None:
        raise AuditError(f"{run_dir}: no experiment seed (no run_config.json and no seed<k> ancestor directory)")
    return from_dir


def run_sort_key(run: dict) -> tuple:
    d, p = run["dataset"], run["partition"]
    return (DATASET_ORDER.index(d) if d in DATASET_ORDER else len(DATASET_ORDER), d,
            PARTITION_ORDER.index(p) if p in PARTITION_ORDER else len(PARTITION_ORDER), p, run["seed"])


def discover(roots: list[Path]) -> list[dict]:
    """Every adaptive DP-SVD-FL run directory below the given roots (each directory once)."""
    seen, runs = set(), []
    for root in roots:
        for summary_path in sorted(root.rglob("public_summary.json")):
            run_dir = summary_path.parent.resolve()
            if run_dir in seen:
                continue
            summary = json.loads(summary_path.read_text())
            if summary.get("method_key") != "adaptive":
                continue
            seen.add(run_dir)
            config = summary["configuration"]
            runs.append({"dir": run_dir, "root": root.resolve(), "dataset": config["dataset"],
                         "partition": partition_label(config), "seed": seed_of(run_dir), "cfg": config})
    runs.sort(key=run_sort_key)
    keys = [(r["dataset"], r["partition"], r["seed"]) for r in runs]
    duplicates = [k for k, c in Counter(keys).items() if c > 1]
    if duplicates:
        raise AuditError(f"several adaptive runs share (dataset, partition, seed): {duplicates}")
    return runs


# ---------------------------------------------------------------- W0 and the rank rule


def load_w0_state() -> dict[str, torch.Tensor]:
    """Public ImageNet ResNet-18 weights (torchvision IMAGENET1K_V1; downloaded to / read from TORCH_HOME)."""
    from torchvision.models import ResNet18_Weights, resnet18

    return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()


def rule_rows(state: dict, layers: tuple[str, ...], label: dict) -> list[dict]:
    """Full rank-rule outcome per layer: matricise -> float64 SVD -> select_rank (the pipeline's code path)."""
    rows = []
    for name in layers:
        W = matricize(state[name].detach().to("cpu"))
        m, n = W.shape
        _, s, _ = reduced_svd(W.to(SVD_DTYPE))
        rr = select_rank(s, m, n)
        sl = s.tolist()
        rows.append({**label, "layer": name, "m": m, "n": n, "selected_k": rr.selected_rank, "branch": rr.branch,
                     "k_star": rr.gap_rank, "max_gap_ratio": rr.max_gap_ratio,
                     "k_star_over_sqrt_mn": rr.gap_rank_fraction, "tau_ok": rr.max_gap_ratio >= TAU,
                     "rho_ok": rr.gap_rank_fraction < RHO_MAX, "k95": rr.energy_rank,
                     "retained_energy_selected": rr.selected_energy_fraction,
                     "k90": energy_rank(sl, 0.90), "k99": energy_rank(sl, 0.99),
                     "s1_over_s2": sl[0] / sl[1], "stable_rank": sum(v * v for v in sl) / sl[0] ** 2})
    return rows


# ---------------------------------------------------------------- loading + validation


def load_run(run: dict, geometry: dict, w0_rule: dict) -> dict:
    """Read one run and run every consistency check; raise AuditError on the first violation."""
    cfg, run_dir = run["cfg"], run["dir"]
    tag = f"{run['dataset']}/{run['partition']}/seed{run['seed']}"
    checks = {}

    def check(name, ok, detail=""):
        checks[name] = "PASS" if ok else f"FAIL {detail}"
        if not ok:
            raise AuditError(f"{tag}: {name}: {detail}")

    T = int(cfg["T"])
    check("rank-rule parameters tau / rho_max / gamma = public values",
          (cfg["tau"], cfg["rho_max"], cfg["gamma"]) == (TAU, RHO_MAX, GAMMA),
          str((cfg["tau"], cfg["rho_max"], cfg["gamma"])))
    status = json.loads((run_dir / "status.json").read_text())
    check("status complete, completed round = T", status.get("status") == "complete"
          and status.get("completed_round", T) == T, str(status))
    config_path = run_dir / "run_config.json"
    if config_path.exists():  # public layout only: the stored snapshot must equal the W0 recomputation
        snapshot = json.loads(config_path.read_text()).get("initial_rank_snapshot")
        if snapshot is not None:
            entries = snapshot.get("layers", snapshot.get("snapshot", {}).get("layers", []))
            stored = {e["name"]: (e["rank"], e["branch"]) for e in entries}
            check("stored initial rank snapshot = W0 recomputation",
                  stored == {n: (w0_rule[n]["selected_k"], w0_rule[n]["branch"]) for n in PROJECTED})

    log = run_dir / "released_round_diagnostics.jsonl"
    records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    check("rounds 1..T present exactly once", [r["round"] for r in records] == list(range(1, T + 1)))
    layers = PROJECTED
    L = len(layers)
    K = np.zeros((T, L), dtype=np.int64)
    G = np.full((T, L), np.nan)
    B = np.empty((T, L), dtype=object)
    D = np.array([r["D"] for r in records], dtype=np.int64)
    Dfull = np.array([r["D_full"] for r in records], dtype=np.int64)
    Dratio = np.array([r["D_ratio"] for r in records], dtype=float)
    for t, r in enumerate(records):
        check("ranks / branches / gap ratios hold exactly the projected layers",
              set(r["ranks"]) == set(r["rank_branches"]) == set(r["gap_ratios"]) == set(layers), f"round {t + 1}")
        for j, name in enumerate(layers):
            K[t, j] = r["ranks"][name]
            B[t, j] = r["rank_branches"][name]
            G[t, j] = np.nan if r["gap_ratios"][name] is None else r["gap_ratios"][name]
    mn = np.array([geometry[n][0] * geometry[n][1] for n in layers], dtype=np.int64)
    rmax = np.array([min(geometry[n]) for n in layers], dtype=np.int64)
    sqrt_mn = np.sqrt(mn.astype(float))
    check("1 <= k <= min(m, n)", bool(((K >= 1) & (K <= rmax)).all()))
    check("branch in {gap, energy, fallback}", set(B.ravel()) <= set(BRANCHES), str(set(B.ravel())))
    gap = B == GAP_BRANCH
    check("gap branch => ratio >= tau and k / sqrt(mn) < rho_max",
          bool(((G[gap] >= TAU) & ((K / sqrt_mn)[gap] < RHO_MAX)).all()))
    check("fallback branch => k = 1", bool((K[B == FALLBACK_BRANCH] == 1).all()))

    U = int(Dfull[0] - mn.sum())
    check("D_full constant", bool((Dfull == Dfull[0]).all()))
    D_rec = U + (K ** 2).sum(axis=1)
    check("D(t) = U + sum_l k_l(t)^2 for every round", bool((D_rec == D).all()),
          f"max |diff| {int(np.abs(D_rec - D).max())}")
    check("D_ratio = D / D_full", bool(np.allclose(Dratio, D / Dfull, rtol=0, atol=1e-12)))
    classes = U / FC_IN_PLUS_BIAS
    check("U = 513 x number of classes (fc 512 -> classes, with bias)", float(classes).is_integer(), str(U))
    try:
        from dpsvdfl.data.datasets import num_classes
        expected_classes = num_classes(run["dataset"])
    except (ImportError, ValueError, KeyError):
        expected_classes = None
    if expected_classes is not None:
        check("number of classes = dataset classes", int(classes) == expected_classes, f"{classes} vs {expected_classes}")

    w0_k = [w0_rule[n]["selected_k"] for n in layers]
    w0_b = [w0_rule[n]["branch"] for n in layers]
    w0_g = np.array([w0_rule[n]["max_gap_ratio"] for n in layers], dtype=float)
    check("round 1 = rank rule recomputed on W0 (rank, branch, gap ratio)",
          list(K[0]) == w0_k and list(B[0]) == w0_b
          and bool((np.abs(w0_g - G[0]) <= GAP_RATIO_RTOL * w0_g).all()))
    return {**run, "tag": tag, "layers": layers, "mn": mn, "rmax": rmax, "sqrt_mn": sqrt_mn, "T": T,
            "K": K, "B": B, "G": G, "D": D, "Dfull": Dfull, "U": U, "classes": int(classes), "checks": checks,
            "log": log, "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()}


# ---------------------------------------------------------------- helpers


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict], columns: list[str] | None = None) -> str:
    columns = columns or list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def fmt(x, nd=1):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def config_key(run):
    return (run["dataset"], run["partition"])


def config_label(key):
    return f"{DATASET_LABEL.get(key[0], key[0])} {key[1]}"


def events(run) -> list[dict]:
    """Every (round t >= 2, layer) where k changes w.r.t. round t-1; dD = k(t)^2 - k(t-1)^2 (signed)."""
    K, B = run["K"], run["B"]
    out = []
    for t in range(1, K.shape[0]):
        for j, name in enumerate(run["layers"]):
            if K[t, j] != K[t - 1, j]:
                out.append({"round": t + 1, "layer": name, "k_prev": int(K[t - 1, j]), "k": int(K[t, j]),
                            "dk": int(K[t, j] - K[t - 1, j]), "dD": int(K[t, j] ** 2 - K[t - 1, j] ** 2),
                            "branch_prev": B[t - 1, j], "branch": B[t, j]})
    return out


def span(values, nd=3) -> str:
    lo, hi = min(values), max(values)
    return f"{lo:.{nd}f}" if f"{lo:.{nd}f}" == f"{hi:.{nd}f}" else f"{lo:.{nd}f}-{hi:.{nd}f}"


# ---------------------------------------------------------------- noise vs signal scale (S10.4)


def noise_share_rows(runs: list[dict]) -> list[dict]:
    """Released update norm vs the expected DP-noise norm sigma C sqrt(D(t)) / (qN), per run.

    The released reduced update is (sum_i clip_C(s_i) + z) / (qN), z ~ N(0, sigma^2 C^2 I_D(t)). Hence
      noise norm   ||z|| / (qN)  ~=  sigma C sqrt(D(t)) / (qN)
      signal norm  ||sum_i clip_C(s_i)|| / (qN)  <=  |S_t| C / (qN)  (triangle inequality; |S_t| = sampled slots)
    The recorded server_update_l2_norm is measured in weight space; the reduced -> weight map is an isometry
    (orthonormal U_k, V_k on projected blocks, identity elsewhere) and the server learning rate must be 1.
    """
    rows = []
    for r in runs:
        cfg = r["cfg"]
        if cfg.get("server_learning_rate", 1.0) != 1.0:
            raise AuditError(f"{r['tag']}: server learning rate {cfg['server_learning_rate']} != 1")
        qN = cfg["q"] * cfg["N"]
        bench_path = r["dir"] / "benchmark_diagnostics_internal.jsonl"
        bench = [json.loads(line) for line in bench_path.open() if line.strip()]
        if [b["round"] for b in bench] != list(range(1, r["T"] + 1)):
            raise AuditError(f"{r['tag']}: benchmark rounds are not 1..T")
        noise = cfg["sigma"] * cfg["C"] * np.sqrt(r["D"].astype(float)) / qN
        signal_bound = np.array([b["num_sampled_slots"] for b in bench]) * cfg["C"] / qN
        update = np.array([b["server_update_l2_norm"] for b in bench])
        norm_before = bench[0]["trainable_model_l2_norm_before"]
        norm_after = bench[-1]["trainable_model_l2_norm_after"]
        growth = norm_after ** 2 - norm_before ** 2
        cumulative = float((noise ** 2).sum())
        rows.append({"dataset": r["dataset"], "partition": r["partition"], "seed": r["seed"], "sigma": cfg["sigma"],
                     "C": cfg["C"], "qN": qN, "update_over_noise_mean": float((update / noise).mean()),
                     "update_over_noise_min": float((update / noise).min()),
                     "update_over_noise_max": float((update / noise).max()),
                     "signal_bound_over_noise_mean": float((signal_bound / noise).mean()),
                     "signal_bound_over_noise_max": float((signal_bound / noise).max()),
                     "trainable_norm_round0": norm_before, "trainable_norm_roundT": norm_after,
                     "trainable_norm2_growth": growth, "cumulative_expected_noise_energy": cumulative,
                     "growth_over_noise_energy_minus_1": growth / cumulative - 1})
    return rows


# ---------------------------------------------------------------- main


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True,
                        help="root directories searched recursively for adaptive run directories")
    parser.add_argument("--out", type=Path, required=True, help="output directory (must be empty or absent)")
    parser.add_argument("--final-models", action="store_true",
                        help="also evaluate the rank rule on every run's final_model.pt")
    parser.add_argument("--threads", type=int, default=4, help="torch CPU threads (default 4)")
    args = parser.parse_args(argv)
    out = args.out
    if out.exists() and any(out.iterdir()):
        print(f"refused: {out} exists and is not empty")
        return 2
    torch.set_num_threads(args.threads)
    tables, endpoints = out / "tables", out / "endpoints"
    tables.mkdir(parents=True, exist_ok=True)
    endpoints.mkdir(parents=True, exist_ok=True)

    # ---- W0: layer geometry and initial ranks (public, data-independent)
    w0 = load_w0_state()
    rows0 = rule_rows(w0, PROJECTED, {"model": "W0"})
    w0_rule = {row["layer"]: row for row in rows0}
    geometry = {row["layer"]: (row["m"], row["n"]) for row in rows0}

    discovered = discover(args.runs)
    if not discovered:
        print("no adaptive run directory found")
        return 1
    runs = [load_run(r, geometry, w0_rule) for r in discovered]
    layers = PROJECTED
    T_values = sorted({r["T"] for r in runs})
    print(f"{len(runs)} runs loaded and validated (T = {T_values})")
    md = []

    # ---- W0 vs round 1 of every log (rank, branch, gap ratio)
    w0_cmp = []
    for r in runs:
        for j, name in enumerate(layers):
            row = w0_rule[name]
            w0_cmp.append({"dataset": r["dataset"], "partition": r["partition"], "seed": r["seed"], "layer": name,
                           "k_w0": row["selected_k"], "k_log_round1": int(r["K"][0, j]), "branch_w0": row["branch"],
                           "branch_log_round1": r["B"][0, j], "gap_ratio_w0": row["max_gap_ratio"],
                           "gap_ratio_log_round1": r["G"][0, j],
                           "gap_ratio_abs_diff": abs(row["max_gap_ratio"] - r["G"][0, j]),
                           "match": (row["selected_k"], row["branch"]) == (int(r["K"][0, j]), r["B"][0, j])
                           and abs(row["max_gap_ratio"] - r["G"][0, j]) <= GAP_RATIO_RTOL * row["max_gap_ratio"]})
    write_csv(endpoints / "rule_at_W0.csv", rows0)
    write_csv(endpoints / "W0_vs_round1_logs.csv", w0_cmp)
    w0_all_match = all(row["match"] for row in w0_cmp)

    # ---- T0 inventory, layer geometry
    inv = []
    for r in runs:
        inv.append({"dataset": r["dataset"], "partition": r["partition"], "seed": r["seed"], "rounds": r["T"],
                    "local_epochs": r["cfg"]["local_epochs"], "tau": r["cfg"]["tau"], "rho_max": r["cfg"]["rho_max"],
                    "gamma": r["cfg"]["gamma"], "sigma": r["cfg"]["sigma"], "U": r["U"], "D_full": int(r["Dfull"][0]),
                    "checks": f"{sum(v == 'PASS' for v in r['checks'].values())}/{len(r['checks'])} PASS",
                    "log": str(r["log"].relative_to(r["root"])), "log_sha256": r["log_sha256"]})
    write_csv(tables / "T0_inventory.csv", inv)
    md.append("## Runs and consistency checks\n\n" + md_table(inv, ["dataset", "partition", "seed", "rounds",
                                                                     "local_epochs", "sigma", "U", "D_full", "checks"]))
    md.append("Checks per run: " + "; ".join(runs[0]["checks"]))
    shape_rows = [{"layer": n, "short": short_name(n), "m x n": f"{geometry[n][0]} x {geometry[n][1]}",
                   "min(m,n)": int(runs[0]["rmax"][j]), "m n": int(runs[0]["mn"][j]),
                   "rho_max sqrt(mn)": fmt(RHO_MAX * runs[0]["sqrt_mn"][j], 2)} for j, n in enumerate(layers)]
    write_csv(tables / "T0b_layer_geometry.csv", shape_rows)

    # ---- per-round long table
    long_rows = []
    for r in runs:
        for t in range(r["T"]):
            for j, n in enumerate(layers):
                long_rows.append({"dataset": r["dataset"], "partition": r["partition"], "seed": r["seed"],
                                  "round": t + 1, "layer": n, "m": geometry[n][0], "n": geometry[n][1],
                                  "k": int(r["K"][t, j]), "branch": r["B"][t, j], "gap_ratio": r["G"][t, j],
                                  "k_squared": int(r["K"][t, j] ** 2), "k_over_sqrt_mn": r["K"][t, j] / r["sqrt_mn"][j],
                                  "D": int(r["D"][t])})
    write_csv(tables / "per_round_layer_long.csv", long_rows)

    by_cfg = defaultdict(list)
    for r in runs:
        by_cfg[config_key(r)].append(r)
    cfg_keys = sorted(by_cfg, key=lambda k: run_sort_key({"dataset": k[0], "partition": k[1], "seed": 0}))

    # ---- T1 branch usage per layer x configuration
    t1 = []
    for key in cfg_keys:
        for j, n in enumerate(layers):
            b = np.concatenate([r["B"][:, j] for r in by_cfg[key]])
            c = Counter(b)
            t1.append({"dataset": key[0], "partition": key[1], "layer": n, "decisions": len(b), "gap": c[GAP_BRANCH],
                       "energy": c[ENERGY_BRANCH], "fallback": c[FALLBACK_BRANCH],
                       "gap_pct": 100 * c[GAP_BRANCH] / len(b), "energy_pct": 100 * c[ENERGY_BRANCH] / len(b)})
    write_csv(tables / "T1_branch_usage_layer_config.csv", t1)
    totals = Counter(np.concatenate([r["B"].ravel() for r in runs]))
    n_dec = sum(totals.values())
    per_layer_branch = []
    for j, n in enumerate(layers):
        c = Counter(np.concatenate([r["B"][:, j] for r in runs]))
        per_layer_branch.append({"layer": n, "decisions": sum(c.values()), "gap": c[GAP_BRANCH],
                                 "energy": c[ENERGY_BRANCH], "fallback": c[FALLBACK_BRANCH]})
    write_csv(tables / "T1c_branch_usage_per_layer.csv", per_layer_branch)

    # ---- T2 rank statistics per layer (over all runs)
    t2 = []
    for j, n in enumerate(layers):
        Ks = np.stack([r["K"][:, j] for r in runs])  # runs x T
        t2.append({"layer": n, "short": short_name(n), "min(m,n)": int(runs[0]["rmax"][j]),
                   "k_1": "/".join(sorted({str(int(v)) for v in Ks[:, 0]})),
                   "k_T_min": int(Ks[:, -1].min()), "k_T_max": int(Ks[:, -1].max()),
                   "k_T_range": int(Ks[:, -1].max() - Ks[:, -1].min()), "k_T_mean": float(Ks[:, -1].mean()),
                   "k_min_over_runs_and_rounds": int(Ks.min()), "k_max_over_runs_and_rounds": int(Ks.max()),
                   "k_T_over_k_1_mean": float(Ks[:, -1].mean() / Ks[:, 0].mean()),
                   "k_T_over_min_mn_mean": float(Ks[:, -1].mean() / runs[0]["rmax"][j])})
    write_csv(tables / "T2_rank_stats.csv", t2)

    # ---- T3 D verification and endpoints
    t3 = []
    for r in runs:
        D_rec = r["U"] + (r["K"] ** 2).sum(axis=1)
        t3.append({"dataset": r["dataset"], "partition": r["partition"], "seed": r["seed"], "rounds_checked": r["T"],
                   "U": r["U"], "classes": r["classes"], "max_abs_D_minus_U_plus_sum_k2": int(np.abs(D_rec - r["D"]).max()),
                   "D_1": int(r["D"][0]), "D_T": int(r["D"][-1]), "D_T_over_D_1": r["D"][-1] / r["D"][0],
                   "reduction_pct": 100 * (1 - r["D"][-1] / r["D"][0]), "D_1_over_D_full": r["D"][0] / r["Dfull"][0],
                   "D_T_over_D_full": r["D"][-1] / r["Dfull"][0]})
    write_csv(tables / "T3_D_verification.csv", t3)
    table_s2 = []
    for d in sorted({r["dataset"] for r in runs}, key=lambda d: run_sort_key({"dataset": d, "partition": "", "seed": 0})):
        rs = [r for r in runs if r["dataset"] == d]
        table_s2.append({"dataset": d, "runs": len(rs), "D_1": "/".join(sorted({str(int(r["D"][0])) for r in rs})),
                         "D_T_min": int(min(r["D"][-1] for r in rs)), "D_T_max": int(max(r["D"][-1] for r in rs)),
                         "D_T_over_D_1_min": float(min(r["D"][-1] / r["D"][0] for r in rs)),
                         "D_T_over_D_1_max": float(max(r["D"][-1] / r["D"][0] for r in rs))})
    write_csv(tables / "TableS2_projected_dimension_per_dataset.csv", table_s2)

    # ---- T4 share of D and of the projected part held by layers on the energy branch
    t4 = []
    for r in runs:
        k2 = (r["K"] ** 2).astype(float)
        energy_mask = r["B"] == ENERGY_BRANCH
        proj = k2.sum(axis=1)
        e_share_D = (k2 * energy_mask).sum(axis=1) / r["D"]
        e_share_proj = (k2 * energy_mask).sum(axis=1) / proj
        t4.append({"dataset": r["dataset"], "partition": r["partition"], "seed": r["seed"],
                   "energy_share_of_D_round1_pct": 100 * e_share_D[0], "energy_share_of_D_roundT_pct": 100 * e_share_D[-1],
                   "energy_share_of_D_mean_pct": 100 * e_share_D.mean(),
                   "energy_share_of_projected_mean_pct": 100 * e_share_proj.mean(),
                   "energy_share_of_projected_min_pct": 100 * e_share_proj.min(),
                   "gap_branch_k2_mean": float((k2 * ~energy_mask).sum(axis=1).mean()),
                   "unprojected_U_share_of_D_mean_pct": 100 * float((r["U"] / r["D"]).mean())})
    write_csv(tables / "T4_energy_branch_share_of_D.csv", t4)

    # ---- T5 decomposition of D_1 - D_T by layer
    t5, share_matrix = [], []
    for r in runs:
        dd = (r["K"][0] ** 2 - r["K"][-1] ** 2).astype(float)  # positive = reduction
        share_matrix.append(dd / dd.sum())
        for j, n in enumerate(layers):
            t5.append({"dataset": r["dataset"], "partition": r["partition"], "seed": r["seed"], "layer": n,
                       "k_1": int(r["K"][0, j]), "k_T": int(r["K"][-1, j]), "reduction_k2": int(dd[j]),
                       "share_of_total_reduction": dd[j] / dd.sum()})
    write_csv(tables / "T5_D_reduction_by_layer.csv", t5)
    share_matrix = np.array(share_matrix)
    mean_share = share_matrix.mean(axis=0)
    order = np.argsort(-mean_share)
    top3 = [layers[j] for j in order[:3]]
    top3_per_run = share_matrix[:, order[:3]].sum(axis=1)
    t5_layer = [{"layer": layers[j], "mean_share_pct": 100 * mean_share[j],
                 "min_share_pct": 100 * share_matrix[:, j].min(), "max_share_pct": 100 * share_matrix[:, j].max()}
                for j in order]
    write_csv(tables / "T5b_D_reduction_share_per_layer.csv", t5_layer)

    # ---- T6 rank-change events
    t6, all_events = [], []
    for r in runs:
        ev = events(r)
        for e in ev:
            all_events.append({"dataset": r["dataset"], "partition": r["partition"], "seed": r["seed"], **e})
        dd = np.array([-e["dD"] for e in ev], dtype=float)  # positive = reduction of D
        dk = np.array([e["dk"] for e in ev])
        red = np.sort(dd[dd > 0])[::-1]
        total_red = dd.sum()
        cum = np.cumsum(red) / red.sum() if red.size else np.array([])
        switches = [e for e in ev if e["branch"] != e["branch_prev"]]
        t6.append({"dataset": r["dataset"], "partition": r["partition"], "seed": r["seed"], "events": len(ev),
                   "decreases": int((dk < 0).sum()), "increases": int((dk > 0).sum()),
                   "abs_dk_eq_1_pct": 100 * float((np.abs(dk) == 1).mean()) if ev else math.nan,
                   "abs_dk_max": int(np.abs(dk).max()) if ev else 0,
                   "rounds_with_change": len({e["round"] for e in ev}), "net_D_reduction": int(total_red),
                   "largest_event_pct_of_net": 100 * float(red[0] / total_red) if red.size else math.nan,
                   "events_for_50pct_of_decreases": int(np.searchsorted(cum, 0.5) + 1) if red.size else 0,
                   "branch_switches": len(switches)})
    write_csv(tables / "T6_rank_change_events_summary.csv", t6)
    if all_events:
        write_csv(tables / "T6b_rank_change_events_long.csv", all_events)

    # ---- T7 recorded max gap ratio vs tau, per layer (all runs)
    t7 = []
    for j, n in enumerate(layers):
        G = np.concatenate([r["G"][:, j] for r in runs])
        Bj = np.concatenate([r["B"][:, j] for r in runs])
        e = Bj == ENERGY_BRANCH
        ge = G[e]
        t7.append({"layer": n, "energy_decisions": int(e.sum()),
                   "energy_with_ratio_lt_tau": int((ge < TAU).sum()),
                   "energy_with_ratio_ge_tau (rho_max binding)": int((ge >= TAU).sum()),
                   "J_empty": int(np.isnan(G).sum()), "ratio_median": float(np.nanmedian(G)),
                   "ratio_min": float(np.nanmin(G)), "ratio_max": float(np.nanmax(G))})
    write_csv(tables / "T7_gap_ratio_vs_tau.csv", t7)

    # ---- T8 gap-branch decisions: k, k/sqrt(mn), ratio
    t8 = []
    for j, n in enumerate(layers):
        K = np.concatenate([r["K"][:, j] for r in runs])
        G = np.concatenate([r["G"][:, j] for r in runs])
        Bj = np.concatenate([r["B"][:, j] for r in runs])
        g = Bj == GAP_BRANCH
        if g.any():
            t8.append({"layer": n, "gap_decisions": int(g.sum()), "k_values": "/".join(map(str, sorted(set(K[g].tolist())))),
                       "k_over_sqrt_mn_max": float((K[g] / runs[0]["sqrt_mn"][j]).max()),
                       "ratio_min": float(G[g].min()), "ratio_max": float(G[g].max())})
    if t8:
        write_csv(tables / "T8_gap_branch_margins.csv", t8)

    # ---- noise vs signal scale of the released update (S10.4)
    ns = noise_share_rows(runs)
    write_csv(tables / "T9_noise_share.csv", ns)

    # ---- optional: rule at the final models W^(T)
    wt_summary = None
    if args.final_models:
        rows_t = []
        for r in runs:
            state = torch.load(r["dir"] / "final_model.pt", map_location="cpu", weights_only=True)
            rows = rule_rows(state, layers, {"model": "W_T", "dataset": r["dataset"], "partition": r["partition"],
                                             "seed": r["seed"]})
            for j, row in enumerate(rows):
                row["logged_k_round_T"] = int(r["K"][-1, j])  # computed from W^(T-1): continuity reference only
                row["logged_branch_round_T"] = r["B"][-1, j]
            rows_t += rows
            print(f"  final model {r['tag']}: done")
        write_csv(endpoints / "rule_at_WT.csv", rows_t)
        energy_rows = [row for row in rows_t if row["branch"] == ENERGY_BRANCH]
        gap_rows = [row for row in rows_t if row["branch"] == GAP_BRANCH]
        wt_summary = {
            "branch_equals_logged_round_T_branch": all(row["branch"] == row["logged_branch_round_T"] for row in rows_t),
            "k_minus_logged_k_round_T_values": sorted({row["selected_k"] - row["logged_k_round_T"] for row in rows_t}),
            "max_gap_ratio_energy_layers": max((row["max_gap_ratio"] for row in energy_rows), default=None),
            "gap_layers": sorted({row["layer"] for row in gap_rows}),
            "gap_layer_ratio_range": [min(row["max_gap_ratio"] for row in gap_rows),
                                      max(row["max_gap_ratio"] for row in gap_rows)] if gap_rows else None,
            "gap_layer_retained_energy_range": [min(row["retained_energy_selected"] for row in gap_rows),
                                                max(row["retained_energy_selected"] for row in gap_rows)]
            if gap_rows else None,
            "gap_layer_k95_values": sorted({row["k95"] for row in gap_rows}),
        }

    # ---- headline numbers (Supplementary S10.1-S10.4)
    gap_layers = sorted({layers[j] for r in runs for j in range(len(layers)) if (r["B"][:, j] == GAP_BRANCH).any()},
                        key=layers.index)
    G_all = np.concatenate([r["G"].ravel() for r in runs])
    B_all = np.concatenate([r["B"].ravel() for r in runs])
    K_all = np.concatenate([r["K"].ravel() for r in runs])
    gap_mask, energy_mask = B_all == GAP_BRANCH, B_all == ENERGY_BRANCH
    w0_gap_rows = [w0_rule[n] for n in gap_layers]
    final_k = np.stack([r["K"][-1] for r in runs])
    summary = {
        "S10_1": {
            "runs": len(runs), "rounds_per_run": T_values, "layers": len(layers), "rank_decisions": int(n_dec),
            "all_checks_pass": all(v == "PASS" for r in runs for v in r["checks"].values()),
            "checks_per_run": len(runs[0]["checks"]),
            "W0_recomputation_matches_round1_rank_branch_gap_ratio_all_runs": w0_all_match,
            "W0_vs_round1_max_abs_gap_ratio_diff": max(row["gap_ratio_abs_diff"] for row in w0_cmp),
            "U_by_dataset": {d: sorted({r["U"] for r in runs if r["dataset"] == d}) for d in sorted({r["dataset"] for r in runs})},
        },
        "S10_2": {
            "gap_decisions": int(totals[GAP_BRANCH]), "gap_pct": 100 * totals[GAP_BRANCH] / n_dec,
            "energy_decisions": int(totals[ENERGY_BRANCH]), "fallback_decisions": int(totals[FALLBACK_BRANCH]),
            "layers_using_gap": gap_layers,
            "gap_k_values": sorted({int(v) for v in K_all[gap_mask]}),
            "gap_ratio_range_in_gap_decisions": [float(G_all[gap_mask].min()), float(G_all[gap_mask].max())]
            if gap_mask.any() else None,
            "gap_layers_at_W0": [{"layer": row["layer"], "k": row["selected_k"],
                                  "retained_energy": row["retained_energy_selected"], "k95": row["k95"]}
                                 for row in w0_gap_rows],
            "max_gap_ratio_in_energy_decisions": float(np.nanmax(G_all[energy_mask])) if energy_mask.any() else None,
            "energy_decisions_with_ratio_ge_tau (rho_max binding)": int((G_all[energy_mask] >= TAU).sum()),
            "branch_switches_total": int(sum(row["branch_switches"] for row in t6)),
        },
        "S10_3": {
            "table_S2": table_s2,
            "final_rank_max_range_across_runs": int((final_k.max(axis=0) - final_k.min(axis=0)).max()),
            "final_rank_range_per_layer": {n: [int(final_k[:, j].min()), int(final_k[:, j].max())]
                                           for j, n in enumerate(layers)},
            "rank_change_events_per_run": [min(row["events"] for row in t6), max(row["events"] for row in t6)],
            "all_events_abs_dk_1": all(abs(e["dk"]) == 1 for e in all_events),
            "increases_per_run": [min(row["increases"] for row in t6), max(row["increases"] for row in t6)],
            "top3_layers_of_D_reduction": top3,
            "top3_share_of_D_reduction_pct_mean": 100 * float(top3_per_run.mean()),
            "top3_share_of_D_reduction_pct_range": [100 * float(top3_per_run.min()), 100 * float(top3_per_run.max())],
            "energy_layers_share_of_projected_coordinates_pct_min_over_runs":
                min(row["energy_share_of_projected_mean_pct"] for row in t4),
        },
        "S10_4": {
            "update_over_noise_mean_range": [min(row["update_over_noise_mean"] for row in ns),
                                             max(row["update_over_noise_mean"] for row in ns)],
            "signal_bound_over_noise_mean_range": [min(row["signal_bound_over_noise_mean"] for row in ns),
                                                   max(row["signal_bound_over_noise_mean"] for row in ns)],
            "max_abs_norm2_growth_over_cumulative_noise_energy_minus_1":
                max(abs(row["growth_over_noise_energy_minus_1"]) for row in ns),
        },
        "final_models": wt_summary,
        "runs": [{"tag": r["tag"], "log": str(r["log"].relative_to(r["root"])), "sha256": r["log_sha256"],
                  "checks": r["checks"]} for r in runs],
    }
    (out / "checks.json").write_text(json.dumps({r["tag"]: r["checks"] for r in runs}, indent=1) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")

    s1, s2, s3, s4 = summary["S10_1"], summary["S10_2"], summary["S10_3"], summary["S10_4"]
    md.insert(0, "# Rank-rule audit (Supplementary S10.1-S10.4)\n\n"
              f"- {s1['runs']} runs, {s1['rank_decisions']} layer-wise rank decisions; all {s1['checks_per_run']} "
              f"checks pass in every run: {s1['all_checks_pass']}.\n"
              f"- Rank rule recomputed on W0 matches round 1 (rank, branch, gap ratio) in all runs: "
              f"{s1['W0_recomputation_matches_round1_rank_branch_gap_ratio_all_runs']} "
              f"(max |ratio difference| {s1['W0_vs_round1_max_abs_gap_ratio_diff']:.2e}).\n"
              f"- Gap branch: {s2['gap_decisions']} decisions ({s2['gap_pct']:.3f}%), layers {s2['layers_using_gap']}, "
              f"k in {s2['gap_k_values']}, ratio {span(s2['gap_ratio_range_in_gap_decisions'], 2) if s2['gap_ratio_range_in_gap_decisions'] else '-'}. "
              f"At W0: " + "; ".join(f"{g['layer']} retains {100 * g['retained_energy']:.1f}% of the weight energy, "
                                      f"k95 = {g['k95']}" for g in s2["gap_layers_at_W0"]) + ".\n"
              f"- Energy branch: {s2['energy_decisions']} decisions, max gap ratio {s2['max_gap_ratio_in_energy_decisions']:.3f}; "
              f"rho_max binding in {s2['energy_decisions_with_ratio_ge_tau (rho_max binding)']}; fallback "
              f"{s2['fallback_decisions']}; branch switches {s2['branch_switches_total']}.\n"
              f"- Final rank range across runs <= {s3['final_rank_max_range_across_runs']}; rank-change events per run "
              f"{s3['rank_change_events_per_run'][0]}-{s3['rank_change_events_per_run'][1]}, all |dk| = 1: "
              f"{s3['all_events_abs_dk_1']}, increases per run {s3['increases_per_run'][0]}-{s3['increases_per_run'][1]}.\n"
              f"- Top-3 layers {[short_name(n) for n in s3['top3_layers_of_D_reduction']]} explain "
              f"{s3['top3_share_of_D_reduction_pct_mean']:.1f}% of the decrease of D (mean over runs); energy-branch "
              f"layers hold >= {s3['energy_layers_share_of_projected_coordinates_pct_min_over_runs']:.4f}% of the "
              f"projected coordinates.\n"
              f"- Released update norm / expected noise norm (mean over rounds): {span(s4['update_over_noise_mean_range'], 4)}; "
              f"signal upper bound / noise scale: {span(s4['signal_bound_over_noise_mean_range'], 3)}; "
              f"|norm^2 growth / cumulative noise energy - 1| <= "
              f"{100 * s4['max_abs_norm2_growth_over_cumulative_noise_energy_minus_1']:.2f}%.\n")
    md.append("## Table S2: projected private dimension per dataset\n\n" + md_table(
        [{**row, "D_T_over_D_1": span([row["D_T_over_D_1_min"], row["D_T_over_D_1_max"]], 3)} for row in table_s2],
        ["dataset", "runs", "D_1", "D_T_min", "D_T_max", "D_T_over_D_1"]))
    md.append("## Branch usage per layer (all runs)\n\n" + md_table(per_layer_branch))
    md.append("## Rank statistics per layer\n\n" + md_table(
        [{**row, "k_T_over_k_1_mean": fmt(row["k_T_over_k_1_mean"], 3)} for row in t2],
        ["short", "min(m,n)", "k_1", "k_T_min", "k_T_max", "k_T_over_k_1_mean"]))
    md.append("## Share of the decrease of D per layer (%, over runs)\n\n" + md_table(
        [{"layer": row["layer"], "mean": fmt(row["mean_share_pct"]), "min": fmt(row["min_share_pct"]),
          "max": fmt(row["max_share_pct"])} for row in t5_layer]))
    md.append("## Rule outcome at W0\n\n" + md_table(
        [{"layer": short_name(row["layer"]), "k": row["selected_k"], "branch": row["branch"], "k*": row["k_star"],
          "ratio": fmt(row["max_gap_ratio"], 3), "k*/sqrt(mn)": fmt(row["k_star_over_sqrt_mn"], 4), "k90": row["k90"],
          "k95": row["k95"], "k99": row["k99"], "retained energy": fmt(row["retained_energy_selected"], 3)}
         for row in rows0]))
    md.append("## Noise vs signal scale of the released update\n\n" + md_table(
        [{"run": f"{row['dataset']}/{row['partition']}/seed{row['seed']}",
          "update/noise mean": fmt(row["update_over_noise_mean"], 4),
          "signal bound/noise mean": fmt(row["signal_bound_over_noise_mean"], 4),
          "norm^2 growth": fmt(row["trainable_norm2_growth"]),
          "cumulative noise energy": fmt(row["cumulative_expected_noise_energy"])} for row in ns]))
    if wt_summary is not None:
        md.append("## Rule outcome at the final models\n\n```\n" + json.dumps(wt_summary, indent=1, default=float) + "\n```")
    (out / "summary.md").write_text("\n\n".join(md) + "\n")
    print((out / "summary.md").read_text().split("\n## ")[0])
    return 0 if summary["S10_1"]["all_checks_pass"] and w0_all_match else 1


if __name__ == "__main__":
    sys.exit(main())
