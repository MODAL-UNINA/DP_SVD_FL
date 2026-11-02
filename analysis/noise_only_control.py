#!/usr/bin/env python
"""Data-free noise-only control of the adaptive rank rule (Supplementary Sec. S10.4). CPU only, no training.

Starting from the public ImageNet ResNet-18 weights W0, every round t = 1..T applies the UNCHANGED rank rule
(``dpsvdfl.spectral.rank_rule.select_rank``) to the current spectrum of each projected layer -- rank and branch are
recorded as "round t", exactly like ``released_round_diagnostics.jsonl``, which records the rule applied to
W^(t-1) -- and then adds a pure-DP-noise server update

    Delta W_l = eta_s * U_k Z V_k^T,   Z in R^(k x k),   Z_ij ~ N(0, (sigma C / (q N))^2) i.i.d.,

with (U_k, V_k) the top-k singular vectors of the current W_l. No data, no clients, no gradients: the clipped client
signal is set to zero. Default parameters are those of the main experiments (sigma = 0.95, C = 0.2, q = 0.1,
N = 500, eta_s = 1, hence a normalised coordinate standard deviation sigma C / (qN) = 0.0038).

Exact spectral reduction. If W = U diag(s) V^T (reduced SVD, r = min(m, n)), then

    W + U_k Z V_k^T = U (diag(s) + [[Z, 0], [0, 0]]) V^T,

and U, V have orthonormal columns, so the singular values of the updated matrix are the union of the singular values
of diag(s[:k]) + Z and of the tail s[k:]. Because the next update is again drawn as i.i.d. Gaussian coefficients in
the CURRENT top-k singular basis, the singular-value vector evolves as a Markov chain that needs only k x k SVDs.
For identical coefficient matrices Z the chain reproduces the full-matrix dynamics exactly (up to floating point);
it is exact in distribution in any case. ``--self-test`` checks this against full-matrix SVDs on synthetic matrices
and ``--validate-w0`` on the real W0 layers.

Random numbers. One ``torch.Generator`` (CPU) per (simulation seed, round, layer), seeded with the first 8 bytes of
SHA-256("noise-only-rank-replay|<seed>|<round>|<layer index>") (63 bits). Z = std * randn(k, k, float64).

Modes:
    --self-test                       synthetic validation of the reduction against full-matrix SVDs
    --validate-w0 --runs ROOT ...     rule on W0 vs round 1 of the released logs; short full-matrix replay on W0
    --run --seeds 0-29 --rounds 300 --out DIR
                                      the simulation; writes trajectories_long.csv, per_round_summary.csv,
                                      spectra_snapshots.csv and run_manifest.json
    --check-against REF_DIR --out DIR compare the shared seeds of two simulation outputs, string by string and
                                      numerically

Example:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src \\
        python analysis/noise_only_control.py --run --seeds 0-29 --rounds 300 --out results/noise_only_30_seeds
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only, before torch is imported

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # without `pip install -e .`
from dpsvdfl.dp.mechanism import (CLIPPING_NORM, NOISE_MULTIPLIER, NUM_CLIENT_SLOTS, SAMPLING_RATE,  # noqa: E402
                                  DPMechanismConfig)
from dpsvdfl.federated.server_update import SERVER_LEARNING_RATE  # noqa: E402
from dpsvdfl.models.partitions import RESNET18_PROJECTED_LAYERS, RESNET18_UNPROJECTED_PARAMETERS  # noqa: E402
from dpsvdfl.spectral.matricization import matricize  # noqa: E402
from dpsvdfl.spectral.projection import compute_projection_basis  # noqa: E402
from dpsvdfl.spectral.rank_rule import GAMMA, RHO_MAX, TAU, select_rank  # noqa: E402

PROJECTED = tuple(f"{name}.weight" for name in RESNET18_PROJECTED_LAYERS)
# Datasets for which D(t) = U + sum_l k_l(t)^2 is written; U = (512 + 1) x classes (fc weight and bias).
CLASSES = {"cifar10": 10, "cifar100": 100, "emnist_balanced": 47}
SNAPSHOT_ROUNDS = (1, 50, 100, 150, 200, 250, 300)
RNG_LABEL = "noise-only-rank-replay"  # part of the seed derivation: changing it changes every random draw
EXTRA_COLS = ["s2_before", "s3_before", "s1_over_s2", "s2_over_s3", "top_ratio_rank_le_3"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_seeds(text: list[str]) -> list[int]:
    """Accept "0-29", "0 1 2" or "0,1,2" (and mixtures)."""
    seeds = []
    for item in " ".join(text).replace(",", " ").split():
        if "-" in item:
            lo, hi = (int(v) for v in item.split("-"))
            seeds.extend(range(lo, hi + 1))
        else:
            seeds.append(int(item))
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds in {text}")
    return seeds


def noise_std(sigma: float, clip: float, q: float, n_slots: int) -> tuple[float, dict]:
    """Per-coordinate std of the released update, from the package's mechanism configuration."""
    config = DPMechanismConfig(clipping_norm=clip, noise_multiplier=sigma, sampling_rate=q, num_client_slots=n_slots)
    std = config.normalized_noise_std
    formula = sigma * clip / (q * n_slots)
    if not math.isclose(std, formula, rel_tol=0, abs_tol=1e-15):
        raise RuntimeError(f"normalized_noise_std {std} != sigma C / (q N) = {formula}")
    eta = SERVER_LEARNING_RATE
    return eta * std, {"sigma": sigma, "C": clip, "q": q, "N": n_slots, "server_learning_rate": eta,
                       "normalized_noise_std": std, "sigma_C_over_qN": formula, "update_coefficient_std": eta * std}


def seed_for(sim_seed: int, round_index: int, layer_index: int) -> int:
    """Independent, reproducible stream per (simulation seed, round, layer)."""
    text = f"{RNG_LABEL}|{sim_seed}|{round_index}|{layer_index}".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big") & ((1 << 63) - 1)


def draw_z(k: int, std: float, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return std * torch.randn((k, k), generator=generator, dtype=torch.float64, device="cpu")


def chain_step(s: torch.Tensor, k: int, z: torch.Tensor) -> torch.Tensor:
    """Singular values of diag(s) + [[Z, 0], [0, 0]]: svd(diag(s[:k]) + Z) united with s[k:], sorted."""
    head = torch.linalg.svdvals(torch.diag(s[:k]) + z)
    return torch.sort(torch.cat([head, s[k:]]), descending=True).values


# ---------------------------------------------------------------- self-test (synthetic, full SVD reference)


def _matrix_with_spectrum(m: int, n: int, s: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    r = min(m, n)
    a, _ = torch.linalg.qr(torch.randn(m, r, generator=gen, dtype=torch.float64))
    b, _ = torch.linalg.qr(torch.randn(n, r, generator=gen, dtype=torch.float64))
    return a @ torch.diag(s) @ b.T


def self_test() -> dict:
    gen = torch.Generator().manual_seed(12345)
    results = {"single_step": [], "multi_round": [], "passed": True}

    # A. One update: full SVD of W + U_k Z V_k^T versus the reduced identity.
    for m, n, k in ((40, 90, 7), (90, 40, 40), (128, 1152, 1), (64, 64, 30), (256, 128, 90)):
        r = min(m, n)
        s = torch.sort(torch.rand(r, generator=gen, dtype=torch.float64) * 3.0, descending=True).values
        w = _matrix_with_spectrum(m, n, s, gen)
        u, s_full, vh = torch.linalg.svd(w, full_matrices=False)
        z = 0.3 * torch.randn(k, k, generator=gen, dtype=torch.float64)
        w_new = w + u[:, :k] @ z @ vh[:k, :]
        reference = torch.linalg.svdvals(w_new)
        reduced = chain_step(s_full, k, z)
        err = float((reference - reduced).abs().max() / reference[0])
        ok = err < 1e-12
        results["single_step"].append({"m": m, "n": n, "k": k, "max_rel_error": err, "pass": ok})
        results["passed"] &= ok

    # B. Many rounds with the rank rule: full-matrix dynamics versus the chain, same Z. Planted gaps (ratio at
    #    j < rho_max sqrt(mn)) start in the gap branch; the ratio 2.05 at j = 1 sits close enough to tau for the noise
    #    to push it across, so branch switches are exercised too.
    cases = {"planted_gap_96x864": (96, 864, 0.02, 40, 3, 2.2),
             "gap_switch_64x576": (64, 576, 0.05, 60, 1, 2.05),
             "energy_only_128x256": (128, 256, 0.05, 40, None, None)}
    for label, (m, n, std, rounds, gap_j, gap_ratio) in cases.items():
        r = min(m, n)
        base = torch.exp(-torch.arange(r, dtype=torch.float64) / 25.0)
        if gap_j is not None:
            base[:gap_j] *= gap_ratio * base[gap_j] / base[gap_j - 1]
            base = torch.sort(base, descending=True).values
        w = _matrix_with_spectrum(m, n, base, gen)
        s_chain = torch.linalg.svdvals(w)
        ranks_full, ranks_chain, branches_full, branches_chain, max_err = [], [], [], [], 0.0
        for t in range(1, rounds + 1):
            u, s_full, vh = torch.linalg.svd(w, full_matrices=False)
            max_err = max(max_err, float((s_full - s_chain).abs().max() / s_full[0]))
            rule_full = select_rank(s_full, m, n, TAU, RHO_MAX, GAMMA)
            rule_chain = select_rank(s_chain, m, n, TAU, RHO_MAX, GAMMA)
            ranks_full.append(rule_full.selected_rank)
            ranks_chain.append(rule_chain.selected_rank)
            branches_full.append(rule_full.branch)
            branches_chain.append(rule_chain.branch)
            k = rule_full.selected_rank
            z = draw_z(k, std, seed_for(999, t, 0))
            w = w + u[:, :k] @ z @ vh[:k, :]
            s_chain = chain_step(s_chain, rule_chain.selected_rank, z)
        ok = ranks_full == ranks_chain and branches_full == branches_chain and max_err < 1e-10
        results["multi_round"].append({
            "case": label, "m": m, "n": n, "noise_std": std, "rounds": rounds,
            "ranks_identical": ranks_full == ranks_chain, "branches_identical": branches_full == branches_chain,
            "max_rel_singular_value_error": max_err, "rank_first_last": [ranks_full[0], ranks_full[-1]],
            "branches_seen": sorted(set(branches_full)),
            "branch_switches": sum(a != b for a, b in zip(branches_full, branches_full[1:])), "pass": ok})
        results["passed"] &= ok
    return results


# ---------------------------------------------------------------- W0


def load_w0() -> tuple[dict, dict]:
    """Projected layers of the public W0 (torchvision IMAGENET1K_V1, honouring TORCH_HOME), matricised, float32."""
    from torchvision.models import ResNet18_Weights, resnet18

    state = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
    layers = {name: matricize(state[name].detach()) for name in PROJECTED}  # float32, as the model stores it
    return layers, {"weights": "torchvision ResNet18_Weights.IMAGENET1K_V1",
                    "fc_in_features": int(state["fc.weight"].shape[1])}


def initial_spectra(layers: dict) -> dict:
    """Same call as the pipeline: compute_projection_basis (float64 SVD + select_rank) on W_l^(0)."""
    out = {}
    for name, matrix in layers.items():
        basis = compute_projection_basis(matrix, tau=TAU, rho_max=RHO_MAX, gamma=GAMMA)
        out[name] = {"shape": tuple(matrix.shape), "s": basis.singular_values.clone(), "rule": basis.rank_rule}
    return out


def validate_w0(run_roots: list[Path], replay_rounds: int, std: float) -> dict:
    """Rule on W0 vs round 1 of every released log; then a few rounds of full-matrix replay vs the chain on W0."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rank_audit import discover  # noqa: E402  (sibling script: run-directory discovery)

    layers, w0_info = load_w0()
    spectra = initial_spectra(layers)
    computed = {name: spectra[name]["rule"] for name in PROJECTED}
    comparisons, d1_checks, all_ok, max_gap_diff = [], [], True, 0.0
    runs = discover(run_roots) if run_roots else []
    for run in runs:
        with open(run["dir"] / "released_round_diagnostics.jsonl") as handle:
            first = json.loads(handle.readline())
        if first["round"] != 1 or set(first["ranks"]) != set(PROJECTED):
            all_ok = False
            comparisons.append({"run": f"{run['dataset']}/{run['partition']}/seed{run['seed']}",
                                "error": "first record is not round 1 or the projected layer set differs"})
            continue
        sum_k2 = 0
        for name in PROJECTED:
            rule = computed[name]
            gap_diff = abs(rule.max_gap_ratio - float(first["gap_ratios"][name]))
            max_gap_diff = max(max_gap_diff, gap_diff)
            ok = (rule.selected_rank == int(first["ranks"][name]) and rule.branch == first["rank_branches"][name]
                  and gap_diff <= 1e-9)
            all_ok &= ok
            sum_k2 += rule.selected_rank ** 2
            comparisons.append({"run": f"{run['dataset']}/{run['partition']}/seed{run['seed']}", "layer": name,
                                "m": spectra[name]["shape"][0], "n": spectra[name]["shape"][1],
                                "k_w0": rule.selected_rank, "k_log": int(first["ranks"][name]),
                                "branch_w0": rule.branch, "branch_log": first["rank_branches"][name],
                                "gap_ratio_w0": repr(rule.max_gap_ratio), "gap_ratio_log": repr(first["gap_ratios"][name]),
                                "gap_ratio_abs_diff": gap_diff, "match": ok})
        if run["dataset"] in CLASSES:
            u = (w0_info["fc_in_features"] + 1) * CLASSES[run["dataset"]]
            d1_checks.append({"run": f"{run['dataset']}/{run['partition']}/seed{run['seed']}", "U": u,
                              "sum_k2": sum_k2, "D1_w0": u + sum_k2, "D1_log": first["D"],
                              "match": u + sum_k2 == first["D"]})
            all_ok &= u + sum_k2 == first["D"]

    # Full-matrix replay on the REAL layers: float32 storage + float64 SVD (pipeline numerics) versus the float64
    # chain, identical Z, a few rounds.
    replay = []
    for index, name in enumerate(PROJECTED):
        w = layers[name].clone()
        m, n = w.shape
        s_chain = spectra[name]["s"].clone()
        rows = []
        for t in range(1, replay_rounds + 1):
            u, s_full, vh = torch.linalg.svd(w.to(torch.float64), full_matrices=False)
            rule_full = select_rank(s_full, m, n, TAU, RHO_MAX, GAMMA)
            rule_chain = select_rank(s_chain, m, n, TAU, RHO_MAX, GAMMA)
            rel = float((s_full - s_chain).abs().max() / s_full[0])
            rows.append({"round": t, "k_full": rule_full.selected_rank, "k_chain": rule_chain.selected_rank,
                         "branch_full": rule_full.branch, "branch_chain": rule_chain.branch, "max_rel_sv_diff": rel})
            k = rule_full.selected_rank
            z = draw_z(k, std, seed_for(0, t, index))
            w = (w.to(torch.float64) + u[:, :k] @ z @ vh[:k, :]).to(torch.float32)
            s_chain = chain_step(s_chain, rule_chain.selected_rank, z)
        same = all(r["k_full"] == r["k_chain"] and r["branch_full"] == r["branch_chain"] for r in rows)
        replay.append({"layer": name, "rounds": rows, "ranks_and_branches_identical": same,
                       "max_rel_sv_diff": max(r["max_rel_sv_diff"] for r in rows)})
    return {"w0": w0_info, "runs_checked": len(runs), "all_match": all_ok, "max_gap_ratio_abs_diff": max_gap_diff,
            "comparisons": comparisons, "D1_checks": d1_checks, "full_matrix_replay": replay,
            "full_matrix_replay_ok": all(r["ranks_and_branches_identical"] for r in replay),
            "computed": {name: {"k": r.selected_rank, "branch": r.branch, "gap_ratio": r.max_gap_ratio,
                                "gap_rank": r.gap_rank, "energy_rank": r.energy_rank} for name, r in computed.items()}}


# ---------------------------------------------------------------- simulation


def simulate(rounds: int, seeds: list[int], std: float, out: Path, log) -> dict:
    layers, w0_info = load_w0()
    spectra0 = initial_spectra(layers)
    names = PROJECTED
    u_by_dataset = {d: (w0_info["fc_in_features"] + 1) * c for d, c in CLASSES.items()}
    long_path, summary_path, snap_path = (out / "trajectories_long.csv", out / "per_round_summary.csv",
                                          out / "spectra_snapshots.csv")
    long_cols = ["sim_seed", "round", "layer", "m", "n", "k", "branch", "gap_ratio", "gap_rank",
                 "gap_rank_fraction", "energy_rank", "selected_energy_fraction", "k_squared", "k_over_sqrt_mn",
                 "fro2_before", "s1_before", "noise_fro2", "projected_dim"] + [f"D_{d}" for d in CLASSES] + EXTRA_COLS
    summary_cols = ["sim_seed", "round", "projected_dim"] + [f"D_{d}" for d in CLASSES] + [
        "n_gap", "n_energy", "n_fallback", "round_seconds"]
    timings, finite_ok = {}, True
    checks = {"k_in_range": True, "gap_branch_conditions": True, "fallback_k1": True, "sorted_nonneg": True}
    with open(long_path, "w", newline="") as lf, open(summary_path, "w", newline="") as sf, \
            open(snap_path, "w", newline="") as pf:
        lw, sw, pw = csv.writer(lf), csv.writer(sf), csv.writer(pf)
        lw.writerow(long_cols)
        sw.writerow(summary_cols)
        pw.writerow(["sim_seed", "round", "layer", "index", "singular_value"])
        for sim_seed in seeds:
            state = {name: spectra0[name]["s"].clone() for name in names}
            start = time.perf_counter()
            for t in range(1, rounds + 1):
                t0 = time.perf_counter()
                # the rule is applied to every layer first (on W^(t-1)), then the noise update layer by layer
                rules, rows = {}, []
                for name in names:
                    m, n = spectra0[name]["shape"]
                    rules[name] = select_rank(state[name], m, n, TAU, RHO_MAX, GAMMA)
                projected_dim = sum(r.selected_rank ** 2 for r in rules.values())
                for index, name in enumerate(names):
                    m, n = spectra0[name]["shape"]
                    rule, s = rules[name], state[name]
                    k = rule.selected_rank
                    checks["k_in_range"] &= 1 <= k <= min(m, n)
                    if rule.branch == "gap":
                        checks["gap_branch_conditions"] &= (rule.max_gap_ratio >= TAU
                                                           and rule.gap_rank_fraction < RHO_MAX)
                    if rule.branch == "fallback":
                        checks["fallback_k1"] &= k == 1
                    if t in SNAPSHOT_ROUNDS:
                        for i, value in enumerate(s.tolist()):
                            pw.writerow([sim_seed, t, name, i + 1, repr(value)])
                    s1, s2, s3 = (float(v) for v in s[:3])
                    extra = [repr(s2), repr(s3), repr(s1 / s2), repr(s2 / s3), int(rule.gap_rank <= 3)]
                    z = draw_z(k, std, seed_for(sim_seed, t, index))
                    new = chain_step(s, k, z)
                    finite_ok &= bool(torch.isfinite(new).all())
                    checks["sorted_nonneg"] &= bool((new >= 0).all()) and bool((new[:-1] >= new[1:]).all())
                    rows.append([sim_seed, t, name, m, n, k, rule.branch, repr(rule.max_gap_ratio), rule.gap_rank,
                                 repr(rule.gap_rank_fraction), rule.energy_rank,
                                 repr(rule.selected_energy_fraction), k * k, repr(k / math.sqrt(m * n)),
                                 repr(float((s * s).sum())), repr(float(s[0])), repr(float((z * z).sum())),
                                 projected_dim] + [u + projected_dim for u in u_by_dataset.values()] + extra)
                    state[name] = new
                lw.writerows(rows)
                branches = [r.branch for r in rules.values()]
                sw.writerow([sim_seed, t, projected_dim] + [u + projected_dim for u in u_by_dataset.values()]
                            + [branches.count("gap"), branches.count("energy"), branches.count("fallback"),
                               f"{time.perf_counter() - t0:.6f}"])
            for name in names:  # spectrum after the last update (not used by any recorded round)
                for i, value in enumerate(state[name].tolist()):
                    pw.writerow([sim_seed, f"after_{rounds}", name, i + 1, repr(value)])
            timings[str(sim_seed)] = time.perf_counter() - start
            log(f"seed {sim_seed} done in {timings[str(sim_seed)]:.2f}s, finite={finite_ok}")
    return {"w0": w0_info, "u_by_dataset": u_by_dataset, "timings_seconds": timings, "all_finite": finite_ok,
            "checks": checks, "files": [p.name for p in (long_path, summary_path, snap_path)]}


# ---------------------------------------------------------------- comparison of two simulation outputs


def _float_or_none(text: str):
    try:
        return float(text)
    except ValueError:
        return None


def check_against(reference: Path, new: Path) -> dict:
    """Compare every column shared by two simulation outputs, for the seeds present in `new`.

    Reports exact string equality per file and, where strings differ, the largest absolute and relative numeric
    difference per column (integer/categorical columns such as k and branch are counted separately).
    """
    result = {"reference": reference.name, "new": new.name}
    for filename, key in (("trajectories_long.csv", ("sim_seed", "round", "layer")),
                          ("per_round_summary.csv", ("sim_seed", "round")),
                          ("spectra_snapshots.csv", ("sim_seed", "round", "layer", "index"))):
        with open(new / filename, newline="") as handle:
            new_rows = list(csv.DictReader(handle))
        seeds = {row["sim_seed"] for row in new_rows}
        with open(reference / filename, newline="") as handle:
            ref_rows = [row for row in csv.DictReader(handle) if row["sim_seed"] in seeds]
        skip = {"round_seconds"}  # wall-clock timing, not a result
        shared = [c for c in new_rows[0] if c in ref_rows[0] and c not in skip]
        ref_index = {tuple(r[c] for c in key): r for r in ref_rows}
        mismatching_rows, missing = 0, 0
        per_column = {c: {"rows_differing": 0, "max_abs_diff": 0.0, "max_rel_diff": 0.0} for c in shared}
        for row in new_rows:
            ref = ref_index.get(tuple(row[c] for c in key))
            if ref is None:
                missing += 1
                continue
            differs = False
            for c in shared:
                if ref[c] != row[c]:
                    differs = True
                    stats = per_column[c]
                    stats["rows_differing"] += 1
                    a, b = _float_or_none(ref[c]), _float_or_none(row[c])
                    if a is not None and b is not None:
                        stats["max_abs_diff"] = max(stats["max_abs_diff"], abs(a - b))
                        if a != 0:
                            stats["max_rel_diff"] = max(stats["max_rel_diff"], abs(a - b) / abs(a))
                    else:
                        stats["max_abs_diff"] = math.inf
            mismatching_rows += differs
        result[filename] = {"seeds": sorted(seeds, key=int), "rows_reference": len(ref_rows),
                            "rows_new": len(new_rows), "rows_missing_in_reference": missing,
                            "shared_columns": shared, "skipped_columns": sorted(skip & set(new_rows[0])),
                            "mismatching_rows": mismatching_rows,
                            "differing_columns": {c: v for c, v in per_column.items() if v["rows_differing"]},
                            "identical": mismatching_rows == 0 and missing == 0 and len(ref_rows) == len(new_rows)}
    result["all_identical"] = all(v["identical"] for v in result.values() if isinstance(v, dict))
    return result


def environment_record(threads: int) -> dict:
    return {"python": sys.version.split()[0], "torch": torch.__version__, "machine": platform.machine(),
            "torch_threads": torch.get_num_threads(), "requested_threads": threads,
            "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                   "MKL_NUM_THREADS")},
            "torch_cuda_initialized": torch.cuda.is_initialized()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true", help="synthetic validation against full-matrix SVDs")
    mode.add_argument("--validate-w0", action="store_true",
                      help="rule on W0 vs round 1 of the released logs (--runs) and a short full-matrix replay")
    mode.add_argument("--run", action="store_true", help="run the noise-only simulation")
    mode.add_argument("--check-against", type=Path, metavar="REF_DIR",
                      help="compare the simulation output in --out with a reference simulation output")
    parser.add_argument("--runs", type=Path, nargs="*", default=[],
                        help="(--validate-w0) roots searched recursively for adaptive run directories")
    parser.add_argument("--out", type=Path, help="output directory")
    parser.add_argument("--rounds", type=int, default=300, help="number of rounds T (default 300)")
    parser.add_argument("--seeds", nargs="+", default=["0"], help='simulation seeds, e.g. "0-29" or "0 1 2"')
    parser.add_argument("--threads", type=int, default=4, help="torch CPU threads (default 4)")
    parser.add_argument("--replay-rounds", type=int, default=3, help="(--validate-w0) full-matrix replay rounds")
    parser.add_argument("--sigma", type=float, default=NOISE_MULTIPLIER, help="noise multiplier (default 0.95)")
    parser.add_argument("--clip", type=float, default=CLIPPING_NORM, help="clipping norm C (default 0.2)")
    parser.add_argument("--q", type=float, default=SAMPLING_RATE, help="sampling rate q (default 0.1)")
    parser.add_argument("--num-slots", type=int, default=NUM_CLIENT_SLOTS, help="client slots N (default 500)")
    args = parser.parse_args()

    if args.check_against:
        if args.out is None:
            parser.error("--check-against needs --out (the new simulation output)")
        result = check_against(args.check_against, args.out)
        with open(args.out / "check_against_reference.json", "w") as handle:
            json.dump(result, handle, indent=1)
        print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()
                                                                  if kk != "shared_columns"})
                          for k, v in result.items()}, indent=1))
        print("STATUS", "IDENTICAL" if result["all_identical"] else "DIFFERENT")
        return 0

    torch.set_num_threads(args.threads)
    std, noise_info = noise_std(args.sigma, args.clip, args.q, args.num_slots)
    started = time.time()
    record = {"command": [Path(sys.argv[0]).name] + sys.argv[1:], "script_sha256": sha256_file(Path(__file__)),
              "noise": noise_info, "rank_rule": {"tau": TAU, "rho_max": RHO_MAX, "gamma": GAMMA},
              "projected_layers": list(PROJECTED), "unprojected": list(RESNET18_UNPROJECTED_PARAMETERS)}

    def log(message: str) -> None:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)

    if args.self_test:
        result = self_test()
        record["self_test"] = result
        status = result["passed"]
        print(json.dumps(result, indent=1))
    elif args.validate_w0:
        result = validate_w0(args.runs, args.replay_rounds, std)
        record["validate_w0"] = result
        status = result["all_match"] and result["full_matrix_replay_ok"]
        if args.out and result["comparisons"]:
            args.out.mkdir(parents=True, exist_ok=True)
            with open(args.out / "w0_vs_round1.csv", "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(result["comparisons"][0]))
                writer.writeheader()
                writer.writerows(result["comparisons"])
        print(json.dumps({k: v for k, v in result.items() if k != "comparisons"}, indent=1))
    else:
        if args.out is None:
            parser.error("--run needs --out")
        seeds = parse_seeds(args.seeds)
        args.out.mkdir(parents=True, exist_ok=False)
        record["simulation"] = {"rounds": args.rounds, "seeds": seeds,
                                "rng": f"torch.Generator(cpu) per (sim_seed, round, layer), seed = sha256 prefix of "
                                       f"'{RNG_LABEL}|sim_seed|round|layer_index'",
                                "round_semantics": "round t records the rule on W^(t-1); the noise update follows",
                                "extra_columns": EXTRA_COLS}
        log(f"start: {len(seeds)} seeds x {args.rounds} rounds, threads={torch.get_num_threads()}")
        result = simulate(args.rounds, seeds, std, args.out, log)
        record["simulation"].update(result)
        status = result["all_finite"] and all(result["checks"].values())
    record["environment"] = environment_record(args.threads)
    record["wall_seconds"] = time.time() - started
    record["status"] = "PASS" if status else "FAIL"
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        name = "self_test.json" if args.self_test else "w0_validation.json" if args.validate_w0 else "run_manifest.json"
        with open(args.out / name, "w") as handle:
            json.dump(record, handle, indent=1, default=str)
    print("STATUS", record["status"], f"wall {record['wall_seconds']:.1f}s")
    return 0 if status else 1


if __name__ == "__main__":
    sys.exit(main())
