#!/usr/bin/env python
"""Recorded adaptive runs vs the data-free noise-only control (Supplementary Sec. S10.4-S10.5, Table S3, Fig. G6).

Inputs:
  --runs ROOT ...  adaptive DP-SVD-FL run directories (recursive discovery, as in rank_audit.py); only the released
                   per-round log is read: k_l(t), branch, r_l(t) = max_j s_j / s_(j+1) (the "gap ratio") and D(t)
  --sim DIR        output of noise_only_control.py --run (trajectories_long.csv)

Quantities (the real trajectories enter only here, after the simulation):
  * Table S3: per dataset, decrease of D from round 1 to round T, real (mean over runs) vs noise-only (mean over
    simulation seeds); mean absolute error between the mean D curves (also in % of D(1)); D(T) ranges.
  * Per layer: mean final rank k_l(T), real vs noise-only; gap-branch decisions, real vs noise-only.
  * Trajectory-level tests. Every trajectory (one simulation seed, or one real run) is reduced to scalars per layer:
        S_late  mean r over the late window (default rounds 251..300; PRIMARY statistic, fixed in advance)
        k_T     final rank;  mean_k  rank averaged over all rounds
    Real runs with the same experiment seed share the DP-noise stream (same noise master seed in every
    dataset x partition configuration), so the independent unit is the experiment SEED: the configurations of a
    seed are averaged into one seed value, and the seed values are compared with the simulation seeds by an EXACT
    two-sided permutation test of the difference in means over all C(n_real + n_sim, n_real) splits
    (p = #{|d*| >= |d_obs|} / #splits, observed split included); the 95% CI of the difference is obtained by
    inverting the same test on a grid of 2401 shifts in +-6 noise-only standard deviations.
  * Holm correction on the seed-level p: primary family = {primary layers} x S_late (default layer3.1.conv2 and
    layer4.0.conv1); exploratory family = the other layers, separately for each statistic.
  * Sensitivity: the primary test without the first three simulation seeds.
  * Figure G6: projected dimension P(t) = sum_l k_l(t)^2 of the recorded runs vs the noise-only control (PDF + PNG).

Example:
    PYTHONPATH=src python analysis/noise_only_statistics.py --runs outputs/runs \\
        --sim results/noise_only_30_seeds --out results/noise_only_statistics
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rank_audit import FC_IN_PLUS_BIAS, PROJECTED, discover, short_name  # noqa: E402

PRIMARY_DEFAULT = ["layer3.1.conv2.weight", "layer4.0.conv1.weight"]
DATASETS = ["cifar10", "cifar100", "emnist_balanced"]
PARTITIONS = ["iid", "dirichlet0.6"]  # ordering of the real runs (also the drawing order of Fig. G6)
STATS = ["S_late", "k_T", "mean_k"]
CI_GRID = np.linspace(-6, 6, 2401)

# Figure style of the paper (serif, column width)
RC = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.direction": "in", "ytick.direction": "in", "grid.linewidth": 0.4, "grid.color": "#d9d9d9",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "legend.frameon": False,
}
OBS, OBS_DARK, NOISE, INK = "#0072B2", "#003C5E", "#E69F00", "#333333"  # Okabe-Ito tones


# ---------------------------------------------------------------- data


def run_order(run: dict) -> tuple:
    d, p = run["dataset"], run["partition"]
    return (DATASETS.index(d) if d in DATASETS else len(DATASETS), d,
            PARTITIONS.index(p) if p in PARTITIONS else len(PARTITIONS), p, run["seed"])


def load(sim_dir: Path, run_roots: list[Path]) -> dict:
    sim = pd.read_csv(sim_dir / "trajectories_long.csv")
    seeds = sorted(sim.sim_seed.unique())
    T = int(sim["round"].max())
    geometry = sim.drop_duplicates("layer").set_index("layer")[["m", "n"]]
    if set(geometry.index) != set(PROJECTED):
        raise ValueError("the simulation does not hold exactly the projected layers")
    sum_mn = int((geometry.m * geometry.n).sum())
    out = {"seeds": seeds, "T": T, "sim": {}, "real": {}}
    sim = sim.sort_values(["sim_seed", "layer", "round"])
    for layer in PROJECTED:
        sub = sim[sim.layer == layer]
        rounds = sub["round"].to_numpy().reshape(len(seeds), T)
        assert (rounds == np.arange(1, T + 1)).all(), "simulation rounds must be 1..T for every seed"
        out["sim"][layer] = {"gap_ratio": sub.gap_ratio.to_numpy(float).reshape(len(seeds), T),
                             "k": sub.k.to_numpy(float).reshape(len(seeds), T),
                             "gap": (sub.branch.to_numpy() == "gap").reshape(len(seeds), T)}
    out["sim_pdim"] = sim.drop_duplicates(["sim_seed", "round"]).sort_values(["sim_seed", "round"]) \
        .projected_dim.to_numpy(float).reshape(len(seeds), T)

    runs = sorted(discover(run_roots), key=run_order)
    if not runs:
        raise ValueError("no adaptive run directory found")
    blocks = {layer: {"gap_ratio": [], "k": [], "gap": []} for layer in PROJECTED}
    pdim, U = [], {}
    for run in runs:
        records = [json.loads(line) for line in (run["dir"] / "released_round_diagnostics.jsonl").read_text()
                   .splitlines() if line.strip()]
        if [r["round"] for r in records] != list(range(1, T + 1)):
            raise ValueError(f"{run['dir'].name}: released rounds are not 1..{T}")
        u = int(records[0]["D_full"]) - sum_mn
        if u % FC_IN_PLUS_BIAS:
            raise ValueError(f"{run['dir'].name}: U = {u} is not a multiple of 513")
        U.setdefault(run["dataset"], u)
        if U[run["dataset"]] != u:
            raise ValueError(f"inconsistent U for dataset {run['dataset']}")
        for layer in PROJECTED:
            g = [r["gap_ratios"][layer] for r in records]
            blocks[layer]["gap_ratio"].append([math.nan if v is None else v for v in g])
            blocks[layer]["k"].append([r["ranks"][layer] for r in records])
            blocks[layer]["gap"].append([r["rank_branches"][layer] == "gap" for r in records])
        pdim.append(np.array([r["D"] for r in records], dtype=float) - u)
    for layer in PROJECTED:
        out["real"][layer] = {key: np.array(value, dtype=float if key != "gap" else bool)
                              for key, value in blocks[layer].items()}
    out["real_pdim"] = np.stack(pdim)
    ksq = sum(out["real"][layer]["k"] ** 2 for layer in PROJECTED)
    assert np.array_equal(ksq, out["real_pdim"]), "D(t) != U + sum_l k_l(t)^2 in a real run"
    out["runs"] = [(r["dataset"], r["partition"], r["seed"]) for r in runs]
    out["U"] = U
    return out


# ---------------------------------------------------------------- statistics


def trajectory_summaries(r: np.ndarray, k: np.ndarray, gap: np.ndarray, late: slice) -> dict:
    return {"S_late": r[:, late].mean(1), "k_1": k[:, 0], "k_T": k[:, -1], "mean_k": k.mean(1),
            "n_gap": gap.sum(1)}


def split_diffs(values: np.ndarray, splits: np.ndarray) -> np.ndarray:
    n, n_real = len(values), splits.shape[1]
    chosen = values[splits].sum(1)
    return chosen / n_real - (values.sum() - chosen) / (n - n_real)


def seed_test(seed_values: np.ndarray, sim: np.ndarray, splits: np.ndarray) -> tuple[float, float, float]:
    """Exact permutation p of mean(seed values) - mean(sim) and its 95% CI by test inversion."""
    d_obs = seed_values.mean() - sim.mean()
    d_star = split_diffs(np.concatenate([seed_values, sim]), splits)
    p = float(np.mean(np.abs(d_star) >= abs(d_obs) - 1e-12))
    scale = sim.std(ddof=1) if sim.std(ddof=1) > 0 else max(abs(d_obs), 1e-9)
    accepted = []
    for delta in d_obs + CI_GRID * scale:
        shifted = seed_values - delta
        d0 = shifted.mean() - sim.mean()
        ds = split_diffs(np.concatenate([shifted, sim]), splits)
        if np.mean(np.abs(ds) >= abs(d0) - 1e-12) >= 0.05:
            accepted.append(delta)
    lo, hi = (min(accepted), max(accepted)) if accepted else (math.nan, math.nan)
    return p, lo, hi


def compare(real: np.ndarray, sim: np.ndarray, seed_of: np.ndarray, splits: np.ndarray) -> dict:
    real, sim = np.asarray(real, float), np.asarray(sim, float)
    seed_values = np.array([real[seed_of == s].mean() for s in np.unique(seed_of)])
    p, lo, hi = seed_test(seed_values, sim, splits)
    sd = float(sim.std(ddof=1))
    return {"real_mean": float(real.mean()), "sim_mean": float(sim.mean()), "diff": float(real.mean() - sim.mean()),
            "sim_sd": sd, "std_diff": float((real.mean() - sim.mean()) / sd) if sd > 0 else math.nan,
            "seed_values": [float(v) for v in seed_values], "p_seed": p, "ci95_low": lo, "ci95_high": hi,
            "real_min": float(real.min()), "real_max": float(real.max()), "sim_min": float(sim.min()),
            "sim_max": float(sim.max())}


def holm(pvalues: dict) -> dict:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[key] = running
    return out


# ---------------------------------------------------------------- figure G6


def figure_g6(data: dict, out: Path, stem: str = "G6_projected_dimension_EN") -> None:
    """Projected dimension P(t) of the recorded runs vs the noise-only control, and the deviation in % of P(1)."""
    ps, pr = data["sim_pdim"], data["real_pdim"]
    x = np.arange(1, ps.shape[1] + 1)
    q = np.quantile(ps, [0.025, 0.975], axis=0)
    base = ps.mean(0)
    scale = ps[:, 0].mean()
    n_sim, n_real = ps.shape[0], pr.shape[0]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.3), sharex=True, layout="constrained",
                                 gridspec_kw={"height_ratios": [1.45, 1.0]})
        ax = axes[0]
        ax.fill_between(x, q[0], q[1], color=NOISE, alpha=0.35, linewidth=0,
                        label=f"Noise-only control: 2.5th–97.5th\npercentiles over {n_sim} seeds")
        ax.plot(x, ps.mean(0), color=NOISE, linestyle=(0, (4, 1.5)), linewidth=1.3,
                label=rf"Noise-only mean, $\bar{{P}}^{{(t)}}_{{\mathrm{{noise}}}}$ ({n_sim} seeds)")
        for i, row in enumerate(pr):
            ax.plot(x, row, color=OBS, linewidth=0.45, alpha=0.45,
                    label=f"Observed DP-SVD-FL runs ({n_real} runs)" if i == 0 else None)
        ax.set_ylabel(r"$P^{(t)}$ ($\times 10^{3}$ coordinates)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e3:g}"))
        ax.set_title(r"(a) Projected subspace dimension, $P^{(t)}=\Sigma_{\ell\in\mathcal{P}}\, k_\ell(t)^2$",
                     loc="left")
        leg = ax.legend(loc="upper right", handlelength=1.8, borderaxespad=0.2, labelspacing=0.35)
        for handle in leg.legend_handles:
            if isinstance(handle, matplotlib.lines.Line2D):
                handle.set_alpha(1.0)
                handle.set_linewidth(max(handle.get_linewidth(), 1.0))

        ax = axes[1]
        ax.fill_between(x, (q[0] - base) / scale * 100, (q[1] - base) / scale * 100, color=NOISE, alpha=0.35,
                        linewidth=0, label=f"Noise-only control: 2.5th–97.5th percentiles ({n_sim} seeds)")
        for i, row in enumerate(pr):
            ax.plot(x, (row - base) / scale * 100, color=OBS, linewidth=0.4, alpha=0.5,
                    label="Individual observed run" if i == 0 else None)
        ax.plot(x, (pr.mean(0) - base) / scale * 100, color=OBS_DARK, linewidth=1.3,
                label=f"Mean of the {n_real} observed runs")
        ax.axhline(0, color=INK, linewidth=0.6)
        ax.set_ylabel(r"$\Delta^{(t)}$ (%)")
        ax.set_title("(b) Deviation from the noise-only mean,\n"
                     r"$\Delta^{(t)}=100\,\left(P^{(t)}-\bar{P}^{(t)}_{\mathrm{noise}}\right)/\,P^{(1)}$", loc="left")
        ax.set_xlabel(r"Communication round $t$")
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_major_locator(MultipleLocator(50))
        leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1, handlelength=1.8, labelspacing=0.3)
        leg.legend_handles[1].set_linewidth(0.8)  # individual run: thin and translucent, as in the plot
        for ax in axes:
            ax.grid(True)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(top=False, right=False)
        fig.savefig(out / f"{stem}.pdf", metadata={"Creator": "noise_only_statistics.py", "CreationDate": None})
        fig.savefig(out / f"{stem}.png", dpi=600)
        plt.close(fig)


# ---------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True,
                        help="root directories searched recursively for adaptive run directories")
    parser.add_argument("--sim", type=Path, required=True, help="noise-only simulation output directory")
    parser.add_argument("--out", type=Path, required=True, help="output directory (must be empty or absent)")
    parser.add_argument("--primary", nargs="+", default=PRIMARY_DEFAULT,
                        help="pre-specified layers of the primary Holm family")
    parser.add_argument("--late-window", type=int, nargs=2, default=[251, 300], metavar=("FIRST", "LAST"),
                        help="rounds of the S_late window (inclusive, default 251 300)")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        print(f"refused: {args.out} exists and is not empty")
        return 2
    args.out.mkdir(parents=True, exist_ok=True)

    data = load(args.sim, args.runs)
    T, runs = data["T"], data["runs"]
    if data["real_pdim"].shape[1] != T:
        raise ValueError("real runs and simulation have a different number of rounds")
    late = slice(args.late_window[0] - 1, args.late_window[1])
    seed_of = np.array([s for _, _, s in runs])
    n_real_units, n_sim = len(np.unique(seed_of)), len(data["seeds"])
    splits = np.array(list(itertools.combinations(range(n_real_units + n_sim), n_real_units)))
    splits_fresh = np.array(list(itertools.combinations(range(n_real_units + n_sim - 3), n_real_units)))
    print(f"{len(runs)} real runs ({n_real_units} experiment seeds), {n_sim} simulation seeds, T = {T}; "
          f"{len(splits)} exact splits")

    # ---- per-trajectory summaries and seed-level tests
    summ = {"sim": {}, "real": {}}
    traj_rows, test_rows, tests = [], [], {layer: {} for layer in PROJECTED}
    for layer in PROJECTED:
        for source in ("sim", "real"):
            block = data[source][layer]
            summ[source][layer] = trajectory_summaries(block["gap_ratio"], block["k"], block["gap"], late)
        for i, seed in enumerate(data["seeds"]):
            traj_rows.append({"source": "noise_only", "dataset": "", "partition": "", "seed": seed, "layer": layer,
                              **{k: v[i] for k, v in summ["sim"][layer].items()}})
        for i, (d, p, s) in enumerate(runs):
            traj_rows.append({"source": "real", "dataset": d, "partition": p, "seed": s, "layer": layer,
                              **{k: v[i] for k, v in summ["real"][layer].items()}})
        for stat in STATS:
            res = compare(summ["real"][layer][stat], summ["sim"][layer][stat], seed_of, splits)
            tests[layer][stat] = res
            test_rows.append({"layer": layer, "short": short_name(layer), "primary_layer": layer in args.primary,
                              "statistic": stat, **{k: v for k, v in res.items() if not isinstance(v, list)},
                              "seed_values": " ".join(f"{v:.6g}" for v in res["seed_values"])})
    pd.DataFrame(traj_rows).to_csv(args.out / "trajectory_summaries.csv", index=False)
    tests_df = pd.DataFrame(test_rows)
    tests_df["holm_family"] = ""
    tests_df["p_holm"] = np.nan
    prim = tests_df[tests_df.primary_layer & (tests_df.statistic == "S_late")]
    for idx, p in holm(prim.p_seed.to_dict()).items():
        tests_df.loc[idx, ["holm_family", "p_holm"]] = [f"primary ({len(prim)} layers x S_late)", p]
    for stat in STATS:
        expl = tests_df[(~tests_df.primary_layer) & (tests_df.statistic == stat) & tests_df.p_seed.notna()]
        for idx, p in holm(expl.p_seed.to_dict()).items():
            tests_df.loc[idx, ["holm_family", "p_holm"]] = [f"exploratory ({len(expl)} layers x {stat})", p]
    tests_df.to_csv(args.out / "trajectory_level_tests.csv", index=False)

    sensitivity = {}
    if n_sim > 3:
        for layer in args.primary:
            res = compare(summ["real"][layer]["S_late"], summ["sim"][layer]["S_late"][3:], seed_of, splits_fresh)
            sensitivity[short_name(layer)] = {k: res[k] for k in ("diff", "p_seed", "ci95_low", "ci95_high")}

    # ---- per-layer ranks and branches
    per_layer = []
    for layer in PROJECTED:
        k_t, sl = tests[layer]["k_T"], tests[layer]["S_late"]
        per_layer.append({"layer": layer, "short": short_name(layer), "primary": layer in args.primary,
                          "k_1": int(summ["sim"][layer]["k_1"][0]),
                          "k_1_real_all_equal": bool(np.all(summ["real"][layer]["k_1"] == summ["sim"][layer]["k_1"][0])),
                          "k_T_real_mean": k_t["real_mean"], "k_T_sim_mean": k_t["sim_mean"], "k_T_diff": k_t["diff"],
                          "k_T_p_seed": k_t["p_seed"],
                          "gap_rounds_real": int(summ["real"][layer]["n_gap"].sum()), "rounds_real": len(runs) * T,
                          "gap_rounds_sim": int(summ["sim"][layer]["n_gap"].sum()), "rounds_sim": n_sim * T,
                          "S_late_real_mean": sl["real_mean"], "S_late_sim_mean": sl["sim_mean"],
                          "S_late_diff": sl["diff"], "S_late_std_diff": sl["std_diff"], "S_late_p_seed": sl["p_seed"],
                          "S_late_p_holm": float(tests_df.loc[(tests_df.layer == layer)
                                                              & (tests_df.statistic == "S_late"), "p_holm"].iloc[0]),
                          "S_late_ci95_low": sl["ci95_low"], "S_late_ci95_high": sl["ci95_high"]})
    per_layer_df = pd.DataFrame(per_layer)
    per_layer_df.to_csv(args.out / "per_layer_ranks_branches_S_late.csv", index=False)

    # ---- Table S3: projected private dimension per dataset
    ps, pr = data["sim_pdim"], data["real_pdim"]
    d_rows = []
    for dataset in [d for d in DATASETS if d in data["U"]] + sorted(set(data["U"]) - set(DATASETS)):
        idx = [i for i, (d, _, _) in enumerate(runs) if d == dataset]
        u = data["U"][dataset]
        real_d, sim_d = pr[idx] + u, ps + u
        red_real, red_sim = 100 * (1 - real_d[:, -1] / real_d[:, 0]), 100 * (1 - sim_d[:, -1] / sim_d[:, 0])
        mae = np.abs(real_d.mean(0) - sim_d.mean(0))
        d_rows.append({"dataset": dataset, "U": u, "real_runs": len(idx), "D1": float(real_d[0, 0]),
                       "D_T_real_mean": float(real_d[:, -1].mean()), "D_T_real_min": float(real_d[:, -1].min()),
                       "D_T_real_max": float(real_d[:, -1].max()), "D_T_sim_mean": float(sim_d[:, -1].mean()),
                       "D_T_sim_sd": float(sim_d[:, -1].std(ddof=1)),
                       "D_T_sim_min": float(sim_d[:, -1].min()), "D_T_sim_max": float(sim_d[:, -1].max()),
                       "reduction_pct_real_mean": float(red_real.mean()), "reduction_pct_real_min": float(red_real.min()),
                       "reduction_pct_real_max": float(red_real.max()), "reduction_pct_sim_mean": float(red_sim.mean()),
                       "reduction_pct_sim_q025": float(np.quantile(red_sim, 0.025)),
                       "reduction_pct_sim_q975": float(np.quantile(red_sim, 0.975)),
                       "MAE_mean_curves": float(mae.mean()), "MAE_mean_curves_pct_D1": float(100 * mae.mean() / real_d[0, 0]),
                       "max_abs_diff_mean_curves_pct_D1": float(100 * mae.max() / real_d[0, 0])})
    table_s3 = pd.DataFrame(d_rows)
    table_s3.to_csv(args.out / "table_S3_projected_dimension.csv", index=False)

    # ---- figure
    figure_g6(data, args.out)

    primary = {short_name(layer): {k: tests[layer]["S_late"][k] for k in ("diff", "std_diff", "p_seed", "ci95_low",
                                                                          "ci95_high", "seed_values")}
               | {"p_holm": float(per_layer_df.loc[per_layer_df.layer == layer, "S_late_p_holm"].iloc[0])}
               for layer in args.primary}
    summary = {
        "inputs": {"real_runs": len(runs), "experiment_seeds": int(n_real_units), "simulation_seeds": n_sim,
                   "rounds": T, "late_window": args.late_window, "exact_splits": int(len(splits))},
        "table_S3": d_rows,
        "MAE_mean_curves_pct_D1_range": [float(table_s3.MAE_mean_curves_pct_D1.min()),
                                         float(table_s3.MAE_mean_curves_pct_D1.max())],
        "max_abs_final_rank_diff": float(per_layer_df.k_T_diff.abs().max()),
        "gap_branch": {row["short"]: [row["gap_rounds_real"], row["rounds_real"], row["gap_rounds_sim"], row["rounds_sim"]]
                       for row in per_layer if row["gap_rounds_real"] or row["gap_rounds_sim"]},
        "primary_S_late": primary,
        "sensitivity_without_first_3_sim_seeds": sensitivity,
        "holm_significant_005": tests_df[tests_df.p_holm < 0.05][["short", "statistic", "diff", "p_seed", "p_holm",
                                                                    "holm_family"]].to_dict(orient="records"),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")

    md = ["# Recorded adaptive runs vs noise-only control", "",
          f"{len(runs)} recorded runs ({n_real_units} experiment seeds) vs {n_sim} noise-only simulation seeds, "
          f"T = {T}.", "", "## Table S3: decrease of D from round 1 to round T", "",
          "| dataset | D(1) | real decrease (%) | noise-only decrease (%) | MAE of mean curves (% of D(1)) |",
          "|---|---|---|---|---|"]
    for row in d_rows:
        md.append(f"| {row['dataset']} | {row['D1']:.0f} | {row['reduction_pct_real_mean']:.3f} | "
                  f"{row['reduction_pct_sim_mean']:.3f} | {row['MAE_mean_curves_pct_D1']:.3f} |")
    md += ["", "## Per layer", "",
           "| layer | k(1) | mean k(T) real | mean k(T) noise | diff | gap real | gap noise | S_late diff | p (seed) | p Holm |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for row in per_layer:
        md.append(f"| {row['short']}{' *' if row['primary'] else ''} | {row['k_1']} | {row['k_T_real_mean']:.2f} | "
                  f"{row['k_T_sim_mean']:.2f} | {row['k_T_diff']:+.2f} | {row['gap_rounds_real']}/{row['rounds_real']} | "
                  f"{row['gap_rounds_sim']}/{row['rounds_sim']} | {row['S_late_diff']:+.3f} | "
                  f"{row['S_late_p_seed']:.3f} | {row['S_late_p_holm']:.3f} |")
    md += ["", "`*` = pre-specified primary layers (Holm family of two); the other layers form the exploratory family.",
           f"S_late = mean r_max over rounds {args.late_window[0]}-{args.late_window[1]}; p = exact seed-level "
           "permutation test (unit = experiment seed).", "",
           f"Holm-significant (< 0.05): {len(summary['holm_significant_005'])} tests.",
           f"Sensitivity without the first three simulation seeds: {json.dumps(sensitivity, default=float)}"]
    (args.out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
