#!/usr/bin/env python
"""Regenerate the three main figures of the paper from run directories (CPU only, read-only on the runs).

Example (layout written by ``scripts/run_experiment.py``)::

    python analysis/make_figures.py --main results/main_e10 --privacy-sweep results/privacy_sweep_e10 --out figures

Figures (each saved as PDF and as 300-dpi PNG):

  fig01_learning_curves_dpsvdfl        DP-SVD-FL test accuracy vs round (rounds 10..300) for CIFAR-10 and
                                       EMNIST Balanced, Dirichlet and IID; 3-seed mean, band = +/-1 sample std;
                                       no smoothing (the DP-SVD-FL curves of Fig. 1).
  fig02_privacy_utility_e10            DP-SVD-FL final accuracy vs nominal epsilon of the CIFAR-10 Dirichlet
                                       privacy sweep; a point is drawn only when all seeds of that sigma are
                                       complete; error bars = +/-1 sample std; top axis = sigma.
  fig03_adaptive_dimension_e10         100 * D_t / D_full of adaptive DP-SVD-FL over the 300 rounds, Dirichlet
                                       runs of the main experiment, CIFAR-10 and EMNIST Balanced.

``figures_report.json`` lists the plotted values, the runs used (paths relative to their root) and the
runs that were excluded. Run discovery and validation are those of ``analysis/common.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    CAMPAIGNS, DEFAULT_SEEDS, Campaign, evaluation_curve, load_campaign, summarize_group,
)

# ----------------------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "grid.linewidth": 0.4,
    "grid.color": "#d9d9d9",
    "lines.linewidth": 1.2,
    "lines.markersize": 3.2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "legend.frameon": False,
})

IEEE_DOUBLE_COLUMN_IN = 7.1
IEEE_SINGLE_COLUMN_IN = 3.5

# Okabe-Ito vermillion, as in the paper.
METHOD_STYLE = {
    "adaptive": dict(label="DP-SVD-FL", color="#D55E00", ls="-", marker="o"),
}
FIG1_METHODS = ["adaptive"]
FIG1_PANELS = [("cifar10", "dirichlet", "(a) CIFAR-10, Dirichlet"), ("cifar10", "iid", "(b) CIFAR-10, IID"),
               ("emnist_balanced", "dirichlet", "(c) EMNIST Balanced, Dirichlet"),
               ("emnist_balanced", "iid", "(d) EMNIST Balanced, IID")]
FIG2_METHODS = ["adaptive"]
FIG3_PANELS = [("cifar10", "(a) CIFAR-10, Dirichlet"), ("emnist_balanced", "(b) EMNIST Balanced, Dirichlet")]


def panel_title(base: str, alpha) -> str:
    return base + (rf" $\alpha={alpha:g}$" if alpha is not None else "")


def save(fig, out: Path, stem: str, formats) -> list[str]:
    files = []
    for ext in formats:
        path = out / f"{stem}.{ext}"
        kwargs = {"dpi": 300} if ext == "png" else {}
        if ext == "pdf":
            kwargs["metadata"] = {"Creator": "analysis/make_figures.py", "CreationDate": None}
        fig.savefig(path, **kwargs)
        files.append(path.name)
    plt.close(fig)
    return files


def single_group(campaign: Campaign | None, dataset, partition, method, sigma=None):
    if campaign is None:
        return None
    return campaign.one(dataset, partition, method, sigma)


def main_sigma(campaign: Campaign) -> float | None:
    sigmas = sorted({g.sigma for g in campaign.groups.values()})
    if len(sigmas) > 1:
        campaign.warnings.append(f"main campaign has several sigma levels {sigmas}; figures use {sigmas[0]}")
    return sigmas[0] if sigmas else None


# ----------------------------------------------------------------------------- figure 1
def figure1(out, main, seeds, formats):
    name = "fig01_learning_curves_dpsvdfl"
    sigma = main_sigma(main)
    fig, axes = plt.subplots(2, 2, figsize=(IEEE_DOUBLE_COLUMN_IN, 4.75), sharex=True, sharey="row")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.095, top=0.865, wspace=0.08, hspace=0.30)
    info = {"sigma": sigma, "statistic": f"mean over seeds {list(seeds)}; band = sample std (ddof=1)",
            "x": "evaluation round (recorded rounds only; no interpolation, no smoothing)",
            "y": "test accuracy x 100", "panels": [], "missing": []}
    for ax, (ds, part, title) in zip(axes.flat, FIG1_PANELS):
        alpha = None
        panel = {"dataset": ds, "partition": part, "methods": {}}
        for k, m in enumerate(FIG1_METHODS):
            g = single_group(main, ds, part, m, sigma)
            if g is None or not g.complete(seeds):
                info["missing"].append({"dataset": ds, "partition": part, "method": m,
                                        "seeds_complete": sorted(g.runs) if g else [],
                                        "reason": "not all required seeds complete"})
                continue
            alpha = g.alpha if g.alpha is not None else alpha
            runs = g.selected(seeds)
            curve = evaluation_curve(runs)
            st = METHOD_STYLE[m]
            x = curve["rounds"]
            y = [100 * v for v in curve["mean"]]
            ax.plot(x, y, color=st["color"], ls=st["ls"], marker=st["marker"], markevery=(k % 3, 3),
                    markerfacecolor="white", markeredgewidth=0.8, label=st["label"], zorder=3)
            ax.fill_between(x, [100 * (mu - sd) for mu, sd in zip(curve["mean"], curve["std"])],
                            [100 * (mu + sd) for mu, sd in zip(curve["mean"], curve["std"])],
                            color=st["color"], alpha=0.15, linewidth=0, zorder=2)
            panel["methods"][m] = {"campaign": g.campaign, "runs": [r.location() for r in runs],
                                   "rounds": x, "mean_pct": y, "std_pct": [100 * v for v in curve["std"]]}
        ax.set_title(panel_title(title, alpha), loc="left", pad=3)
        ax.grid(True, axis="y")
        ax.set_xlim(0, 305)
        ax.set_xticks(range(0, 301, 50))
        info["panels"].append(panel)
    for ax in axes[1]:
        ax.set_xlabel("Communication rounds")
    for ax in axes[:, 0]:
        ax.set_ylabel("Test accuracy (%)")
    handles = {}
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    order = [METHOD_STYLE[m]["label"] for m in FIG1_METHODS if METHOD_STYLE[m]["label"] in handles]
    fig.legend([handles[l] for l in order], order, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.995),
               columnspacing=1.6, handlelength=3.2, borderaxespad=0.2)
    info["files"] = save(fig, out, name, formats)
    return name, info


# ----------------------------------------------------------------------------- figure 2
def figure2(out, sweep, seeds, formats):
    name = "fig02_privacy_utility_e10"
    settings = sorted({(g.dataset, g.partition) for g in sweep.groups.values()})
    if ("cifar10", "dirichlet") not in settings:
        sweep.warnings.append("no CIFAR-10 Dirichlet runs in the privacy sweep; figure 2 not drawn")
        return name, None
    sigmas = sorted({g.sigma for g in sweep.groups.values() if (g.dataset, g.partition) == ("cifar10", "dirichlet")})
    data, missing = {}, []
    for m in FIG2_METHODS:
        pts = []
        for s in sigmas:
            g = single_group(sweep, "cifar10", "dirichlet", m, s)
            if g is None or not g.complete(seeds):
                missing.append({"method": m, "sigma": s, "seeds_complete": sorted(g.runs) if g else []})
                pts.append({"sigma": s, "available": False})
                continue
            summ = summarize_group(g, seeds)
            if summ["epsilon"] is None:
                missing.append({"method": m, "sigma": s, "reason": "recorded epsilon differs across seeds"})
                pts.append({"sigma": s, "available": False})
                continue
            pts.append({"sigma": s, "available": True, "epsilon": summ["epsilon"],
                        "mean_pct": 100 * summ["accuracy_mean"], "std_pct": 100 * summ["accuracy_std"],
                        "per_seed_pct": {str(k): 100 * v for k, v in summ["per_seed_accuracy"].items()},
                        "runs": summ["runs"]})
        data[m] = pts
    # Sigma levels that exist only as excluded (incomplete) runs are reported, not drawn.
    excluded_sigmas = sorted({round(float(e["sigma"]), 6) for e in sweep.excluded if e.get("sigma") is not None}
                             - set(sigmas))

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COLUMN_IN, 2.75))
    fig.subplots_adjust(left=0.135, right=0.975, bottom=0.155, top=0.845)
    for m in FIG2_METHODS:
        st = METHOD_STYLE[m]
        pts = data[m]
        for a, b in zip(pts, pts[1:]):  # connect only adjacent sigma levels that are both drawn
            if a["available"] and b["available"]:
                ax.plot([a["epsilon"], b["epsilon"]], [a["mean_pct"], b["mean_pct"]], color=st["color"],
                        ls=st["ls"], lw=1.0, zorder=2)
        av = [p for p in pts if p["available"]]
        if av:
            ax.errorbar([p["epsilon"] for p in av], [p["mean_pct"] for p in av], yerr=[p["std_pct"] for p in av],
                        fmt=st["marker"], color=st["color"], markerfacecolor="white", markeredgewidth=0.9,
                        markersize=4.2, capsize=2.2, elinewidth=0.8, capthick=0.8, label=st["label"], zorder=3)
    ticks = sorted({p["epsilon"]: p["sigma"] for m in data for p in data[m] if p["available"]}.items())
    if ticks:
        eps = [e for e, _ in ticks]
        ax.set_xticks(eps)
        ax.set_xticklabels([f"{e:.2f}" for e in eps])
        span = max(eps) - min(eps)
        ax.set_xlim(min(eps) - 0.1 * span - 0.5, max(eps) + 0.1 * span + 0.5)
        top = ax.secondary_xaxis("top")
        top.set_xticks(eps)
        top.set_xticklabels([f"{s:g}" for _, s in ticks])
        top.set_xlabel(r"Noise multiplier $\sigma$", labelpad=3)
        top.tick_params(direction="in", length=2.5, width=0.6)
    deltas = {r.delta for r in sweep.runs}
    delta_txt = rf" ($\delta={deltas.pop():g}$)" if len(deltas) == 1 else ""
    ax.set_xlabel(r"Nominal privacy budget $\varepsilon$" + delta_txt)
    ax.set_ylabel("Final test accuracy (%)")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", handletextpad=0.4, borderaxespad=0.4)
    files = save(fig, out, name, formats)
    return name, {"dataset": "cifar10", "partition": "dirichlet", "sigma_levels_with_complete_runs": sigmas,
                  "sigma_levels_without_any_complete_run": excluded_sigmas,
                  "x": "epsilon recorded in public_summary.json (privacy.epsilon)",
                  "y": f"final (round 300) test accuracy x 100, mean +/- sample std over seeds {list(seeds)}",
                  "inclusion": "a point only when all required seeds are complete; lines only between adjacent "
                               "sigma levels that are both drawn",
                  "methods": data, "not_drawn": missing, "files": files}


# ----------------------------------------------------------------------------- figure 3
def figure3(out, main, seeds, formats):
    name = "fig03_adaptive_dimension_e10"
    sigma = main_sigma(main)
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COLUMN_IN, 2.45), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.88, wspace=0.08)
    panels, missing = [], []
    st = METHOD_STYLE["adaptive"]
    for ax, (ds, title) in zip(axes, FIG3_PANELS):
        g = single_group(main, ds, "dirichlet", "adaptive", sigma)
        if g is None or not g.complete(seeds):
            missing.append({"dataset": ds, "seeds_complete": sorted(g.runs) if g else []})
            ax.set_title(title, loc="left", pad=3)
            continue
        runs = g.selected(seeds)
        n_rounds = len(runs[0].records)
        ratios = [[100.0 * x["D"] / x["D_full"] for x in r.records] for r in runs]
        mean = [statistics.mean(v[i] for v in ratios) for i in range(n_rounds)]
        sd = [statistics.stdev(v[i] for v in ratios) if len(ratios) > 1 else 0.0 for i in range(n_rounds)]
        rounds = list(range(1, n_rounds + 1))
        ax.fill_between(rounds, [m - s for m, s in zip(mean, sd)], [m + s for m, s in zip(mean, sd)],
                        color=st["color"], alpha=0.22, linewidth=0, label=rf"$\pm 1$ SD (max {max(sd):.3f} pp)")
        ax.plot(rounds, mean, color=st["color"], lw=1.3, label=f"Mean over seeds {', '.join(map(str, seeds))}")
        ax.plot([1, n_rounds], [mean[0], mean[-1]], ls="none", marker="o", color=st["color"],
                markerfacecolor="white", markeredgewidth=0.9, markersize=4, zorder=4)
        ax.annotate(f"initial {mean[0]:.2f}%", (1, mean[0]), xytext=(7, 1), textcoords="offset points",
                    va="bottom", ha="left", fontsize=7.5)
        ax.annotate(f"final {mean[-1]:.2f}%", (n_rounds, mean[-1]), xytext=(-2, -7), textcoords="offset points",
                    va="top", ha="right", fontsize=7.5)
        ax.set_title(panel_title(title, g.alpha), loc="left", pad=3)
        ax.set_xlim(0, n_rounds + 5)
        ax.set_xticks(range(0, n_rounds + 1, 50))
        ax.grid(True, axis="y")
        ax.legend(loc="upper right", handlelength=2.2)
        D_full = sorted({x["D_full"] for r in runs for x in r.records})
        panels.append({"dataset": ds, "runs": [r.location() for r in runs], "D_full": D_full,
                       "ratio_pct_mean": mean, "ratio_pct_std": sd, "initial_pct": mean[0], "final_pct": mean[-1],
                       "time_mean_pct": statistics.mean(mean)})
    for ax in axes:
        ax.set_xlabel("Communication rounds")
    axes[0].set_ylabel("Privatized dim. / full update dim. (%)")
    if panels:
        lo = min(min(p["ratio_pct_mean"]) - max(p["ratio_pct_std"]) for p in panels)
        hi = max(max(p["ratio_pct_mean"]) + max(p["ratio_pct_std"]) for p in panels)
        axes[0].set_ylim(math.floor(lo * 2) / 2 - 0.25, math.ceil(hi * 2) / 2 + 0.5)
    files = save(fig, out, name, formats)
    return name, {"method": "adaptive", "sigma": sigma,
                  "quantity": "100 * D_t / D_full (fields D and D_full of released_round_diagnostics.jsonl)",
                  "statistic": f"mean over seeds {list(seeds)}; band = sample std (ddof=1)",
                  "panels": panels, "missing": missing, "files": files}


# ----------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Regenerate the paper's main figures from run directories.")
    p.add_argument("--main", nargs="+", required=True, metavar="ROOT", help="roots of the main DP-SVD-FL runs")
    p.add_argument("--privacy-sweep", nargs="+", metavar="ROOT", help="roots of the privacy-sweep runs (figure 2)")
    p.add_argument("--fixed-rank", nargs="+", metavar="ROOT",
                   help="accepted for symmetry with make_tables.py; not used by the figures")
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
                   help="seeds that must all be complete for a curve or point to be drawn (default: 0 1 2)")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"], help="output formats (default: pdf png)")
    p.add_argument("--out", required=True, type=Path, help="output directory")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    seeds = tuple(args.seeds)
    args.out.mkdir(parents=True, exist_ok=True)
    campaigns = {}
    for name, roots in (("main", args.main), ("privacy_sweep", args.privacy_sweep)):
        if roots:
            campaigns[name] = c = load_campaign(name, [Path(r) for r in roots])
            print(f"[{name}] {len(c.runs)} complete runs included, {len(c.excluded)} excluded, "
                  f"{len(c.ignored)} ignored (other methods)")
        else:
            print(f"[{name}] no roots given: {CAMPAIGNS[name]} not plotted")
    if args.fixed_rank:
        print("[fixed_rank] not used by the figures")
    main_c, sweep = campaigns["main"], campaigns.get("privacy_sweep")

    figures, skipped = {}, []
    name, info = figure1(args.out, main_c, seeds, args.formats)
    figures[name] = info
    if sweep is not None:
        name, info = figure2(args.out, sweep, seeds, args.formats)
        if info is None:
            skipped.append(f"{name}: no CIFAR-10 Dirichlet runs in the privacy sweep")
        else:
            figures[name] = info
    else:
        skipped.append("fig02_privacy_utility_e10: no --privacy-sweep roots")
    name, info = figure3(args.out, main_c, seeds, args.formats)
    figures[name] = info

    report = {"matplotlib": matplotlib.__version__, "figures": figures, "skipped": skipped,
              "campaigns": {k: {kk: v for kk, v in c.report().items() if kk != "groups"}
                            for k, c in campaigns.items()}}
    (args.out / "figures_report.json").write_text(json.dumps(report, indent=1) + "\n")
    for s in skipped:
        print(f"skipped {s}")
    for name, c in campaigns.items():
        for w in c.warnings:
            print(f"warning [{name}]: {w}")
    for name, info in figures.items():
        for item in info.get("missing", []) + info.get("not_drawn", []):
            print(f"{name}: not drawn {item}")
        for s in info.get("sigma_levels_without_any_complete_run", []):
            print(f"{name}: sigma {s:g} has no complete run; not drawn")
        print(f"wrote {', '.join(info['files'])}")


if __name__ == "__main__":
    main()
