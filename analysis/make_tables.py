#!/usr/bin/env python
"""Regenerate the result tables of the paper from run directories (CPU only, read-only on the runs).

Example (layout written by ``scripts/run_experiment.py``)::

    python analysis/make_tables.py --main results/main_e10 --privacy-sweep results/privacy_sweep_e10 \\
        --fixed-rank results/fixed_rank_e10 --out tables

Only ``--main`` is required; each campaign option accepts one or more roots, which are searched
recursively for run directories (see ``analysis/common.py``). Tables whose campaign is not given are
skipped with a message. For each table the script writes ``<name>.csv`` (full precision, long format),
``<name>.md`` and ``<name>.tex`` (a LaTeX ``tabular`` snippet, needs ``booktabs``); ``tables_report.json``
lists the included runs (paths relative to their root), the excluded/incomplete runs with the reason,
and the ignored runs of other methods.

Statistics: final test accuracy at round 300 (never the best checkpoint), mean +/- sample standard
deviation (ddof = 1) over seeds 0, 1, 2, in percent with two decimals. A cell is filled only when every
required seed is complete.

Tables:
  table3_dpsvdfl_accuracy        Table III  final accuracy of DP-SVD-FL (3 datasets x 2 partitions)
  table4_privatized_dimension    Table IV   privatized-vector dimension of DP-SVD-FL vs the full update
  table5_privacy_sweep           Table V    privacy sweep of DP-SVD-FL (CIFAR-10 Dirichlet), all sigma levels
  fixed_rank_ablation            fixed-rank spectral projection vs adaptive DP-SVD-FL
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DATASET_LABELS, DATASET_ORDER, DEFAULT_SEEDS, METHOD_LABELS, PARTITION_ORDER,
    CAMPAIGNS, Campaign, load_campaign, pm, setting_label, summarize_group,
)

ADAPTIVE = "adaptive"
FIXED = "fixed_spectral"


# ----------------------------------------------------------------------------- writers
def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in r.items()})


def markdown(header: list[str], rows: list[list[str]], title: str = "", notes: list[str] = ()) -> str:
    lines = [f"### {title}", ""] if title else []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join([":--"] + ["--:"] * (len(header) - 1)) + "|")
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    if notes:
        lines.append("")
        lines += [f"{n}" for n in notes]
    return "\n".join(lines) + "\n"


def latex(header: list[str], rows: list[list[str]]) -> str:
    def tex(cell: str) -> str:
        return (cell.replace(" ± ", "$\\pm$").replace("±", "$\\pm$").replace("%", "\\%")
                .replace("−", "$-$").replace("σ", "$\\sigma$").replace("ε", "$\\varepsilon$")
                .replace("_", "\\_").replace("→", "$\\to$"))
    spec = "l" + "c" * (len(header) - 1)
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(tex(h) for h in header) + " \\\\", "\\midrule"]
    lines += [" & ".join(tex(c) for c in r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def emit(out: Path, stem: str, csv_rows: list[dict], header: list[str], rows: list[list[str]],
         title: str, notes: list[str] = ()) -> list[str]:
    write_csv(out / f"{stem}.csv", csv_rows)
    (out / f"{stem}.md").write_text(markdown(header, rows, title, notes))
    (out / f"{stem}.tex").write_text(latex(header, rows))
    return [f"{stem}.csv", f"{stem}.md", f"{stem}.tex"]


def pct_value(x):
    return None if x is None else 100.0 * x


def cell(summary: dict | None) -> str:
    if summary is None:
        return "n/a"
    if not summary["complete"]:
        return f"n/a ({summary['n_seeds']}/{len(summary['required_seeds'])} seeds)"
    return pm(summary["accuracy_mean"], summary["accuracy_std"])


def long_row(summary: dict) -> dict:
    return {"campaign": summary["campaign"], "dataset": summary["dataset"], "partition": summary["partition"],
            "dirichlet_alpha": summary["dirichlet_alpha"], "sigma": summary["sigma"],
            "epsilon": summary["epsilon"], "method": summary["method"],
            "method_label": METHOD_LABELS.get(summary["method"], summary["method"]),
            "n_seeds": summary["n_seeds"], "complete": summary["complete"],
            "accuracy_mean_pct": pct_value(summary["accuracy_mean"]),
            "accuracy_std_pct": pct_value(summary["accuracy_std"]),
            "cell": cell(summary),
            "per_seed_accuracy_pct": {str(k): 100.0 * v for k, v in summary["per_seed_accuracy"].items()},
            "runs": summary["runs"]}


def settings_of(campaign: Campaign) -> list[tuple]:
    keys = {(g.dataset, g.partition, g.alpha, g.sigma) for g in campaign.groups.values()}
    return sorted(keys, key=lambda k: (DATASET_ORDER.index(k[0]) if k[0] in DATASET_ORDER else 99,
                                       PARTITION_ORDER.index(k[1]) if k[1] in PARTITION_ORDER else 99,
                                       k[2] if k[2] is not None else -1, k[3]))


def lookup(campaign: Campaign | None, dataset, partition, alpha, sigma, method):
    if campaign is None:
        return None
    groups = [g for g in campaign.find(dataset, partition, method, sigma) if g.alpha == alpha]
    if len(groups) > 1:
        campaign.warnings.append(f"{len(groups)} configurations for {dataset}/{partition}/{method}/sigma={sigma}; "
                                 "not pooled, the first one is used")
    return groups[0] if groups else None


def row_label(dataset, partition, alpha, sigma, multi_sigma=False):
    label = setting_label(dataset, partition)
    return f"{label} (σ={sigma:g})" if multi_sigma else label


# ----------------------------------------------------------------------------- Table III
def table3(out, main, seeds):
    settings = settings_of(main)
    multi_sigma = len({s[3] for s in settings}) > 1
    header = ["Setting", METHOD_LABELS[ADAPTIVE]]
    rows, csv_rows = [], []
    for ds, part, alpha, sigma in settings:
        g = lookup(main, ds, part, alpha, sigma, ADAPTIVE)
        s = summarize_group(g, seeds) if g else None
        rows.append([row_label(ds, part, alpha, sigma, multi_sigma), cell(s)])
        if s:
            csv_rows.append(long_row(s))
    notes = [f"Final test accuracy (%) at round 300, mean ± sample std (ddof=1) over seeds {list(seeds)}."]
    return emit(out, "table3_dpsvdfl_accuracy", csv_rows, header, rows, "Table III: DP-SVD-FL final accuracy", notes)


# ----------------------------------------------------------------------------- Table IV
def dimension_stats(group, seeds):
    runs = group.selected(seeds)
    D = [[rec["D"] for rec in r.records] for r in runs]
    D_full_values = sorted({rec["D_full"] for r in runs for rec in r.records})
    D_full = D_full_values[0] if len(D_full_values) == 1 else None
    complete = group.complete(seeds)
    per_seed_mean = [statistics.mean(d) for d in D]
    return {
        "D_full": D_full, "D_full_values": D_full_values, "complete": complete, "n_seeds": len(runs),
        "D_mean_all_rounds_seeds": statistics.mean(per_seed_mean) if complete else None,
        "D_round1_mean": statistics.mean(d[0] for d in D) if complete else None,
        "D_final_mean": statistics.mean(d[-1] for d in D) if complete else None,
        "D_constant": len({x for d in D for x in d}) == 1,
        "D_values_if_constant": sorted({x for d in D for x in d}) if len({x for d in D for x in d}) == 1 else None,
        "ratio_round1_pct": statistics.mean(100.0 * r.records[0]["D"] / r.records[0]["D_full"] for r in runs)
        if complete else None,
        "ratio_final_pct": statistics.mean(100.0 * r.records[-1]["D"] / r.records[-1]["D_full"] for r in runs)
        if complete else None,
        "ratio_time_mean_pct": statistics.mean(statistics.mean(100.0 * x["D"] / x["D_full"] for x in r.records)
                                               for r in runs) if complete else None,
        "D_initial_per_seed": {str(r.seed): r.records[0]["D"] for r in runs},
        "D_final_per_seed": {str(r.seed): r.records[-1]["D"] for r in runs},
    }


def fmt_int(x):
    return "n/a" if x is None else f"{round(x):,}"


def fmt_ratio(x):
    return "n/a" if x is None else f"{x:.2f}%"


def table4(out, main, seeds):
    csv_rows, rows = [], []
    for ds, part, alpha, sigma in settings_of(main):
        g = lookup(main, ds, part, alpha, sigma, ADAPTIVE)
        if g is None:
            continue
        st = dimension_stats(g, seeds)
        csv_rows.append({"campaign": "main", "dataset": ds, "partition": part, "dirichlet_alpha": alpha,
                         "sigma": sigma, "method": ADAPTIVE, **st})
        if part == "dirichlet":
            rows.append([DATASET_LABELS.get(ds, ds), fmt_int(st["D_full"]), fmt_int(st["D_mean_all_rounds_seeds"]),
                         fmt_int(st["D_final_mean"]),
                         f"{fmt_ratio(st['ratio_round1_pct'])} → {fmt_ratio(st['ratio_final_pct'])}"])
    header = ["Dataset", "Full update D_full", "DP-SVD-FL mean D", "DP-SVD-FL final D (t=300)",
              "100·D/D_full (t=1 → t=300)"]
    notes = ["Dirichlet settings of the main experiment. Mean D: average over all 300 rounds and seeds "
             f"{list(seeds)}; final D: round 300, mean over seeds. The CSV also lists the IID settings."]
    return emit(out, "table4_privatized_dimension", csv_rows, header, rows,
                "Table IV: privatized-vector dimension", notes)


# ----------------------------------------------------------------------------- Table V
def table5(out, sweep, seeds):
    excluded = [e for e in sweep.excluded if e.get("method") == ADAPTIVE]
    methods = [ADAPTIVE]
    settings = sorted({(g.dataset, g.partition, g.alpha) for g in sweep.groups.values()},
                      key=lambda k: (k[0], k[1]))
    sigmas = sorted({g.sigma for g in sweep.groups.values()}
                    | {round(float(e["sigma"]), 6) for e in excluded if e.get("sigma") is not None})
    csv_rows, rows, incomplete = [], [], []
    for ds, part, alpha in settings:
        for sigma in sigmas:
            row_cells = {}
            for m in methods:
                g = lookup(sweep, ds, part, alpha, sigma, m)
                failed = sorted(e["seed"] for e in excluded if e["method"] == m and e.get("sigma") is not None
                                and math.isclose(float(e["sigma"]), sigma) and e.get("dataset") in (ds, None)
                                and e.get("partition") in (part, None) and e.get("seed") is not None)
                s = summarize_group(g, seeds) if g else None
                r = long_row(s) if s else {"campaign": "privacy_sweep", "dataset": ds, "partition": part,
                                           "dirichlet_alpha": alpha, "sigma": sigma, "epsilon": None, "method": m,
                                           "method_label": METHOD_LABELS.get(m, m), "n_seeds": 0,
                                           "complete": False, "cell": "n/a"}
                r["excluded_seeds"] = failed
                csv_rows.append(r)
                row_cells[m] = r
                if not r["complete"]:
                    incomplete.append(f"{setting_label(ds, part)}, σ={sigma:g}, {METHOD_LABELS.get(m, m)}: "
                                      f"{r['n_seeds']}/{len(seeds)} seeds complete"
                                      + (f"; excluded/incomplete seeds {failed}" if failed else ""))
            eps = {r["epsilon"] for r in row_cells.values() if r.get("epsilon") is not None}
            eps_txt = f"{eps.pop():.4f}" if len(eps) == 1 else ("n/a" if not eps else "differs")
            rows.append([f"{sigma:.3f}", eps_txt] + [row_cells[m]["cell"] if row_cells[m]["complete"]
                                                     else f"n/a ({row_cells[m]['n_seeds']}/{len(seeds)})"
                                                     for m in methods])
    header = ["σ", "ε (δ=0.002)"] + [METHOD_LABELS.get(m, m) for m in methods]
    notes = [f"Privacy sweep, final accuracy (%) at round 300, mean ± sample std over seeds {list(seeds)}; "
             "a value is shown only when all seeds of that σ are complete.",
             "ε is the nominal epsilon recorded in public_summary.json.", "", "Incomplete levels:", ""]
    notes += [f"- {x}" for x in incomplete] or ["- none"]
    files = emit(out, "table5_privacy_sweep", csv_rows, header, rows, "Table V: privacy sweep", notes)
    return files, incomplete


# ----------------------------------------------------------------------------- fixed rank
def fixed_rank(out, fixed, main, seeds):
    csv_rows, rows = [], []
    for g in sorted(fixed.groups.values(), key=lambda g: (g.dataset, g.partition, g.sigma)):
        if g.method != FIXED:
            continue
        s = summarize_group(g, seeds)
        st = dimension_stats(g, seeds)
        ref = lookup(main, g.dataset, g.partition, g.alpha, g.sigma, ADAPTIVE) if main else None
        rs = summarize_group(ref, seeds) if ref else None
        rst = dimension_stats(ref, seeds) if ref else None
        r = long_row(s)
        r.update({"D_constant": st["D_constant"], "D_value_if_constant": st["D_values_if_constant"],
                  "D_full": st["D_full"], "D_mean_all_rounds_seeds": st["D_mean_all_rounds_seeds"],
                  "adaptive_main_cell": cell(rs), "adaptive_main_D_mean": rst["D_mean_all_rounds_seeds"] if rst else None})
        csv_rows.append(r)
        dim = fmt_int(st["D_values_if_constant"][0]) + " (constant)" if st["D_constant"] else \
            fmt_int(st["D_mean_all_rounds_seeds"]) + " (mean)"
        rows.append([setting_label(g.dataset, g.partition), cell(s), dim, cell(rs),
                     fmt_int(rst["D_mean_all_rounds_seeds"]) + " (mean)" if rst else "n/a"])
    header = ["Setting", "Fixed-rank spectral acc.", "Fixed-rank private dim.", "DP-SVD-FL acc. (main)",
              "DP-SVD-FL private dim. (main)"]
    notes = ["Fixed-rank spectral projection (ranks frozen at round 1) vs adaptive DP-SVD-FL of the main campaign."]
    return emit(out, "fixed_rank_ablation", csv_rows, header, rows, "Fixed-rank spectral ablation", notes)


# ----------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Regenerate the paper's result tables from run directories.")
    p.add_argument("--main", nargs="+", required=True, metavar="ROOT", help="roots of the main DP-SVD-FL runs")
    p.add_argument("--privacy-sweep", nargs="+", metavar="ROOT", help="roots of the privacy-sweep runs")
    p.add_argument("--fixed-rank", nargs="+", metavar="ROOT", help="roots of the fixed-rank ablation runs")
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
                   help="seeds that must all be complete for a cell to be reported (default: 0 1 2)")
    p.add_argument("--out", required=True, type=Path, help="output directory")
    return p.parse_args(argv)


def load_all(args) -> dict:
    campaigns = {}
    for name, roots in (("main", args.main), ("privacy_sweep", args.privacy_sweep), ("fixed_rank", args.fixed_rank)):
        if roots:
            campaigns[name] = load_campaign(name, [Path(r) for r in roots])
            c = campaigns[name]
            print(f"[{name}] {len(c.runs)} complete runs included, {len(c.excluded)} excluded, "
                  f"{len(c.ignored)} ignored (other methods)")
        else:
            print(f"[{name}] no roots given: {CAMPAIGNS[name]} skipped")
    return campaigns


def main(argv=None) -> None:
    args = parse_args(argv)
    seeds = tuple(args.seeds)
    args.out.mkdir(parents=True, exist_ok=True)
    campaigns = load_all(args)
    main_c, sweep, fixed = (campaigns.get(k) for k in ("main", "privacy_sweep", "fixed_rank"))
    written, skipped, extra = {}, [], {}

    written["table3"] = table3(args.out, main_c, seeds)
    written["table4"] = table4(args.out, main_c, seeds)
    if sweep is not None:
        written["table5"], extra["privacy_sweep_incomplete_levels"] = table5(args.out, sweep, seeds)
    else:
        skipped.append("table5_privacy_sweep: no --privacy-sweep roots")
    if fixed is not None:
        written["fixed_rank"] = fixed_rank(args.out, fixed, main_c, seeds)
    else:
        skipped.append("fixed_rank_ablation: no --fixed-rank roots")

    all_rows = []
    for c in campaigns.values():
        for g in c.groups.values():
            all_rows.append(long_row(summarize_group(g, seeds)))
    write_csv(args.out / "method_aggregates.csv", all_rows)

    report = {"statistic": f"final test accuracy at round 300; mean and sample std (ddof=1) over seeds {list(seeds)}",
              "files": written, "skipped": skipped, **extra,
              "campaigns": {name: c.report() for name, c in campaigns.items()}}
    (args.out / "tables_report.json").write_text(json.dumps(report, indent=1, sort_keys=False) + "\n")
    for s in skipped:
        print(f"skipped {s}")
    for name, c in campaigns.items():
        for w in c.warnings:
            print(f"warning [{name}]: {w}")
    print(f"wrote {sum(len(v) for v in written.values()) + 2} files to {args.out}")


if __name__ == "__main__":
    main()
