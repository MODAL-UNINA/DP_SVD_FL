#!/usr/bin/env python
"""Run (or resume) the runs of an experiment configuration, one after the other.

Examples
--------
List the 18 runs of the main experiment without running anything::

    python scripts/run_experiment.py --config configs/main_e10.json --list

Run DP-SVD-FL on CIFAR-10 Dirichlet, seed 0, on the first GPU::

    python scripts/run_experiment.py --config configs/main_e10.json --dataset cifar10 --partition dirichlet \\
        --method adaptive --seed 0 --data-root data --output-root results --device cuda:0

Run one level of the privacy sweep::

    python scripts/run_experiment.py --config configs/privacy_sweep_e10.json --sigma 0.788 --method adaptive --seed 0

Interrupted runs resume automatically from their last checkpoint when the same command is run again. Each run uses one
process and one device; to run several cells in parallel, start several processes with different filters (one GPU
each, e.g. ``CUDA_VISIBLE_DEVICES=1 ... --device cuda:0``).

``--set key=value`` overrides a protocol entry (e.g. ``--set num_rounds=2 --set local_epochs=1`` for a smoke test).
Overridden runs are NOT the experiments of the paper; they are written under a separate ``<name>_override`` folder.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpsvdfl.federated.experiment import (  # noqa: E402
    RunResources,
    expand_runs,
    load_experiment_config,
    resolve_device,
    run_directory,
    run_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DP-SVD-FL experiments of the paper from a configuration file.")
    parser.add_argument("--config", type=Path, required=True, help="experiment configuration (configs/*.json)")
    parser.add_argument("--dataset", nargs="+", help="restrict to these datasets (default: the whole grid)")
    parser.add_argument("--partition", nargs="+", help="restrict to these partitions: iid, dirichlet")
    parser.add_argument("--seed", nargs="+", type=int, help="restrict to these experiment seeds")
    parser.add_argument("--method", nargs="+", help="restrict to these variants: adaptive (DP-SVD-FL) or fixed_spectral (see dpsvdfl.federated.methods)")
    parser.add_argument("--sigma", nargs="+", type=float, help="restrict to these noise multipliers (privacy sweep)")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data", help="dataset directory (default: data)")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results",
                        help="results directory (default: results)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N (default: auto)")
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="save a checkpoint every k rounds and after the last one (default: 5; private_release "
                             "requires 1)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker processes (default 0, as in the paper; a different value changes the "
                             "augmentation random stream)")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--allow-device-change", action="store_true",
                        help="allow resuming a run on another device type (no longer bitwise reproducible)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a protocol entry (smoke tests only)")
    parser.add_argument("--list", action="store_true", help="only list the selected runs and their directories")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        config = load_experiment_config(args.config)
        if args.overrides:
            config = {**config, "name": f"{config['name']}_override"}
        runs = expand_runs(config, datasets=args.dataset, partitions=args.partition, seeds=args.seed,
                           methods=args.method, noise_multipliers=args.sigma, overrides=args.overrides)
        device = None if args.list else resolve_device(args.device)
    except ValueError as error:
        sys.exit(f"error: {error}")
    sweep = bool(config["grid"].get("noise_multipliers"))
    directories = [run_directory(args.output_root, spec, sweep=sweep) for spec in runs]
    if args.list:
        for spec, directory in zip(runs, directories):
            print(f"{spec.dataset:16s} {spec.partition:10s} sigma={spec.config.dp.noise_multiplier:<6g} "
                  f"seed={spec.seed} {spec.method:20s} -> {directory}")
        print(f"{len(runs)} run(s)")
        return
    if args.checkpoint_every < 1:
        sys.exit("error: --checkpoint-every must be a positive integer")
    resources = RunResources(data_root=args.data_root, device=device, num_workers=args.num_workers,
                             eval_batch_size=args.eval_batch_size, checkpoint_every=args.checkpoint_every,
                             allow_device_change=args.allow_device_change, log=lambda line: print(line, flush=True))
    print(f"{len(runs)} run(s) selected; device {device}; torch {torch.__version__}", flush=True)
    for index, (spec, directory) in enumerate(zip(runs, directories), start=1):
        print(f"[{index}/{len(runs)}] {spec.dataset} {spec.partition} seed={spec.seed} {spec.method} "
              f"sigma={spec.config.dp.noise_multiplier:g} -> {directory}", flush=True)
        run_experiment(spec, directory, resources)


if __name__ == "__main__":
    main()
