# Reproducibility checklist

## Environment

- [ ] Linux with Python ≥ 3.10. The paper's runs used Python 3.12, PyTorch 2.13.0 + CUDA 12.9, torchvision 0.28.0,
      NumPy 2.5.3 and SciPy 1.18.1.
- [ ] `pip install -r requirements.txt && pip install -e .`
- [ ] `python -m pytest tests` passes. The equivalence tests in `tests/equivalence` are skipped without the reference
      implementation; this is expected.
- [ ] `python scripts/prepare_data.py --data-root data` downloads CIFAR-10, CIFAR-100, EMNIST Balanced and the
      `ResNet18_Weights.IMAGENET1K_V1` checkpoint.
- [ ] `DPSVDFL_DATA_ROOT=data python -m pytest tests/test_paper_inputs.py` passes. It checks that W(0), the client
      partitions and the Poisson schedules equal those recorded for the paper.

## Protocol (Table II)

Each item below is fixed in `configs/*.json` and checked by `tests/test_training_and_experiment.py`.

- [ ] N = 500, q = 0.1 (Poisson), T = 300, E = 10, minibatch 50 with `fixed_padded_last`.
- [ ] Plain SGD; learning rate 0.003 with per-round decay 0.998; η_s = 1.
- [ ] C = 0.2 (one joint clip per client), σ = 0.95 (main comparison), δ = 1/N = 0.002, ε ≈ 10.7948.
- [ ] τ = 2.0, ρ_max = 0.05, γ = 0.95.
- [ ] Seeds 0, 1, 2; IID and Dirichlet α = 0.6; evaluation every 10 rounds; result taken at round 300.
- [ ] `benchmark_reproducible` privacy mode. The reported ε is the nominal budget of the ideal mechanism, not an
      operational guarantee.

## Runs

| Result | Command (add `--data-root`, `--output-root`, `--device`) | Number of runs |
|---|---|---|
| Table III and Fig. 1 (DP-SVD-FL), Fig. 3, Table IV, S10 | `scripts/run_experiment.py --config configs/main_e10.json` | 18 |
| Fixed-rank ablation (Sec. VI-C) | `scripts/run_experiment.py --config configs/fixed_rank_e10.json` | 3 |
| Table V and Fig. 2 (DP-SVD-FL) | `scripts/run_experiment.py --config configs/privacy_sweep_e10.json` | 12 |

Only DP-SVD-FL and its fixed-rank ablation are in this repository; the comparison methods of the paper are not.

- [ ] Every run ends with `status.json` = complete, 300 consecutive records in `released_round_diagnostics.jsonl` and
      `final_model.pt`.
- [ ] Failed runs are reported, not dropped silently: tables and figures use a setting only when all three seeds are
      complete.

## Tables and figures

- [ ] `python analysis/make_tables.py ...` produces the DP-SVD-FL results of Tables III, IV and V and the fixed-rank
      result.
- [ ] `python analysis/make_figures.py ...` produces the DP-SVD-FL curves of Figs. 1–2 and Fig. 3.
- [ ] `python analysis/rank_audit.py --runs <main results> ...` produces S10.1–S10.4 and Table S2.
- [ ] `python analysis/noise_only_control.py --run --seeds 0-29 --rounds 300 ...` followed by
      `python analysis/noise_only_statistics.py ...` produces Table S3, S10.5 and Fig. G6.
- [ ] `tables_report.json` and `figures_report.json` list the runs used and the runs excluded.

## Expected agreement

- [ ] **Exact.** Data partitions, Poisson schedules, seeds, W(0), the rank rule on W(0) (D(1) = 596,622 / 642,792 /
      615,603) and ε are machine-independent and are reproduced exactly.
- [ ] **Up to run-to-run variability.** Training trajectories can differ on other GPUs, CPUs, CUDA/cuDNN/MKL/PyTorch
      versions or thread counts (CPU and CUDA noise generators, last-bit differences of float64 SVDs). Final accuracies
      are therefore expected to agree with the paper up to the seed-level variability of each table, not digit for
      digit.
- [ ] **Bitwise.** On a fixed machine and software stack, a run is deterministic, and a resumed run equals an
      uninterrupted one.
- [ ] **Noise-only control.** The data-free simulation reproduces the stored integers (ranks, branches, D) exactly;
      floating-point columns may differ by about 1e-13 across library versions.
