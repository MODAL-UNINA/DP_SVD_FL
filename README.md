# DP-SVD-FL: Adaptive Spectral Subspace Projection for Client-Level Differentially Private Federated Learning

This repository contains the code needed to reproduce the experiments and analyses of the paper

> Lingyu Qiu, Daniela Annunziata, Stefano Izzo, Fabio Giampaolo, Francesco Piccialli.
> *DP-SVD-FL: Adaptive Spectral Subspace Projection for Client-Level Differentially Private Federated Learning.*
> Manuscript submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence.

The paper has not been published yet. Citation details will be added on publication (see [Citation](#citation)).

## Method in brief

DP-SVD-FL is client-level differentially private federated learning with a trusted server. Each client update is
represented in the coordinate system of the leading singular vectors of the *previously released* global weights.

In every round t:

1. **Rank selection.** For each projected layer, the server computes the SVD of the released weight W_l^(t-1). It
   selects a rank k_l with a spectral-gap rule that falls back to an energy criterion. The parameters are tau = 2,
   rho_max = 0.05 and gamma = 0.95.
2. **Client sampling and local training.** Each of the N client slots is sampled independently with probability q
   (Poisson sampling). Every sampled client runs E epochs of local SGD from W^(t-1).
3. **Encoding.** Each client encodes its update as one vector s_i. Projected layers contribute the k x k coefficients
   U_k^T ΔW V_k; the classifier head is concatenated unprojected.
4. **Private aggregation.** The trusted server:
   - clips every s_i once, jointly, to norm C;
   - sums the clipped vectors;
   - adds Gaussian noise N(0, σ²C² I_D) once to the sum;
   - divides by the public constant qN.
5. **Server update.** The server reconstructs U_k Z V_k^T and applies one update with η_s = 1.

Bases and ranks depend only on public or already released quantities, so adapting them costs no extra privacy. The
privacy accountant uses Rényi DP for the Poisson-subsampled Gaussian mechanism. It depends only on (q, σ, T, δ); the
paper's setting (q, σ, T, δ) = (0.1, 0.95, 300, 0.002) gives ε ≈ 10.7948.

Main characteristics of the implementation:

* **Joint clipping.** One joint clip over all transmitted coordinates gives sensitivity C under add/remove adjacency.
* **Noise energy.** Noise is added once to the unnormalised sum, so its expected energy scales with the reduced
  dimension D instead of D_full (about 4% of D_full on average in the paper's runs).
* **Frozen BatchNorm.** Every BatchNorm parameter is frozen and every BatchNorm module stays in evaluation mode during
  local training.
* **Fixed-rank ablation.** The `fixed_spectral` variant keeps the ranks selected on W(0) for all rounds (Sec. VI-C).
* **Seeds.** Every random stream derives from an explicit experiment seed. At a fixed seed, the adaptive and fixed-rank
  variants share the data partition, the initialisation, the sampling schedule and the local-training random streams;
  each draws its own DP noise.

**Scope.** This repository contains only DP-SVD-FL and its fixed-rank ablation. The other methods the paper compares
against are not part of it: the tables and figures produced here contain the DP-SVD-FL results only.

## Repository structure

```
configs/                      experiment configurations (one per protocol reported in the paper)
  main_e10.json               DP-SVD-FL on 3 datasets x 2 partitions (Table III, Fig. 1, Table IV, Fig. 3)
  fixed_rank_e10.json         fixed-rank spectral ablation (Sec. VI-C)
  privacy_sweep_e10.json      CIFAR-10 privacy sweep of DP-SVD-FL (Table V, Fig. 2)
src/dpsvdfl/                  the library
  data/                       datasets, transforms, IID / Dirichlet client partitions
  models/                     ResNet-18 initialisation and parameter partitions
  spectral/                   rank rule, SVD bases, projection / reconstruction, reduced layout, strategies
  dp/                         joint clipping and Gaussian mechanism
  privacy/                    RDP accountant, privacy modes, OS-CSPRNG randomness (private_release)
  training/                   local SGD, minibatching, evaluation
  federated/                  seeds, Poisson sampling, one round, variants (adaptive / fixed-rank), experiment driver
scripts/
  prepare_data.py             download the datasets and the pretrained ResNet-18
  run_experiment.py           run / resume the runs of a configuration
  compute_privacy.py          nominal epsilon for given (q, sigma, T, delta)
analysis/
  make_tables.py              DP-SVD-FL results of Tables III, IV, V and the fixed-rank ablation
  make_figures.py             DP-SVD-FL curves of Figs. 1, 2 and Fig. 3
  rank_audit.py               Supplementary S10.1-S10.4 (rank-rule audit)
  noise_only_control.py       Supplementary S10.4 data-free noise-only replay (30 seeds)
  noise_only_statistics.py    Supplementary Table S3, S10.5 tests, Fig. G6
tests/                        CPU unit tests; tests/equivalence: release-time checks against the reference code
```

`LICENSE` is the MIT License of the repository. `PAPER_TO_CODE_MAP.md` maps every algorithm, experiment, table and figure of the paper to code.
`REPRODUCIBILITY_CHECKLIST.md` lists what is needed to reproduce each result, and `PUBLIC_RELEASE_REPORT.md`
documents how this repository was prepared and verified.

## Requirements

- Linux. The code was tested with Python 3.12, PyTorch 2.13 (CUDA 12.9), torchvision 0.28, NumPy 2.5 and SciPy 1.18.
  The analysis scripts also need pandas and matplotlib.
- A CUDA GPU for the full experiments. Every run uses one process and one device; CPU execution works but is only
  practical for tests.

Indicative wall-clock time of one 300-round run of the original experiments, on a single GPU shared with other runs:

| Method | CIFAR-10 / CIFAR-100 | EMNIST Balanced |
|---|---|---|
| DP-SVD-FL (adaptive or fixed-rank) | about 6–7 h | about 14 h |

The original launchers reserved about 3 GB of GPU memory per run.

## Installation

```bash
git clone <repository-url> dpsvdfl-paper
cd dpsvdfl-paper
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Install the PyTorch build that matches your CUDA version first if needed (see pytorch.org).

## Datasets and pretrained weights

```bash
python scripts/prepare_data.py --data-root data
```

This command:

- downloads CIFAR-10, CIFAR-100 and EMNIST (Balanced split) with torchvision into `data/`;
- downloads the `ResNet18_Weights.IMAGENET1K_V1` checkpoint into the torchvision cache (`$TORCH_HOME`, default
  `~/.cache/torch`).

The official training split of each dataset is partitioned over N = 500 client slots. The official test split is used
only for evaluation.

## Minimal example

A short smoke test of DP-SVD-FL on CIFAR-10: 2 rounds, 1 local epoch, about 5 sampled clients per round, and one
evaluation on the full test set. It took under 3 minutes on a 16-core CPU with 8 threads.

```bash
python scripts/run_experiment.py --config configs/main_e10.json --dataset cifar10 --partition dirichlet \
    --method adaptive --seed 0 --device cpu \
    --set num_rounds=2 --set local_epochs=1 --set sampling_rate=0.01 --set eval_every=2
```

Runs with `--set` overrides are written under `results/main_e10_override/` and are **not** the experiments of the
paper.

The unit tests run on CPU in a few minutes:

```bash
python -m pytest tests
```

## Reproducing the experiments of the paper

Every result is identified by (configuration, dataset, partition, seed, method[, sigma]). To list the runs of a
configuration:

```bash
python scripts/run_experiment.py --config configs/main_e10.json --list
```

| Configuration | Runs | Paper |
|---|---|---|
| `main_e10.json` | 3 datasets × {Dirichlet α = 0.6, IID} × seeds {0,1,2} = 18 DP-SVD-FL runs | Table III and Fig. 1 (DP-SVD-FL), Table IV, Fig. 3, Suppl. S10 |
| `fixed_rank_e10.json` | CIFAR-10 Dirichlet × 3 seeds, fixed-rank spectral ablation = 3 | Sec. VI-C |
| `privacy_sweep_e10.json` | CIFAR-10 Dirichlet × σ ∈ {0.788, 0.95, 1.11, 1.687} × 3 seeds = 12 DP-SVD-FL runs | Table V and Fig. 2 (DP-SVD-FL) |

Shared protocol (Table II of the paper):

- N = 500 slots, Poisson sampling with q = 0.1, T = 300 rounds;
- E = 10 local epochs, minibatch 50 with the `fixed_padded_last` policy, plain SGD;
- learning rate 0.003 with a per-round decay of 0.998, server learning rate 1;
- C = 0.2 (joint clipping), σ = 0.95, δ = 1/N = 0.002;
- ResNet-18 pretrained on ImageNet-1K: layer3/layer4 convolutions projected, classifier head unprojected, everything
  else frozen;
- evaluation every 10 rounds.

To run one cell on one GPU:

```bash
python scripts/run_experiment.py --config configs/main_e10.json --dataset cifar10 --partition dirichlet \
    --seed 0 --method adaptive --data-root data --output-root results --device cuda:0
```

Further options:

- **Variants.** `--method adaptive` is DP-SVD-FL; `--method fixed_spectral` is the fixed-rank ablation (only in
  `fixed_rank_e10.json`).
- **Filters.** Omitting a filter runs all the matching runs one after the other. Several processes, for example one
  per GPU, can run different filters in parallel; each run writes only into its own directory.
- **Resume.** An interrupted run resumes from its last checkpoint when the same command is run again
  (`--checkpoint-every`, default 5 rounds).
- **Privacy sweep.** Select the noise level with `--sigma`, e.g.
  `--config configs/privacy_sweep_e10.json --sigma 0.788`. Each σ level is a separate release with its own DP-noise
  stream, so the σ = 0.95 level is a different run from the main campaign.

## Tables and figures

```bash
python analysis/make_tables.py  --main results/main_e10 --privacy-sweep results/privacy_sweep_e10 \
    --fixed-rank results/fixed_rank_e10 --out paper_outputs/tables
python analysis/make_figures.py --main results/main_e10 --privacy-sweep results/privacy_sweep_e10 \
    --out paper_outputs/figures
```

All statistics follow the paper's reporting rules:

- accuracy is taken at round 300, never at the best checkpoint;
- tables report the mean ± *sample* standard deviation over seeds 0, 1, 2;
- campaigns and noise levels are never pooled.

The scripts report only DP-SVD-FL and the fixed-rank ablation; runs of any other method found under the given roots are
listed as ignored.

`tables_report.json` and `figures_report.json` list the runs used and the runs excluded, with the reason.

Supplementary Section S10 (rank-rule audit and noise-only control):

```bash
python analysis/rank_audit.py --runs results/main_e10 --out paper_outputs/rank_audit [--final-models]
python analysis/noise_only_control.py --self-test --out paper_outputs/noise_only/self_test
python analysis/noise_only_control.py --run --seeds 0-29 --rounds 300 --out paper_outputs/noise_only/simulation
python analysis/noise_only_statistics.py --runs results/main_e10 --sim paper_outputs/noise_only/simulation \
    --out paper_outputs/noise_only/statistics
```

The noise-only control is data-free: it needs only the public pretrained weights and takes a few minutes on CPU.

## Organisation of the results

```
results/<configuration>/<dataset>_<partition>[_sigma<1000σ>]/seed<k>/<adaptive|fixed_spectral>/
    run_config.json                       public configuration, fingerprints of W(0), partition and schedule
    released_round_diagnostics.jsonl      per round: D, D_full, ranks, rank-rule branches, gap ratios, lr,
                                          test accuracy every 10 rounds
    benchmark_diagnostics_internal.jsonl  benchmark-only diagnostics (clipping counts, norms): NOT a DP release
    checkpoints/                          latest checkpoint (resume)
    final_model.pt, public_summary.json, status.json
```

## Privacy notes

- **Benchmark mode.** All results of the paper were produced in `benchmark_reproducible` mode, the default of every
  configuration. Client sampling and DP noise derive from the public experiment seeds, so the runs can be replayed
  exactly. Anyone who knows the seeds can regenerate the sampled sets and subtract the noise. The reported ε is
  therefore the **nominal budget of the ideal randomized mechanism**. These runs are **not** an operationally private
  release.
- **Private-release mode.** `private_release` draws client sampling and Gaussian noise from the operating-system CSPRNG.
  This randomness is never seeded or stored. Every round attempt is logged before its randomness is drawn, and the
  final ε composes all recorded attempts. To use it, set `"privacy_mode": "private_release"` in a configuration and run
  with `--checkpoint-every 1`. The paper makes no claim about this mode. Floating-point and timing side channels are
  not covered.
- **Scope of the guarantee.** It concerns the noisy aggregates and the released models in the trusted-server model,
  under add/remove adjacency of one client's whole dataset. It does not cover the server's view of individual
  updates, the sampled sets, the benchmark diagnostics, or several experiments on overlapping client populations,
  which would require composition.
- **Choice of δ.** δ = 1/N = 0.002 follows the paper. A different δ requires recomputing ε
  (`scripts/compute_privacy.py --delta ...`).

## Reproducibility limitations

- **Same machine.** On the same hardware, software stack and thread count, a run is deterministic, and a resumed run
  matches an uninterrupted one.
- **Across hardware or software.** Bitwise reproducibility is **not** guaranteed across GPUs, CPU types, CUDA/cuDNN,
  MKL or PyTorch versions, or thread counts:
  - CPU and CUDA generators give different noise realisations for the same seed;
  - float64 SVDs from different LAPACK/MKL builds, or with different memory layouts or thread counts, can differ in
    the last bits;
  - these differences then propagate through training.
- **Expected agreement.** Re-running the configurations should reproduce the paper's numbers up to this run-to-run
  variability, not digit for digit.
- **What is machine-independent.** Data partitions, sampling schedules, seeds and W(0) are deterministic and
  machine-independent: `tests/test_paper_inputs.py` checks them against the fingerprints recorded for the paper.
- **Data loading.** `--num-workers` other than 0 changes the augmentation random stream; the paper used 0.

## Citation

The manuscript is under review. Until it is published, please cite it as:

```
L. Qiu, D. Annunziata, S. Izzo, F. Giampaolo, F. Piccialli,
"DP-SVD-FL: Adaptive Spectral Subspace Projection for Client-Level Differentially Private Federated Learning,"
manuscript submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence.
```

See also `CITATION.cff`.

## License

The code is released under the MIT License; see [`LICENSE`](LICENSE).
Copyright (c) 2026 Università degli Studi di Napoli Federico II.

The license covers the code and documentation of this repository only. The datasets (CIFAR-10, CIFAR-100, EMNIST) and
the pretrained `ResNet18_Weights.IMAGENET1K_V1` checkpoint are not distributed here; they are downloaded from their
original sources and remain subject to their own terms. If you use this code in academic work, please cite the paper
(see [Citation](#citation) and `CITATION.cff`).
