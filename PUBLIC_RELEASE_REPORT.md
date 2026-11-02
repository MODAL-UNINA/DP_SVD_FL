# Public release report

**Status: READY_FOR_REVIEW.** The repository is complete, every planned test passes and the license has been applied
(MIT, section K). It contains DP-SVD-FL and its fixed-rank ablation only (scope below). The remaining items of section L
are author review and publication details, not open technical issues.

## Scope

This report documents how this repository was prepared from the internal research repository that produced the results
of the manuscript *DP-SVD-FL: Adaptive Spectral Subspace Projection for Client-Level Differentially Private Federated
Learning* (IEEE TPAMI submission).

The selection follows the current manuscript, `main.tex` and `supplementary.tex` (latest versions supplied by the
authors), and nothing else. The research repository was only read; nothing in it was modified. No training campaign
was run.

**Scope decision (authors).** The repository contains only the authors' method, DP-SVD-FL, and its fixed-rank ablation
(Sec. VI-C). The comparison methods of the paper were removed at the authors' request, after a first version of this
repository had included them: their code, configurations, analysis outputs and tests are not part of it. Consequently,
only the DP-SVD-FL entries of Table III, Table V and Figs. 1–2 can be regenerated; Table IV, Fig. 3, the fixed-rank
ablation and Supplementary S10 are complete.

## A. Included files and provenance

The paths of the original files are relative to the research repository. `ORIGINAL_TO_PUBLIC_FILES.csv` gives one row
per original source module or script (191 rows), with its public counterpart and its status: copied, pruned, merged,
replaced, ported or excluded.

### Library (`src/dpsvdfl`, 33 Python files including package `__init__` files, about 4,450 lines)

- **Copied, docstrings only updated.** The numerical code is unchanged in:
  - `spectral/rank_rule.py`, `spectral/matricization.py`;
  - `privacy/accountant.py`, `dp/mechanism.py`;
  - `training/evaluation.py`, `reproducibility.py`, `federated/sampling.py`;
  - `data/partitioning.py` (from `federated/partitioning.py`), `data/datasets.py`.
- **Pruned.** Branches for every method other than DP-SVD-FL and its fixed-rank ablation were removed:
  - `federated/seeds.py`, `federated/runner.py`;
  - `training/local.py`, `training/batching.py`;
  - `spectral/projection.py`, `spectral/representation.py`, `spectral/strategies.py`;
  - `models/resnet.py`, `models/partitions.py`;
  - `privacy/modes.py`, `privacy/randomness.py`, `federated/server_update.py` (from `server/update.py`), `data/transforms.py`.
- **New, replacing the orchestration layer:**
  - `federated/methods.py` replaces the method registries (DP-SVD-FL and the fixed-rank ablation);
  - `federated/experiment.py` replaces campaign preparation, execution, checkpointing and the suite layer;
  - `federated/io.py` keeps the atomic writes and the fingerprints.

### Scripts (`scripts/`, 3 files)

- `run_experiment.py`, which replaces the campaign launchers;
- `prepare_data.py`;
- `compute_privacy.py`.

### Analysis (`analysis/`, 6 files)

- `make_tables.py`, `make_figures.py` and `common.py`, ported from `outputs/paper_tpami_e10/aggregate_results.py`,
  `recompute_tables.py` and `figures/make_tpami_figures.py`;
- `rank_audit.py`, ported from `rank_rule_audit_e10/scripts/{audit_rank_rule,recompute_endpoints,noise_share}.py`;
- `noise_only_control.py` and `noise_only_statistics.py`, ported from
  `outputs/rank_rule_noise_only_e10/scripts/{noise_only_rank_replay,noise_only_30_seeds,verify_state_30_seeds,analyse_30_seeds}.py`
  and `G6_EN_verification/make_G6_projected_dimension_EN.py`.

### Configurations (`configs/`)

- `main_e10.json` (18 DP-SVD-FL runs), `fixed_rank_e10.json` (3) and `privacy_sweep_e10.json` (12), rebuilt from the
  `adaptive` and `fixed_spectral` arms of the E3, E6-A2 and E9 campaign manifests.

### Tests (`tests/`, 10 Python files)

- unit tests;
- real-data input checks;
- release-time equivalence tests against the research code.

### Documentation

- `README.md`, `PAPER_TO_CODE_MAP.md`, `REPRODUCIBILITY_CHECKLIST.md`, this report and `ORIGINAL_TO_PUBLIC_FILES.csv`;
- `CITATION.cff`, `pyproject.toml`, `requirements.txt` and `.gitignore`.

## B. Excluded files and reasons

**Comparison methods of the paper** (removed at the authors' request): the baselines in `baselines/`, their local
objectives, partitions, configurations, tables, figure curves and tests.

**Methods and experiments that are not in the current manuscript:**

- the Local DP extension (`experiments/local_dp_extension/`);
- DP-FedPGN, DP-SCAFFOLD and DP-LoRA (`baselines/dp_fedpgn.py`, `dp_scaffold.py`, `scaffold_*`, `dp_lora.py`,
  `models/lora.py`);
- Fed-SMP rand-k (`baselines/fed_smp_randk.py`), which is cited only in Related Work;
- the LIN comparisons (`experiments/lin_comparison/`, `LIN/`, `analysis/`);
- the two-phase / hybrid experiments (`experiments/two_phase_dp/`);
- the SG-3 campaigns (`subspace_geometry/`);
- the Haar ablations (`spectral/random_basis.py`, `fixed_haar`);
- ViT and ResNet-34 (`models/vit.py`, `models/logical_blocks.py`, `models/architectures.py`);
- the E2 random initialisation;
- the E4–E10 diagnostics, pilots and sweeps (`diagnostics/`, `federated/{tau_sensitivity,rank_sensitivity,basis_privacy_sweep,lr_stability_pilot,initialization_pair,e3_e9_v2}.py`);
- the legacy v1 accountant.

**Infrastructure:**

- machine-specific launchers, schedulers, dashboards and progress reporting (`run_e_esecuzioni/`,
  `utils/{gpu_scheduler,live_status,progress,threads}.py`);
- the cluster job scripts;
- the execution contract and the integrity tooling of stored campaigns;
- GPU augmentation (`data/gpu_augmentation.py`), which no reported run used.

**Data and outputs:**

- datasets, pretrained weights, logs, checkpoints, archives and every experiment output;
- internal reports and notes (`docs/`, `scope/`, the Italian reports).

**Exploratory analyses not used in the paper:**

- `tail_invariance.py` and `audit_figures.py`;
- the extra tests, G1–G5 and G7–G9 of `analyse_30_seeds.py`;
- the superseded `compare_noise_only_vs_e3.py`.

## C. Mapping of the paper's sections

See `PAPER_TO_CODE_MAP.md`. Every algorithmic step of Algorithm 1, Mechanism 1, the accountant, the release model,
each DP-SVD-FL experiment, the DP-SVD-FL entries of Tables II–V and S2–S3, Figs. 1–3 and G6, and the S10 statistics
map to identified code.

The final layout of Fig. G6 had no source script in the research repository; its labels were taken from its released v2
archive, and it is now generated by `analysis/noise_only_statistics.py`.

## D. Clean-up

- **Size.** The import closure of the original campaign runner was about 20,300 lines. The public library has about
  4,450 lines, and the whole repository about 8,300 lines of Python including analysis and tests.
- **Removed:**
  - the comparison methods, their local objectives (proximal term, SAM, gradient-norm penalty) and the
    full-dimensional (P = empty) representation;
  - dead parameters and unused hooks: PGN terms, SCAFFOLD control terms, coordinate selection, released-update
    post-processing, step observers, E4 pre-round hooks, the legacy batching policy and legacy accountant records;
  - logical-block dispatch;
  - progress and dashboard plumbing.
- **No hard-coded paths.** Data and output roots and the device are CLI options. The pretrained weights come from the
  torchvision cache (`$TORCH_HOME`).
- **No infrastructure references.** There are no machine names, IP addresses, user names or absolute paths in any
  file (section K).

## E. Simplifications

- **Inputs.** The original prepared-campaign workflow stored the partition, schedule and manifests and then ran each
  run. The public driver regenerates the partition, the Poisson schedule and W(0) from the experiment seed for each
  run, because they are pure functions of the seed. It records their SHA-256 fingerprints in `run_config.json` and
  checks them on resume. `tests/test_paper_inputs.py` shows that the regenerated inputs equal the recorded ones for all
  18 (dataset, partition, seed) cells.
- **One round function.** One `run_round` serves both variants; only the round state differs. The adaptive variant
  builds its round state through `spectral.strategies.build_round_state`, which calls the same
  `compute_projection_basis` as the original `build_round_spectral_state`.
- **Checkpoints.** A simple latest-checkpoint scheme with log truncation on resume replaces the multi-file identity
  and checkpoint machinery. The private_release write-ahead attempt ledger and its accounting are kept.
- **Default values.** `LocalTrainingConfig` defaults are now the paper's values (E = 10, lr = 0.003). The research code
  defaulted to an earlier E = 30, lr = 0.1 setting, which the configurations always overrode.

**Equivalence.** Every simplification that touches a numerical path was checked for exact equality against the
research code (section H). None needed to be reverted.

## F. Comments and documentation

- Module docstrings were rewritten in English with explicit references to the manuscript: section, Algorithm 1 line,
  Mechanism 1, Lemma 1, Theorems 1–2, Remarks, and Supplementary sections S1–S11. They state tensor shapes,
  normalisation conventions (noise on the unnormalised sum, the public qN), the rank-rule branches, the clipping order
  and the meaning of the two privacy modes.
- References to internal checkpoints, campaign codes, defect identifiers and the earlier methodology document were
  removed from the code. Internal campaign names (E3, E9, E6-A2) appear only in the provenance columns of
  `PAPER_TO_CODE_MAP.md`.
- **Italian text.** A global search found no Italian text. The original analysis scripts and reports contained
  Italian; the ported versions are in English.

## G. Tests run

All tests were run on CPU (`CUDA_VISIBLE_DEVICES=`). No GPU was used and no full campaign was started.

**Full suite.** 44 tests, 44 passed (6 min 39 s), after the comparison methods were removed:

- unit tests (31):
  - seeds, sampling and partitions;
  - rank rule (gap, ρ_max rejection, energy minimality, zero-spectrum fallback), orthonormality, isometry, idempotent
    projector, encode/decode layout and dimension;
  - joint clipping, normalisation by qN, noise standard deviation and seed dependence;
  - accountant values;
  - private_release record and plan;
  - local training: frozen BatchNorm, empty slot, determinism;
  - DP-SVD-FL and the fixed-rank ablation end-to-end (2 rounds);
  - bitwise resume, refusal of a different specification, private_release run;
  - configuration grids, CLIs;
- real-data input fingerprints (3);
- equivalence with the research code (10).

**Public scenario.** Without the reference code and without datasets: 31 passed, 13 skipped, as expected.

**Other checks:**

- syntax check (`py_compile`) of every file;
- `--help` of every CLI;
- `--list` of every configuration (18, 3 and 12 runs).

**Real smoke test.** One actual run of the minimal README example: CIFAR-10 Dirichlet, DP-SVD-FL, seed 0, 2 rounds,
1 local epoch, q = 0.01, CPU. After the comparison methods were removed it completed in 2 min 45 s with
D(1) = 596,622 (5.69%), the value of the paper.

**Analysis scripts.** Run on the original result files of the paper (section H).

## H. Comparison with the original code

**Equivalence tests** (`tests/equivalence`; reference and public code in separate subprocesses on the same machine,
CPU, `MKL_CBWR=COMPATIBLE`, 4 threads). Everything is **bitwise identical**:

- **Seed plans.** Both noise namespaces × 5 contexts × 4 seeds.
- **Partitions and schedules.** IID and Dirichlet partitions, 300-round Poisson schedules.
- **Rank rule and SVD bases.** On random matrices, special cases and the 10 projected layers of W(0).
- **Gaussian mechanism.** Including an empty round.
- **Accountant.** Seven (q, σ, T) settings, including the paper's four σ values.
- **End-to-end runs.** Five tiny federated runs (8 slots, 2 rounds, evaluation every round). They cover DP-SVD-FL and
  the fixed-rank ablation, IID and Dirichlet partitions, and two privacy-sweep noise contexts. In each run the final parameters and buffers, the
  released round records and the benchmark round records are identical.

**Real-data inputs.** On the real CIFAR-10, CIFAR-100 and EMNIST Balanced data, the W(0), partition and schedule
fingerprints of all 18 (dataset, partition, seed) cells equal those recorded when the paper's campaigns were prepared.
These inputs are shared by the main, fixed-rank and privacy-sweep campaigns.

**Accountant values.** ε = 16.0183862528, 10.7947905408, 8.0038933869 and 4.0017467678 for σ = 0.788, 0.95, 1.11 and
1.687, equal to the recorded values.

**Paper numbers recomputed from the original result files with the public analysis scripts:**

- Table III, DP-SVD-FL column: 6/6 cells (e.g. CIFAR-10 Dir. 76.09 ± 0.31, EMNIST Bal. IID 80.66 ± 0.16).
- Table IV and the D fractions (5.69% → 2.91%, 5.86% → 3.08%).
- Table V, DP-SVD-FL: 80.46 ± 0.44, 75.94 ± 0.39, 71.05 ± 1.46 and 52.37 ± 0.54.
- Fixed-rank ablation: 73.70 ± 0.34.
- Figs. 1–3 (DP-SVD-FL): the curve data equal the archived values within 1e-9.
- After the comparison methods were removed, the reduced scripts were run again on the same result files: every
  DP-SVD-FL value of Tables III–V, the fixed-rank table and the curve data of Figs. 1–3 are identical to those of the
  previous, verified version.
- Cross-check: 1,629 full-precision checks against the archived aggregates.
- S10.1–S10.4:
  - 14/14 checks in every run; round 1 recomputed from W(0) in all 18 runs;
  - 5,400 gap decisions, gap ratios 2.62–2.73, k95 = 413;
  - 587–595 rank-change events per run; 73.1% of the decrease of D from three layer4 blocks;
  - noise ratios 1.0000–1.0001 and 0.077–0.082; noise share within 1.49%.
- S10.4 noise-only control (Table S3) and S10.5:
  - D decreases 48.752/48.765, 45.200/45.262 and 47.305/47.261;
  - mean-curve MAE 0.043–0.096%;
  - p = 0.159 and 0.045, Holm-adjusted 0.159 and 0.090;
  - these values are bitwise equal to the archived statistics.
- Noise-only simulation. Regenerated for all 30 seeds × 300 rounds, its integers (ranks, branches, D) are identical to
  the archived simulation; the floating-point columns differ by at most 5e-14 relative (newer PyTorch/MKL).

## I. Unresolved differences

1. **MKL last-bit sensitivity.** Without `MKL_CBWR=COMPATIBLE`, the float64 CPU SVD of Intel MKL can differ in the last
   bit depending on the memory-allocation history of the process. The *same* public code then produced runs that
   differed by about 3e-8 after the first SVD of an updated weight, so this is a property of the numerical library,
   not of either implementation. With conditional bitwise reproducibility, the public and research implementations are
   identical. The equivalence tests set it explicitly. It is listed under the reproducibility limitations.
2. **Manuscript Table S2.** The EMNIST lower bound of D(300)/D(1) is 323,497/615,603 = 0.5255, which rounds to 0.525.
   Table S2 prints 0.526–0.528. The manuscript was not modified.
3. **Manuscript titles.** The Supplementary Material carries the title "Low-Rank Projection for Differentially Private
   Federated Learning via Spectral Subspaces of Pretrained Models", which differs from the main manuscript's title. The
   manuscript was not modified.
4. **Comparison results not regenerated.** By the authors' choice, the comparison columns of Table III, the paired
   differences, the comparison points of Table V and Figs. 1–2, and Table S4 cannot be regenerated from this repository.

## J. Reproducibility limitations

- **Across hardware or software.** Bitwise reproducibility is not promised across GPUs, CPUs, CUDA/cuDNN/MKL/PyTorch
  versions or thread counts:
  - CPU and CUDA noise generators differ for the same seed;
  - float64 SVDs can differ in the last bits;
  - these differences propagate through training.
- **What is exact.** Partitions, schedules, seeds, W(0), D(1) and ε are machine-independent and are reproduced exactly.
- **Original hardware.** The paper's runs executed on several CUDA machines; the hardware is recorded per run in the
  research logs but is not needed to rerun the code. Re-running should reproduce the reported numbers up to the
  seed-level variability.
- **Incomplete settings.** A table cell or a figure point is reported only when all three seeds are complete.
- **DataLoader workers.** `--num-workers` other than 0 changes the augmentation stream; the paper used 0.

## K. Security checks

A recursive search of all files of the repository (65 files, including this report) found:

- no SSH keys, tokens, passwords, credentials or `.env` files;
- no IP addresses;
- no absolute or personal paths, user names, host names or cluster configuration;
- no dataset, checkpoint, weight file or archive. The largest file is under 200 KB and the whole repository is
  684 KB.

`.gitignore` excludes `data/`, `results/`, `paper_outputs/`, `*.pt`, `*.pth`, `*.npz`, archives, logs, caches,
virtual environments and `.env*`.

### License check

The authors chose the MIT License. Before `LICENSE` was added, the following was checked:

- **Pre-existing notices.** Neither this repository nor the files of the research repository it was built from
  contain copyright headers, SPDX identifiers or `LICENSE`, `NOTICE`, `COPYING` or `AUTHORS` files, so no notice had
  to be preserved.
- **Third-party code.** No file used for this repository carries a third-party license or attribution notice. The
  RDP accountant is an independent implementation (its own certified numerical integration) of results from the
  literature (Mironov, Talwar and Zhang 2019; Balle et al. 2020), not a copy of Opacus or TensorFlow Privacy.
  ResNet-18 is obtained through
  the torchvision API and its code is not copied.
- **Dependencies.** NumPy, SciPy and pandas (BSD-3-Clause), PyTorch and torchvision (BSD-3-Clause), Matplotlib
  (Matplotlib License, permissive) and pytest (MIT) are compatible with MIT. None of them is redistributed here.
- **Data and weights.** The datasets and the pretrained ImageNet checkpoint are not distributed; they are downloaded
  by the user and remain under their own terms (stated in `README.md`).
- **Copyright holder.** No holder could be verified from the files. The authors stated the holder: Università degli
  Studi di Napoli Federico II; the year, 2026, matches the history of the research repository.

Files changed: `LICENSE` added (official MIT text); `README.md` (License section and repository structure);
`pyproject.toml` (`license = "MIT"`, `license-files`, and `setuptools>=77`, the first version that accepts an SPDX
license expression); `CITATION.cff` (`license: MIT`; the citation metadata is unchanged). No code was changed.

## L. Open items before publication

1. **Manuscript.** Consider correcting Table S2 (0.525) and the Supplementary title (items 2–3 in section I).
2. **Citation.** Add the DOI and publication details to `CITATION.cff` and `README.md` after acceptance.
3. **Author review.** Review the whole repository, including the author list and order, before creating the GitHub
   repository. No remote was created and nothing was pushed or published.
