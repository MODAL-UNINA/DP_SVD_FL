# Paper-to-code map

The map refers to the current manuscript (main text and Supplementary Material). Section numbers follow the compiled
manuscript: III = Method, IV = Theory, V = Setup, VI = Results, S1–S11 = Supplementary sections.

This repository contains DP-SVD-FL and its fixed-rank ablation only; the methods the paper compares against are not
included, and only the DP-SVD-FL entries of the tables and figures are mapped below.

**Original files** are paths in the internal research repository that produced the paper's results; that repository
is not published. **Public files** are paths in this repository. Every public file was checked against the original
implementation; the checks are listed in `PUBLIC_RELEASE_REPORT.md`, section H.

## 1. Algorithm and mechanism

### 1.1 Data, model and partitions

- **Initial model W(0)**
  - Paper: V-A; S6, S6.3.
  - Original files: `models/resnet.py`.
  - Functions: `load_pretrained_resnet18`, `replace_classifier`.
  - Dependencies: torchvision weights `ResNet18_Weights.IMAGENET1K_V1`; seeds `classifier_seed`.
  - Public file: `src/dpsvdfl/models/resnet.py`.
  - Why included: the pretrained backbone plus a classifier initialised from the seed; the same for both variants of a
    seed.
- **Trainable, projected and frozen blocks; frozen BatchNorm**
  - Paper: III-A; S6.3.
  - Original files: `models/partitions.py`.
  - Functions: `resnet18_partition`, `apply_partition`, `set_batchnorm_eval`.
  - Dependencies: `spectral/matricization.py`.
  - Public file: `src/dpsvdfl/models/partitions.py`.
  - Why included: P is the 10 layer3/layer4 convolution weights; B \ P is fc. BatchNorm stays in evaluation mode.
- **Datasets and preprocessing**
  - Paper: V-A; S6.1, Table S1.
  - Original files: `data/datasets.py`, `data/transforms.py`.
  - Functions: `load_benchmark`, `train_transform`, `eval_transform`.
  - Dependencies: torchvision.
  - Public files: `src/dpsvdfl/data/`; `scripts/prepare_data.py`.
  - Why included: CIFAR-10, CIFAR-100 and EMNIST Balanced at 224×224 with ImageNet normalisation.
- **Client-slot partition (IID, Dirichlet α = 0.6)**
  - Paper: V-A; S6.2.
  - Original files: `federated/partitioning.py`.
  - Functions: `iid_partition`, `dirichlet_partition`, `make_partition`.
  - Dependencies: `reproducibility.numpy_generator`.
  - Public file: `src/dpsvdfl/data/partitioning.py`.
  - Why included: N = 500 slots; empty slots are kept.
- **Seed plan**
  - Paper: V-D; S9.
  - Original files: `federated/seeds.py`, `utils/reproducibility.py`.
  - Functions: `RunSeedPlan`, `contextual_noise_namespace`, `privacy_sweep_noise_context`, `derive_indexed_seed`.
  - Dependencies: NumPy `SeedSequence`, SHA-256.
  - Public files: `src/dpsvdfl/federated/seeds.py`, `src/dpsvdfl/reproducibility.py`.
  - Why included: the common streams (classifier, partition, sampling, local training) are shared by the adaptive and
    fixed-rank variants; each variant has its own DP-noise namespace.

### 1.2 Client sampling and local training (Algorithm 1)

- **Client selection (line 9)**
  - Paper: III-A; S7.2.
  - Original files: `federated/sampling.py`.
  - Functions: `generate_sampling_schedule`.
  - Dependencies: seed plan.
  - Public file: `src/dpsvdfl/federated/sampling.py`.
  - Why included: Bernoulli(q) per slot, not a fixed-size selection.
- **Local training (lines 13–14)**
  - Paper: III-C; S7.1.
  - Original files: `training/local.py`, `training/batching.py`.
  - Functions: `train_client`, `run_local_sgd`, `make_client_dataloader`, `PaddedLastBatchSampler`,
    `round_learning_rate`.
  - Dependencies: partitions; seed plan.
  - Public files: `src/dpsvdfl/training/local.py`, `src/dpsvdfl/training/batching.py`.
  - Why included: E = 10, B = 50, `fixed_padded_last`, plain SGD, lr 0.003·0.998^(t−1), fresh optimizer, an exact
    zero update for an empty slot.

### 1.3 Spectral representation (Algorithm 1, lines 3–8, 15 and 24)

- **SVD of the released weights and rank rule (lines 3–6)**
  - Paper: III-B, Eq. "rank rule"; Definition 1.
  - Original files: `spectral/rank_rule.py`, `spectral/projection.py`.
  - Functions: `select_rank`, `largest_gap`, `energy_rank`, `compute_projection_basis`, `reduced_svd`.
  - Dependencies: float64 `torch.linalg.svd`.
  - Public files: `src/dpsvdfl/spectral/rank_rule.py`, `src/dpsvdfl/spectral/projection.py`.
  - Why included: the gap branch (τ = 2, ρ_max = 0.05) with the smallest-maximiser tie break, the energy fallback
    (γ = 0.95), and k = 1 for a zero spectrum.
- **Basis update every round**
  - Paper: III-B; Algorithm 1 line 4.
  - Original files: `federated/runner.py` (`run_dpsvdfl_round`); `spectral/strategies.py`.
  - Functions: `build_round_state`, `adaptive_spectral_strategy`.
  - Dependencies: `spectral/representation.py`.
  - Public file: `src/dpsvdfl/spectral/strategies.py`.
  - Why included: the bases and ranks are computed once per round from W^(t−1).
- **Projection P_l and reconstruction R_l**
  - Paper: III-B; Lemma 1; S1.
  - Original files: `spectral/projection.py`.
  - Functions: `project_update`, `reconstruct_update`.
  - Public file: `src/dpsvdfl/spectral/projection.py`.
  - Why included: U_kᵀ X V_k and U_k S V_kᵀ.
- **Encode, reduced dimension D(t) and layout (lines 8, 15)**
  - Paper: III-C.
  - Original files: `spectral/representation.py`.
  - Functions: `build_reduced_layout`, `encode_client_update`.
  - Dependencies: matricisation.
  - Public file: `src/dpsvdfl/spectral/representation.py`.
  - Why included: s_i is the projected coefficients followed by the unprojected blocks, with
    D = Σ k_l² + Σ m_l n_l.
- **Decode and server update (lines 24–26)**
  - Paper: III-C; Mechanism 1 step 5.
  - Original files: `server/update.py`.
  - Functions: `reconstruct_server_update`, `apply_server_update`.
  - Public file: `src/dpsvdfl/federated/server_update.py`.
  - Why included: the same round bases are used; η_s = 1; frozen blocks are untouched.

### 1.4 Clipping, noise and aggregation (Algorithm 1, lines 19–23)

- **Joint clipping (line 19)**
  - Paper: III-C; Theorem 1; Remark 2(a).
  - Original files: `dp/mechanism.py`.
  - Functions: `joint_clip_reduced_update`.
  - Public file: `src/dpsvdfl/dp/mechanism.py`.
  - Why included: one float64 norm over all blocks; clip(0) = 0.
- **Gaussian noise, normalisation and aggregation (lines 20–23)**
  - Paper: III-C; Mechanism 1; Proposition 1.
  - Original files: `dp/mechanism.py`.
  - Functions: `apply_reduced_dp_mechanism`, `sample_gaussian_noise`.
  - Dependencies: seed plan, or `privacy/randomness.py` in private_release.
  - Public file: `src/dpsvdfl/dp/mechanism.py`.
  - Why included: N(0, σ²C²I_D) is added once to the sum, which is then divided once by the public qN.
- **One round and the T-round loop**
  - Paper: Algorithm 1.
  - Original files: `federated/runner.py`.
  - Functions: `run_round`, `run_federated_rounds`.
  - Dependencies: all of the above.
  - Public file: `src/dpsvdfl/federated/runner.py`.
  - Why included: the order of operations is local training → Encode → clip → sum → noise → /qN → Decode → update.

### 1.5 Privacy accounting

- **Privacy accountant**
  - Paper: IV-C, IV-D; Theorem 2; S8, S8.1.
  - Original files: `privacy/accountant.py`.
  - Functions: `compute_privacy_accounting`, `default_rdp_orders`, `rdp_to_epsilon`.
  - Dependencies: NumPy, SciPy.
  - Public files: `src/dpsvdfl/privacy/accountant.py`; `scripts/compute_privacy.py`.
  - Why included: `dpsvdfl-sgm-rdp-certified-v2`, the 221-order grid and the Balle et al. conversion:
    ε(0.1, 0.95, 300, 0.002) = 10.7948.
- **Release model: benchmark_reproducible versus private_release**
  - Paper: V-E; S8.2.
  - Original files: `privacy/modes.py`, `privacy/randomness.py`, `federated/campaign.py` (`RoundAttemptLedger`).
  - Functions: `PrivacyConfiguration`, `public_privacy_record`, `PrivateReleaseSeedPlan`, `OSRandomness`,
    `RoundAttemptLedger`.
  - Dependencies: `os.urandom`.
  - Public files: `src/dpsvdfl/privacy/modes.py`, `src/dpsvdfl/privacy/randomness.py`,
    `src/dpsvdfl/federated/experiment.py`.
  - Why included: every reported result used benchmark_reproducible, which gives a nominal ε. private_release is
    provided as stated in the paper; no claim is made about its use.

### 1.6 Execution, checkpoints and evaluation

- **Checkpoint and resume**
  - Paper: S9 (run records).
  - Original files: `federated/campaign.py`, `federated/checkpointing.py`.
  - Functions: `run_experiment`, `_save_checkpoint`, `truncate_jsonl`.
  - Dependencies: `federated/io.py`.
  - Public files: `src/dpsvdfl/federated/experiment.py`, `src/dpsvdfl/federated/io.py`.
  - Why included: a resumed run uses absolute round indices; the test suite checks that a resumed run is bitwise
    identical to an uninterrupted one.
- **Final evaluation**
  - Paper: S6.1; S9.
  - Original files: `training/evaluation.py`, `federated/runner.py`.
  - Functions: `evaluate_classification_accuracy`.
  - Public file: `src/dpsvdfl/training/evaluation.py`.
  - Why included: top-1 accuracy on the full test split, every 10 rounds and at round 300; never the best checkpoint.

## 2. Variants

| Variant | Key in this repository | Ranks | Noise namespace |
|---|---|---|---|
| DP-SVD-FL (adaptive) | `adaptive` | rank rule on W^(t−1) at every round | `dp-svd-fl` |
| Fixed-rank spectral ablation (Sec. VI-C) | `fixed_spectral` | ranks k_l^(0) from W(0), bases recomputed every round | `fixed-spectral` |

Both variants use the same trainable set (layer3/4 convolutions projected, fc unprojected) and plain local SGD.

Code locations:

- Registry: `src/dpsvdfl/federated/methods.py`.
- Rank strategies: `src/dpsvdfl/spectral/strategies.py`.
- Original files: `federated/method_registry.py`, `federated/variants.py` and `spectral/strategies.py` (fixed spectral);
  `federated/e3_e9_registry.py` (noise contexts).

## 3. Experiments, tables and figures

### 3.1 Experiments (configurations)

| Paper | Experiment | Configuration | Original campaign |
|---|---|---|---|
| V, VI-A, Table III (DP-SVD-FL) | DP-SVD-FL, 3 datasets × 2 partitions × 3 seeds = 18 runs | `configs/main_e10.json` | E3 suites (`adaptive` arm) |
| VI-C fixed-rank ablation | CIFAR-10 Dir., 3 seeds | `configs/fixed_rank_e10.json` | E6-A2 (`fixed_spectral` arm) |
| VI-C, Table V, Fig. 2 (DP-SVD-FL) | privacy sweep σ ∈ {0.788, 0.95, 1.11, 1.687}, 12 runs | `configs/privacy_sweep_e10.json` | E9 (`adaptive` arm, noise context `e9:sigma<1000σ>`) |

### 3.2 Tables and statistics

| Paper | Output | Script | Original script |
|---|---|---|---|
| Table II | shared protocol | `configs/*.json` (`protocol`) | campaign manifests |
| Table III (DP-SVD-FL column) | `table3_dpsvdfl_accuracy` | `analysis/make_tables.py` | `outputs/paper_tpami_e10/aggregate_results.py`, `recompute_tables.py` |
| Table IV, D fractions (VI-B) | `table4_privatized_dimension` | `analysis/make_tables.py` | same |
| Table V (DP-SVD-FL) | `table5_privacy_sweep` | `analysis/make_tables.py` | same |
| Fixed-rank ablation numbers | `fixed_rank_ablation` | `analysis/make_tables.py` | same |
| S9 statistics (round 300, sample std) | all tables | `analysis/common.py` | same |
| Epsilon values (Tables II, V; S8.1) | console | `scripts/compute_privacy.py` | `privacy/accountant.py` |

### 3.3 Figures

| Paper | Output | Script | Original script |
|---|---|---|---|
| Fig. 1 (DP-SVD-FL curves) | `fig01_learning_curves_dpsvdfl` | `analysis/make_figures.py` | `outputs/paper_tpami_e10/figures/make_tpami_figures.py` |
| Fig. 2 (DP-SVD-FL) | `fig02_privacy_utility_e10` | `analysis/make_figures.py` | same |
| Fig. 3 | `fig03_adaptive_dimension_e10` | `analysis/make_figures.py` | same |
| Fig. G6 (S10.4) | `G6_projected_dimension_EN` | `analysis/noise_only_statistics.py` | `outputs/rank_rule_noise_only_e10/G6_EN_verification/make_G6_projected_dimension_EN.py` |

### 3.4 Supplementary analyses

| Paper | Analysis | Script | Original script |
|---|---|---|---|
| S10.1–S10.3, Table S2 | rank-rule audit (checks, branches, D(t), rank-change events) | `analysis/rank_audit.py` | `rank_rule_audit_e10/scripts/audit_rank_rule.py`, `recompute_endpoints.py` |
| S10.4 (noise ratios) | server-update norm / noise scale | `analysis/rank_audit.py` | `rank_rule_audit_e10/scripts/noise_share.py` |
| S10.4, Table S3 | data-free noise-only control, 30 seeds | `analysis/noise_only_control.py`, `analysis/noise_only_statistics.py` | `outputs/rank_rule_noise_only_e10/scripts/noise_only_rank_replay.py`, `noise_only_30_seeds.py`, `analyse_30_seeds.py` |
| S10.5 | S_late differences, seed-level permutation p-values, Holm | `analysis/noise_only_statistics.py` | `analyse_30_seeds.py` |

## 4. Theory

The derivations of Sec. IV and S1–S5 are mathematical results. Their numerically checkable consequences are covered by
the unit tests:

- **Isometry, Parseval identity and orthogonal projector (Lemma 1, S1):** `tests/test_spectral.py`.
- **Sensitivity C under joint clipping (Theorem 1):** `tests/test_privacy.py`.
- **Noise covariance σ²C²/(qN)² per coordinate (Proposition 1, S2):** `tests/test_privacy.py`.
- **Accountant values (Theorem 2, S8):** `tests/test_privacy.py`, `tests/equivalence`.

## 5. Not included

Only DP-SVD-FL and its fixed-rank ablation are included. The comparison methods of the paper, and the extensions,
pilots and exploratory campaigns of the internal research repository, are not part of this repository. See
`ORIGINAL_TO_PUBLIC_FILES.csv`.
