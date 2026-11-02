"""Release-time equivalence checks of this repository against the reference (internal research) implementation.

The reference implementation produced the results of the paper; it is not part of this repository. These tests run
only when ``DPSVDFL_REFERENCE_SRC`` points at its ``src`` directory, e.g.::

    DPSVDFL_REFERENCE_SRC=/path/to/reference/src python -m pytest tests/equivalence -v

They also need the ImageNet ResNet-18 weights in the torchvision cache (``$TORCH_HOME``). Both implementations are run
on the same machine, on CPU, on identical synthetic inputs, and their outputs are compared for EXACT (bitwise)
equality: model parameters and buffers after T rounds, released and benchmark round records, client partitions,
seeds, rank-rule outputs, projection bases, the Gaussian mechanism and the privacy accountant.

Every implementation runs in its own subprocess with ``MKL_CBWR=COMPATIBLE`` and the same thread count. Without
conditional bitwise reproducibility, Intel MKL's float64 SVD can differ in the last bit depending on the memory
allocation history of the process: the SAME code then gives runs that differ by about 1 ulp after the first SVD of an
updated weight. This is a property of the numerical library, not of either implementation.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

REFERENCE_SRC = os.environ.get("DPSVDFL_REFERENCE_SRC")
pytestmark = pytest.mark.skipif(not REFERENCE_SRC, reason="DPSVDFL_REFERENCE_SRC is not set")

TESTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(TESTS_DIR))

from synthetic_data import synthetic_benchmark  # noqa: E402

from dpsvdfl.dp.mechanism import DPMechanismConfig  # noqa: E402
from dpsvdfl.federated.experiment import RunSpec, build_federation  # noqa: E402
from dpsvdfl.federated.methods import method_parameters  # noqa: E402
from dpsvdfl.federated.runner import DPSVDFLConfig, SpectralConfig  # noqa: E402
from dpsvdfl.training.local import LocalTrainingConfig  # noqa: E402

TINY = {"N": 8, "q": 0.5, "T": 2, "E": 1, "B": 4, "C": 0.2, "sigma": 0.95, "lr": 0.003, "lr_decay": 0.998,
        "alpha": 0.6, "eval_every": 1, "data_seed": 3}


SUBPROCESS_ENV = {"CUDA_VISIBLE_DEVICES": "", "MKL_CBWR": "COMPATIBLE", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}


def _reference(code: str) -> str:
    """Run ``code`` with the reference package on the path; return its stdout."""
    environment = {**os.environ, "PYTHONPATH": REFERENCE_SRC, **SUBPROCESS_ENV}
    completed = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=environment, capture_output=True,
                               text=True, check=False, cwd=TESTS_DIR)
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def _tiny_spec(method: str, seed: int, partition: str, noise_context: str | None = None) -> RunSpec:
    config = DPSVDFLConfig(
        num_rounds=TINY["T"],
        local=LocalTrainingConfig(local_epochs=TINY["E"], batch_size=TINY["B"], base_learning_rate=TINY["lr"],
                                  learning_rate_decay=TINY["lr_decay"]),
        dp=DPMechanismConfig(clipping_norm=TINY["C"], noise_multiplier=TINY["sigma"], sampling_rate=TINY["q"],
                             num_client_slots=TINY["N"]),
        spectral=SpectralConfig(),
    )
    return RunSpec(dataset="cifar10", partition=partition, dirichlet_alpha=TINY["alpha"] if partition == "dirichlet"
                   else None, seed=seed, method=method, method_parameters=method_parameters(method),
                   config=config, eval_every=TINY["eval_every"], noise_context=noise_context,
                   experiment_name="equivalence")


# ---------------------------------------------------------------- end-to-end runs


CASES = [
    ("adaptive", 0, "dirichlet", None),
    ("adaptive", 1, "iid", "e9:sigma0788"),
    ("adaptive", 2, "dirichlet", "e9:sigma0950"),
    ("adaptive", 0, "iid", None),
    ("fixed_spectral", 2, "dirichlet", None),
]


@pytest.mark.parametrize("method,seed,partition,noise_context", CASES)
def test_end_to_end_run_is_bitwise_identical(tmp_path, method, seed, partition, noise_context):
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps({**TINY, "method": method, "seed": seed, "partition": partition,
                                     "noise_context": noise_context}))
    reference_output, run_dir = tmp_path / "reference.pt", tmp_path / "public"
    for worker, pythonpath, target in (("reference_worker.py", REFERENCE_SRC, reference_output),
                                       ("public_worker.py", "", run_dir)):
        environment = {**os.environ, "PYTHONPATH": pythonpath, **SUBPROCESS_ENV}
        completed = subprocess.run([sys.executable, str(Path(__file__).with_name(worker)), str(spec_file),
                                    str(target)], env=environment, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
    reference = torch.load(reference_output, weights_only=False)

    train_set, _ = synthetic_benchmark(TINY["data_seed"])
    federation = build_federation(_tiny_spec(method, seed, partition, noise_context), train_set)
    assert [slot.tolist() for slot in federation.client_slots] == reference["client_slots"]
    public_state = torch.load(run_dir / "final_model.pt", weights_only=True)
    assert set(public_state) == set(reference["state"])
    for name, value in reference["state"].items():
        assert torch.equal(public_state[name], value), f"{method}: {name} differs"
    released = [json.loads(line) for line in (run_dir / "released_round_diagnostics.jsonl").read_text().splitlines()]
    assert released == json.loads(json.dumps(reference["released"]))
    benchmark = [json.loads(line) for line in (run_dir / "benchmark_diagnostics_internal.jsonl").read_text().splitlines()]
    assert benchmark == json.loads(json.dumps(reference["benchmark"]))


# ---------------------------------------------------------------- components


def test_seed_plans_and_noise_namespaces():
    from dpsvdfl.federated.seeds import RunSeedPlan, contextual_noise_namespace

    namespaces = ["dp-svd-fl", "fixed-spectral"]
    contexts = [None, "e9:sigma0788", "e9:sigma0950", "e9:sigma1110", "e9:sigma1687"]
    reference = json.loads(_reference(f"""
        import json
        from dpsvdfl.federated.seeds import RunSeedPlan, contextual_noise_namespace
        out = []
        for seed in range(4):
            for ns in {namespaces!r}:
                for ctx in {contexts!r}:
                    p = RunSeedPlan.create(seed, contextual_noise_namespace(ns, ctx))
                    out.append([p.classifier_seed, p.partition_seed, p.sampling_seed, p.local_training_master_seed,
                                p.noise_master_seed, p.local_training_seed(7, 123), p.noise_seed(299)])
        print(json.dumps(out))
    """))
    public = []
    for seed in range(4):
        for namespace in namespaces:
            for context in contexts:
                plan = RunSeedPlan.create(seed, contextual_noise_namespace(namespace, context))
                public.append([plan.classifier_seed, plan.partition_seed, plan.sampling_seed,
                               plan.local_training_master_seed, plan.noise_master_seed,
                               plan.local_training_seed(7, 123), plan.noise_seed(299)])
    assert public == reference


def test_partitions_and_schedules():
    from dpsvdfl.data.partitioning import make_partition
    from dpsvdfl.federated.sampling import generate_sampling_schedule

    reference = json.loads(_reference("""
        import json, numpy as np
        from dpsvdfl.federated.partitioning import make_partition
        from dpsvdfl.federated.sampling import generate_sampling_schedule
        labels = np.random.default_rng(5).integers(0, 47, 4000)
        out = {"iid": [s.tolist() for s in make_partition(labels, "iid", 500, 11)],
               "dir": [s.tolist() for s in make_partition(labels, "dirichlet", 500, 12, alpha=0.6)],
               "schedule": [s.tolist() for s in generate_sampling_schedule(500, 0.1, 300, 13)]}
        print(json.dumps(out))
    """))
    labels = np.random.default_rng(5).integers(0, 47, 4000)
    assert [s.tolist() for s in make_partition(labels, "iid", 500, 11)] == reference["iid"]
    assert [s.tolist() for s in make_partition(labels, "dirichlet", 500, 12, alpha=0.6)] == reference["dir"]
    assert [s.tolist() for s in generate_sampling_schedule(500, 0.1, 300, 13)] == reference["schedule"]


def test_rank_rule_and_bases_on_random_and_pretrained_matrices(tmp_path):
    from dpsvdfl.models.partitions import resnet18_partition
    from dpsvdfl.models.resnet import load_pretrained_resnet18
    from dpsvdfl.spectral.matricization import matricize

    generator = torch.Generator().manual_seed(0)
    matrices = [torch.randn(m, n, generator=generator) for m, n in ((40, 90), (90, 40), (64, 64), (5, 300))]
    matrices.append(torch.diag(torch.tensor([9.0, 4.0, 1.0, 0.5, 0.0])))  # gap branch, zero tail
    matrices.append(torch.zeros(6, 4))  # zero spectrum: fallback k = 1
    model = load_pretrained_resnet18(10, 0)
    parameters = dict(model.named_parameters())
    matrices += [matricize(parameters[name].detach()) for name in resnet18_partition(model).projected]
    torch.save(matrices, tmp_path / "matrices.pt")
    # The same code runs against both packages, each in its own subprocess (identical numerical environment).
    code = """
        import sys, torch
        from dpsvdfl.spectral.projection import compute_projection_basis
        out = []
        for w in torch.load(sys.argv[1]):
            b = compute_projection_basis(w)
            r = b.rank_rule
            out.append((b.U_k, b.V_k, b.singular_values, (r.selected_rank, r.branch, r.gap_rank, r.max_gap_ratio,
                        r.gap_rank_fraction, r.energy_rank, r.selected_energy_fraction)))
        torch.save(out, sys.argv[2])
    """
    results = []
    for name, pythonpath in (("reference", REFERENCE_SRC), ("public", str(REPO_ROOT / "src"))):
        environment = {**os.environ, "PYTHONPATH": pythonpath, **SUBPROCESS_ENV}
        completed = subprocess.run([sys.executable, "-c", textwrap.dedent(code), str(tmp_path / "matrices.pt"),
                                    str(tmp_path / f"{name}.pt")], env=environment, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        results.append(torch.load(tmp_path / f"{name}.pt", weights_only=False))
    for (U, V, s, rule), (U_public, V_public, s_public, rule_public) in zip(*results):
        assert rule_public[:6] == rule[:6]
        assert rule_public[6] == rule[6] or (np.isnan(rule_public[6]) and np.isnan(rule[6]))
        assert torch.equal(U_public, U) and torch.equal(V_public, V) and torch.equal(s_public, s)


def test_gaussian_mechanism(tmp_path):
    from dpsvdfl.dp.mechanism import apply_reduced_dp_mechanism

    generator = torch.Generator().manual_seed(1)
    vectors = [torch.randn(1000, generator=generator) * scale for scale in (0.0, 1e-3, 0.01, 1.0, 5.0)]
    torch.save(vectors, tmp_path / "vectors.pt")
    _reference(f"""
        import torch
        from dpsvdfl.dp.mechanism import DPMechanismConfig, apply_reduced_dp_mechanism
        vectors = torch.load({str(tmp_path / 'vectors.pt')!r})
        config = DPMechanismConfig(clipping_norm=0.2, noise_multiplier=0.95, sampling_rate=0.1, num_client_slots=500)
        out = [apply_reduced_dp_mechanism(iter(vectors), 1000, config, noise_seed=seed).normalized_noisy_update
               for seed in (0, 1, 2 ** 62)]
        out.append(apply_reduced_dp_mechanism(iter([]), 1000, config, noise_seed=5).normalized_noisy_update)
        torch.save(out, {str(tmp_path / 'reference.pt')!r})
    """)
    reference = torch.load(tmp_path / "reference.pt")
    config = DPMechanismConfig(clipping_norm=0.2, noise_multiplier=0.95, sampling_rate=0.1, num_client_slots=500)
    public = [apply_reduced_dp_mechanism(iter(vectors), 1000, config, noise_seed=seed).normalized_noisy_update
              for seed in (0, 1, 2 ** 62)]
    public.append(apply_reduced_dp_mechanism(iter([]), 1000, config, noise_seed=5).normalized_noisy_update)
    for a, b in zip(public, reference):
        assert torch.equal(a, b)


def test_privacy_accountant():
    from dpsvdfl.privacy.accountant import PrivacyAccountantConfig, compute_privacy_accounting

    grid = [(0.1, 0.95, 300), (0.1, 0.788, 300), (0.1, 1.11, 300), (0.1, 1.687, 300), (0.1, 2.774, 300),
            (0.05, 0.7, 50), (0.3, 1.3, 1000)]
    reference = json.loads(_reference(f"""
        import json
        from dpsvdfl.privacy.accountant import PrivacyAccountantConfig, compute_privacy_accounting
        out = []
        for q, sigma, T in {grid!r}:
            r = compute_privacy_accounting(PrivacyAccountantConfig(sampling_rate=q, noise_multiplier=sigma,
                                                                   num_rounds=T, num_client_slots=500))
            out.append([r.epsilon, r.optimal_order, r.delta])
        print(json.dumps(out))
    """))
    public = []
    for q, sigma, T in grid:
        result = compute_privacy_accounting(PrivacyAccountantConfig(sampling_rate=q, noise_multiplier=sigma,
                                                                    num_rounds=T, num_client_slots=500))
        public.append([result.epsilon, result.optimal_order, result.delta])
    assert public == reference
