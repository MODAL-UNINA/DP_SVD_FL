"""Run a tiny experiment with the REFERENCE (internal research) implementation and save its outputs.

This script imports the reference package, NOT this repository's ``dpsvdfl``: it is executed in a subprocess whose
PYTHONPATH points at the reference source tree (``DPSVDFL_REFERENCE_SRC``). It mirrors exactly what the reference
campaign code did for one run: seed plan of the variant's noise namespace, client partition from the common partition
seed, Poisson schedule from the common sampling seed, W^(0) = pretrained ResNet-18 + classifier from the classifier
seed, ``apply_partition`` of the trainable set, then the reference runner of DP-SVD-FL or of the fixed-rank ablation.

    python reference_worker.py <spec.json> <output.pt>
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/ (synthetic_data, no dpsvdfl import)

from synthetic_data import synthetic_benchmark  # noqa: E402

from dpsvdfl.data.datasets import dataset_labels  # noqa: E402
from dpsvdfl.dp.mechanism import DPMechanismConfig  # noqa: E402
from dpsvdfl.federated.partitioning import make_partition  # noqa: E402
from dpsvdfl.federated.runner import (  # noqa: E402
    DPSVDFLConfig,
    SpectralConfig,
    benchmark_round_record,
    released_round_record,
    run_dpsvdfl,
)
from dpsvdfl.federated.seeds import RunSeedPlan, contextual_noise_namespace  # noqa: E402
from dpsvdfl.federated.variants import run_projection_variant  # noqa: E402
from dpsvdfl.models.partitions import apply_partition, resnet18_partition  # noqa: E402
from dpsvdfl.models.resnet import load_pretrained_resnet18  # noqa: E402
from dpsvdfl.spectral.strategies import compute_initial_rank_snapshot, fixed_spectral_strategy  # noqa: E402
from dpsvdfl.training.local import LocalTrainingConfig  # noqa: E402

NAMESPACES = {"adaptive": "dp-svd-fl", "fixed_spectral": "fixed-spectral"}


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text())
    method, seed, context = spec["method"], spec["seed"], spec.get("noise_context")
    train_set, test_set = synthetic_benchmark(spec["data_seed"])
    config = DPSVDFLConfig(
        num_rounds=spec["T"],
        local=LocalTrainingConfig(local_epochs=spec["E"], batch_size=spec["B"], base_learning_rate=spec["lr"],
                                  learning_rate_decay=spec["lr_decay"], batching_policy="fixed_padded_last"),
        dp=DPMechanismConfig(clipping_norm=spec["C"], noise_multiplier=spec["sigma"], sampling_rate=spec["q"],
                             num_client_slots=spec["N"]),
        spectral=SpectralConfig(tau=2.0, rho_max=0.05, gamma=0.95),
    )
    plan = RunSeedPlan.create(seed, contextual_noise_namespace(NAMESPACES[method], context))
    labels = dataset_labels(train_set)
    alpha = spec["alpha"] if spec["partition"] == "dirichlet" else None
    client_slots = make_partition(labels, spec["partition"], spec["N"], plan.partition_seed, alpha=alpha)
    model = load_pretrained_resnet18(num_classes=10, classifier_seed=plan.classifier_seed)
    canonical = resnet18_partition(model)
    records, benchmark = [], []

    def on_round_committed(round_index, global_model, result, evaluation):
        record = released_round_record(result)
        if evaluation is not None:
            record["test_accuracy"] = evaluation["test_accuracy"]
        records.append(record)
        benchmark.append(benchmark_round_record(result))

    common = dict(test_set=test_set, eval_every=spec["eval_every"], device="cpu", on_round_committed=on_round_committed)
    if method == "adaptive":
        partition, run = canonical, run_dpsvdfl
    else:  # fixed_spectral
        snapshot = compute_initial_rank_snapshot(model, canonical, tau=2.0, rho_max=0.05, gamma=0.95)
        partition = canonical
        strategy = fixed_spectral_strategy(snapshot)

        def run(*args, **kwargs):
            return run_projection_variant(*args, strategy=strategy, noise_context=context, **kwargs)
    apply_partition(model, partition)
    run(model, partition, train_set, client_slots, plan, config, **common)
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    torch.save({"state": state, "released": records, "benchmark": benchmark,
                "client_slots": [list(map(int, slot)) for slot in client_slots]}, sys.argv[2])


if __name__ == "__main__":
    main()
