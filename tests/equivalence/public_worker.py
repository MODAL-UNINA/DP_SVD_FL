"""Run the same tiny experiment as ``reference_worker.py`` with THIS repository's code (subprocess counterpart).

    python public_worker.py <spec.json> <run_dir>
"""

import json
import sys
from pathlib import Path

import torch

TESTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(TESTS_DIR.parent / "src"))

from synthetic_data import synthetic_benchmark  # noqa: E402

from dpsvdfl.dp.mechanism import DPMechanismConfig  # noqa: E402
from dpsvdfl.federated.experiment import RunResources, RunSpec, run_experiment  # noqa: E402
from dpsvdfl.federated.methods import method_parameters  # noqa: E402
from dpsvdfl.federated.runner import DPSVDFLConfig, SpectralConfig  # noqa: E402
from dpsvdfl.training.local import LocalTrainingConfig  # noqa: E402


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text())
    method = spec["method"]
    config = DPSVDFLConfig(
        num_rounds=spec["T"],
        local=LocalTrainingConfig(local_epochs=spec["E"], batch_size=spec["B"], base_learning_rate=spec["lr"],
                                  learning_rate_decay=spec["lr_decay"]),
        dp=DPMechanismConfig(clipping_norm=spec["C"], noise_multiplier=spec["sigma"], sampling_rate=spec["q"],
                             num_client_slots=spec["N"]),
        spectral=SpectralConfig(),
    )
    run_spec = RunSpec(dataset="cifar10", partition=spec["partition"],
                       dirichlet_alpha=spec["alpha"] if spec["partition"] == "dirichlet" else None, seed=spec["seed"],
                       method=method, method_parameters=method_parameters(method), config=config,
                       eval_every=spec["eval_every"], noise_context=spec.get("noise_context"),
                       experiment_name="equivalence")
    train_set, test_set = synthetic_benchmark(spec["data_seed"])
    run_dir = Path(sys.argv[2])
    run_experiment(run_spec, run_dir, RunResources(data_root=run_dir, device=torch.device("cpu"), checkpoint_every=1,
                                                   log=lambda line: None), train_set=train_set, test_set=test_set)


if __name__ == "__main__":
    main()
