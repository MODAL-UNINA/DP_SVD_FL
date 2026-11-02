"""The experimental inputs of the paper are reproduced exactly on the real datasets (requires the downloaded data).

For every (dataset, partition, seed) of the main campaign, the SHA-256 fingerprints of W^(0) (pretrained ResNet-18 +
seeded classifier), of the client partition and of the 300-round Poisson schedule must equal the values recorded when
the experiments of the paper were prepared. The same inputs were shared by every run of a seed, including the
separate fixed-rank and privacy-sweep campaigns. Set ``DPSVDFL_DATA_ROOT`` (default ``data``) to the
dataset directory; the test is skipped when a dataset is missing.
"""

import os
from pathlib import Path

import pytest

from dpsvdfl.data.datasets import load_benchmark
from dpsvdfl.federated.experiment import build_federation, expand_runs, load_experiment_config
from dpsvdfl.federated.io import cpu_state_dict, state_dict_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DPSVDFL_DATA_ROOT", REPO_ROOT / "data"))

# (dataset, partition, seed): (W^(0), partition, sampling schedule) fingerprints recorded for the paper's runs.
RECORDED = {
    ("cifar10", "dirichlet", 0): ("fd31f7f8994bd127c87ec3cde123a0b5759526a1714ca7fcc13bb44bd2ffd5d3",
        "f2686077e54f042bd1b832e9b231bfbc3f047ef17d9e31e7646ade0e2ce785a3",
        "0bd1a0804646cf4474386b2b0eaa8d9d9e65d899963f5a0f5209ddeaa4310982"),
    ("cifar10", "dirichlet", 1): ("496d2fb78731bd5ca9d84680352f26369e7f2f45efe872ed6a606fd9cefa5adc",
        "edf655628503658c9ab336d5edc04da6e7db5b3b35d3473a219db8f82b59b7e9",
        "285a79c18d1625de435d03a674e58b9004dcc6ef246db9a3b140c5f203b86b37"),
    ("cifar10", "dirichlet", 2): ("1370fdc950e690a02eb90d21010f5372572226cdd5b782a6968d1584cb49dff3",
        "9712c9ac6717260e96d336a410db9ce26027b1a0ef9ac1e48f5b426d9f544a1a",
        "0a686d1f9d86363eb4609285d4071dd2300a35922a3701b85b88b9b63791c11f"),
    ("cifar10", "iid", 0): ("fd31f7f8994bd127c87ec3cde123a0b5759526a1714ca7fcc13bb44bd2ffd5d3",
        "d57f034ad0e5e25e05b210fc47fec039a205c3e5fee710006b772c52791d6341",
        "0bd1a0804646cf4474386b2b0eaa8d9d9e65d899963f5a0f5209ddeaa4310982"),
    ("cifar10", "iid", 1): ("496d2fb78731bd5ca9d84680352f26369e7f2f45efe872ed6a606fd9cefa5adc",
        "254e12ea4bff5171f749c8e9c60aced14a3837a0759d8db2e0e4e8a13ee2aded",
        "285a79c18d1625de435d03a674e58b9004dcc6ef246db9a3b140c5f203b86b37"),
    ("cifar10", "iid", 2): ("1370fdc950e690a02eb90d21010f5372572226cdd5b782a6968d1584cb49dff3",
        "3f034ce649b0fd5c2231b8c2d8177c7220f614911b1bfa656d3603322759fc14",
        "0a686d1f9d86363eb4609285d4071dd2300a35922a3701b85b88b9b63791c11f"),
    ("cifar100", "dirichlet", 0): ("3549f28289c24811fa1014c1c6cae99b168611ddce8a7fc554f56b8c44faad75",
        "81b6c1b5416968cd90ff124f1dad0d2050d64ac4decb1255082009e7233299ba",
        "0bd1a0804646cf4474386b2b0eaa8d9d9e65d899963f5a0f5209ddeaa4310982"),
    ("cifar100", "dirichlet", 1): ("0108378349277e7aeb39ed0649ce055dc2748b112b3c66c5d97addde1b053ea1",
        "60b4a5aa5dd678df1e4a2435e7ab3db062df705e0a8037850fc8de6d93a979e1",
        "285a79c18d1625de435d03a674e58b9004dcc6ef246db9a3b140c5f203b86b37"),
    ("cifar100", "dirichlet", 2): ("31d98225f490e831d6d68f26df569899f12451eef8448498d2497d9b46e4d4af",
        "1cd7519176d92eb0d7163bce6b868f539490fedaad4ef73f79f6194371290cee",
        "0a686d1f9d86363eb4609285d4071dd2300a35922a3701b85b88b9b63791c11f"),
    ("cifar100", "iid", 0): ("3549f28289c24811fa1014c1c6cae99b168611ddce8a7fc554f56b8c44faad75",
        "d57f034ad0e5e25e05b210fc47fec039a205c3e5fee710006b772c52791d6341",
        "0bd1a0804646cf4474386b2b0eaa8d9d9e65d899963f5a0f5209ddeaa4310982"),
    ("cifar100", "iid", 1): ("0108378349277e7aeb39ed0649ce055dc2748b112b3c66c5d97addde1b053ea1",
        "254e12ea4bff5171f749c8e9c60aced14a3837a0759d8db2e0e4e8a13ee2aded",
        "285a79c18d1625de435d03a674e58b9004dcc6ef246db9a3b140c5f203b86b37"),
    ("cifar100", "iid", 2): ("31d98225f490e831d6d68f26df569899f12451eef8448498d2497d9b46e4d4af",
        "3f034ce649b0fd5c2231b8c2d8177c7220f614911b1bfa656d3603322759fc14",
        "0a686d1f9d86363eb4609285d4071dd2300a35922a3701b85b88b9b63791c11f"),
    ("emnist_balanced", "dirichlet", 0): ("7157a830aa5a835c2acd40b7210ca364d65304570a3a19908792939b64642a9f",
        "5ea055b76186cc4fb55b3e95624f0da205dddaf2672ca7a3f8ed4c3dcd4161fc",
        "0bd1a0804646cf4474386b2b0eaa8d9d9e65d899963f5a0f5209ddeaa4310982"),
    ("emnist_balanced", "dirichlet", 1): ("bf4ce0bcb22489ccdf3eb29751b006cfc0cddc8d5ff450e480340d254c7b23eb",
        "1657a787b888f267875ae4bd45bf94cd7bde064aca29d6d650f988c6bc7abfc8",
        "285a79c18d1625de435d03a674e58b9004dcc6ef246db9a3b140c5f203b86b37"),
    ("emnist_balanced", "dirichlet", 2): ("8ad8288a21c9c84a600e1818345eac6a0fa62502244f8c6f5201d2d7f710c13d",
        "6c03cb026f29423896516221e41d88bb4e297964567d0f6dbd6cf045a1bcf37f",
        "0a686d1f9d86363eb4609285d4071dd2300a35922a3701b85b88b9b63791c11f"),
    ("emnist_balanced", "iid", 0): ("7157a830aa5a835c2acd40b7210ca364d65304570a3a19908792939b64642a9f",
        "de35ac9639ce4586cc8263926faff3ac6eebdc683433848f65d421d95ac6d1bc",
        "0bd1a0804646cf4474386b2b0eaa8d9d9e65d899963f5a0f5209ddeaa4310982"),
    ("emnist_balanced", "iid", 1): ("bf4ce0bcb22489ccdf3eb29751b006cfc0cddc8d5ff450e480340d254c7b23eb",
        "7034e6961ffc42f2511a317afaa49ecd05713579b38c6db798efcafc768f2474",
        "285a79c18d1625de435d03a674e58b9004dcc6ef246db9a3b140c5f203b86b37"),
    ("emnist_balanced", "iid", 2): ("8ad8288a21c9c84a600e1818345eac6a0fa62502244f8c6f5201d2d7f710c13d",
        "a8769747c551a39f18c2d67370af9dc16df2966fdd3ae66662f4994deb312e41",
        "0a686d1f9d86363eb4609285d4071dd2300a35922a3701b85b88b9b63791c11f"),
}


def _load(dataset):
    try:
        return load_benchmark(dataset, DATA_ROOT, download=False)
    except RuntimeError:
        pytest.skip(f"{dataset} is not available under {DATA_ROOT}")


@pytest.mark.parametrize("dataset", ["cifar10", "cifar100", "emnist_balanced"])
def test_initial_model_partition_and_schedule_fingerprints(dataset):
    train_set, _ = _load(dataset)
    config = load_experiment_config(REPO_ROOT / "configs" / "main_e10.json")
    for spec in expand_runs(config, datasets=[dataset], methods=["adaptive"]):
        federation = build_federation(spec, train_set)
        model_fingerprint = state_dict_fingerprint(cpu_state_dict(federation.initial_model()))
        expected = RECORDED[(dataset, spec.partition, spec.seed)]
        assert (model_fingerprint, federation.fingerprints["partition"],
                federation.fingerprints["sampling_schedule"]) == expected, (dataset, spec.partition, spec.seed)
