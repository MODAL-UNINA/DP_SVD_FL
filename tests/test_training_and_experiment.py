import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from dpsvdfl.dp.mechanism import DPMechanismConfig
from dpsvdfl.federated.experiment import RunResources, RunSpec, expand_runs, load_experiment_config, run_experiment
from dpsvdfl.federated.methods import METHODS, method_parameters
from dpsvdfl.federated.runner import DPSVDFLConfig
from dpsvdfl.models.partitions import apply_partition, resnet18_partition
from dpsvdfl.models.resnet import load_pretrained_resnet18
from dpsvdfl.training.local import LocalTrainingConfig, round_learning_rate, train_client

REPO_ROOT = Path(__file__).resolve().parents[1]


def tiny_spec(method, seed=0, rounds=2, privacy_mode="benchmark_reproducible"):
    config = DPSVDFLConfig(num_rounds=rounds, local=LocalTrainingConfig(local_epochs=1, batch_size=4),
                           dp=DPMechanismConfig(sampling_rate=0.5, num_client_slots=8))
    return RunSpec(dataset="cifar10", partition="dirichlet", dirichlet_alpha=0.6, seed=seed, method=method,
                   method_parameters=method_parameters(method), config=config, eval_every=1,
                   privacy_mode=privacy_mode, experiment_name="test")


def resources(tmp_path, **kwargs):
    return RunResources(data_root=tmp_path, device=torch.device("cpu"), log=lambda line: None, **kwargs)


def test_learning_rate_schedule():
    assert round_learning_rate(0, 0.003, 0.998) == 0.003
    assert round_learning_rate(299, 0.003, 0.998) == pytest.approx(0.003 * 0.998 ** 299)


def test_local_training_keeps_batchnorm_frozen_and_empty_slot_is_zero(pretrained_available, tiny_data):
    train_set, _ = tiny_data
    model = load_pretrained_resnet18(10, 0)
    partition = resnet18_partition(model)
    apply_partition(model, partition)
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    weights = {name: value.detach().clone() for name, value in model.named_parameters()}
    config = LocalTrainingConfig(local_epochs=1, batch_size=4)
    result = train_client(model, partition, train_set, np.arange(10), 0, 123, config)
    assert result.num_batches == 3  # ceil(10 / 4): the last batch is padded to 4
    assert set(result.delta) == set(partition.trainable)
    for name, value in model.named_buffers():  # running statistics never change
        assert torch.equal(value, buffers[name])
    again = train_client(model, partition, train_set, np.arange(10), 0, 123, config)
    assert all(torch.equal(result.delta[k], again.delta[k]) for k in result.delta)  # same seed, same update
    empty = train_client(model, partition, train_set, np.array([], dtype=np.int64), 0, 1, config)
    assert empty.num_batches == 0 and all(not value.any() for value in empty.delta.values())
    for name, value in model.named_parameters():  # training runs on a copy: the global model W^(t-1) is unchanged
        assert torch.equal(value, weights[name])


@pytest.mark.parametrize("method", sorted(METHODS))
def test_both_variants_run_and_write_their_outputs(pretrained_available, tiny_data, tmp_path, method):
    train_set, test_set = tiny_data
    summary = run_experiment(tiny_spec(method), tmp_path / method, resources(tmp_path), train_set, test_set)
    records = [json.loads(line) for line in (tmp_path / method / "released_round_diagnostics.jsonl").open()]
    assert [r["round"] for r in records] == [1, 2] and "test_accuracy" in records[-1]
    assert summary["privacy"]["privacy_mode"] == "benchmark_reproducible"
    assert records[0]["D"] < records[0]["D_full"] and "ranks" in records[0]
    # a completed run is recognised and not rerun
    assert run_experiment(tiny_spec(method), tmp_path / method, resources(tmp_path), train_set, test_set) == summary


def test_resume_is_bitwise_identical(pretrained_available, tiny_data, tmp_path):
    train_set, test_set = tiny_data
    run_experiment(tiny_spec("adaptive", rounds=3), tmp_path / "straight", resources(tmp_path), train_set, test_set)

    class Interrupt(Exception):
        pass

    def stop_after_round_2(line):
        if "round 002/003" in line:
            raise Interrupt

    with pytest.raises(Interrupt):
        run_experiment(tiny_spec("adaptive", rounds=3), tmp_path / "resumed",
                       RunResources(data_root=tmp_path, device=torch.device("cpu"), log=stop_after_round_2,
                                    checkpoint_every=1), train_set, test_set)
    run_experiment(tiny_spec("adaptive", rounds=3), tmp_path / "resumed", resources(tmp_path, checkpoint_every=1),
                   train_set, test_set)
    straight = torch.load(tmp_path / "straight" / "final_model.pt")
    resumed = torch.load(tmp_path / "resumed" / "final_model.pt")
    assert all(torch.equal(straight[k], resumed[k]) for k in straight)
    assert ((tmp_path / "straight" / "released_round_diagnostics.jsonl").read_text()
            == (tmp_path / "resumed" / "released_round_diagnostics.jsonl").read_text())


def test_resume_refuses_a_different_specification(pretrained_available, tiny_data, tmp_path):
    train_set, test_set = tiny_data
    run_experiment(tiny_spec("adaptive", rounds=1), tmp_path / "run", resources(tmp_path), train_set, test_set)
    (tmp_path / "run" / "status.json").unlink()
    with pytest.raises(RuntimeError, match="differs"):
        run_experiment(tiny_spec("adaptive", seed=1, rounds=1), tmp_path / "run", resources(tmp_path), train_set,
                       test_set)


def test_private_release_mode(pretrained_available, tiny_data, tmp_path):
    train_set, test_set = tiny_data
    spec = tiny_spec("adaptive", privacy_mode="private_release")
    with pytest.raises(ValueError):
        run_experiment(spec, tmp_path / "private", resources(tmp_path, checkpoint_every=5), train_set, test_set)
    summary = run_experiment(spec, tmp_path / "private", resources(tmp_path, checkpoint_every=1), train_set, test_set)
    assert summary["privacy"]["privacy_mode"] == "private_release"
    assert summary["privacy"]["mechanism_invocations"] == 2
    attempts = (tmp_path / "private" / "round_attempts_internal.jsonl").read_text().splitlines()
    assert len(attempts) == 2


@pytest.mark.parametrize("name,count", [("main_e10", 18), ("fixed_rank_e10", 3), ("privacy_sweep_e10", 12)])
def test_configurations_expand_to_the_paper_grids(name, count):
    config = load_experiment_config(REPO_ROOT / "configs" / f"{name}.json")
    runs = expand_runs(config)
    assert len(runs) == count
    for spec in runs:
        assert (spec.config.dp.num_client_slots, spec.config.dp.sampling_rate, spec.config.num_rounds) == (500, 0.1, 300)
        assert (spec.config.local.local_epochs, spec.config.local.batch_size) == (10, 50)
        assert (spec.config.local.base_learning_rate, spec.config.local.learning_rate_decay) == (0.003, 0.998)
        assert spec.config.dp.clipping_norm == 0.2 and spec.config.accountant_config().target_delta == 0.002
        assert spec.privacy_mode == "benchmark_reproducible"
    if name == "privacy_sweep_e10":
        assert {spec.noise_namespace for spec in runs} == {
            "dp-svd-fl|e9:sigma0788", "dp-svd-fl|e9:sigma0950", "dp-svd-fl|e9:sigma1110", "dp-svd-fl|e9:sigma1687"}
    else:
        assert all(spec.config.dp.noise_multiplier == 0.95 and spec.noise_context is None for spec in runs)


def test_command_line_interfaces():
    listed = subprocess.run([sys.executable, "scripts/run_experiment.py", "--config", "configs/fixed_rank_e10.json",
                             "--list"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert "3 run(s)" in listed.stdout
    privacy = subprocess.run([sys.executable, "scripts/compute_privacy.py"], cwd=REPO_ROOT, capture_output=True,
                             text=True, check=True)
    assert "epsilon=10.79479" in privacy.stdout
