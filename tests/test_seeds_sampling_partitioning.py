import numpy as np
import pytest

from dpsvdfl.data.partitioning import dirichlet_partition, iid_partition, validate_partition
from dpsvdfl.federated.sampling import generate_sampling_schedule, validate_sampling_schedule
from dpsvdfl.federated.seeds import RunSeedPlan, contextual_noise_namespace, privacy_sweep_noise_context


def test_seed_plan_values_recorded_for_the_paper():
    # Values recorded for experiment seed 0 when the main campaign was prepared.
    plan = RunSeedPlan.create(0, "dp-svd-fl")
    assert plan.common_streams() == (8668861027912758289, 4881901421217228719, 7229315352737646089,
                                     4015017263998684094)
    assert plan.noise_master_seed == 4068616271134144534


def test_variants_share_common_streams_but_not_noise():
    plans = [RunSeedPlan.create(1, ns) for ns in ("dp-svd-fl", "fixed-spectral", "dp-svd-fl|e9:sigma0788")]
    assert len({plan.common_streams() for plan in plans}) == 1
    assert len({plan.noise_master_seed for plan in plans}) == len(plans)
    assert plans[0].local_training_seed(5, 17) == plans[1].local_training_seed(5, 17)


def test_noise_context():
    assert privacy_sweep_noise_context(0.788) == "e9:sigma0788"
    assert privacy_sweep_noise_context(1.687) == "e9:sigma1687"
    assert contextual_noise_namespace("dp-svd-fl", "e9:sigma0950") == "dp-svd-fl|e9:sigma0950"
    assert contextual_noise_namespace("dp-svd-fl", None) == "dp-svd-fl"
    with pytest.raises(ValueError):
        privacy_sweep_noise_context(0.7885)
    with pytest.raises(ValueError):
        contextual_noise_namespace("dp-svd-fl", "bad context")


def test_iid_partition_sizes_differ_by_at_most_one():
    slots = iid_partition(10_007, 500, seed=3)
    validate_partition(slots, 10_007, 500)
    sizes = [len(slot) for slot in slots]
    assert max(sizes) - min(sizes) <= 1


def test_dirichlet_partition_covers_every_example_once_and_is_deterministic():
    labels = np.random.default_rng(0).integers(0, 10, 5000)
    first = dirichlet_partition(labels, 500, 0.6, seed=9)
    second = dirichlet_partition(labels, 500, 0.6, seed=9)
    validate_partition(first, 5000, 500)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert any(len(slot) == 0 for slot in first)  # empty slots are kept, never redrawn


def test_poisson_schedule():
    schedule = generate_sampling_schedule(500, 0.1, 300, seed=4)
    validate_sampling_schedule(schedule, 500, 300)
    sizes = np.array([len(s) for s in schedule])
    assert abs(sizes.mean() - 50) < 2  # E|S_t| = qN = 50 (standard error ~0.4)
    assert all(np.array_equal(a, b) for a, b in zip(schedule, generate_sampling_schedule(500, 0.1, 300, seed=4)))
