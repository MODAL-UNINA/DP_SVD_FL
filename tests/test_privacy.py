import math

import pytest
import torch

from dpsvdfl.dp.mechanism import DPMechanismConfig, apply_reduced_dp_mechanism, joint_clip_reduced_update
from dpsvdfl.privacy.accountant import PrivacyAccountantConfig, compute_privacy_accounting, default_rdp_orders
from dpsvdfl.privacy.modes import PrivacyConfiguration, public_privacy_record
from dpsvdfl.privacy.randomness import OSRandomness, PrivateModeViolation, PrivateReleaseSeedPlan
from dpsvdfl.federated.seeds import RunSeedPlan

CONFIG = DPMechanismConfig(clipping_norm=0.2, noise_multiplier=0.95, sampling_rate=0.1, num_client_slots=500)


def test_joint_clipping():
    s = torch.full((100,), 0.1)  # ||s|| = 1 > C
    clipped = joint_clip_reduced_update(s, 0.2)
    assert clipped.was_clipped and float(torch.linalg.norm(clipped.vector.double())) == pytest.approx(0.2, rel=1e-6)
    small = torch.full((4,), 0.01)
    assert torch.equal(joint_clip_reduced_update(small, 0.2).vector, small)
    zero = joint_clip_reduced_update(torch.zeros(5), 0.2)
    assert not zero.was_clipped and torch.equal(zero.vector, torch.zeros(5))


def test_mechanism_sum_noise_and_normalisation():
    updates = [torch.full((10,), 0.01), torch.full((10,), 1.0)]
    no_noise = DPMechanismConfig(clipping_norm=0.2, noise_multiplier=0.0, sampling_rate=0.1, num_client_slots=500)
    released = apply_reduced_dp_mechanism(iter(updates), 10, no_noise, noise_seed=0).normalized_noisy_update
    expected = (updates[0] + updates[1] * (0.2 / math.sqrt(10))) / 50.0  # divided by the PUBLIC qN, not |S_t|
    assert torch.allclose(released, expected, atol=1e-7)


def test_noise_distribution_and_seed_dependence():
    D = 200_000
    a = apply_reduced_dp_mechanism(iter([]), D, CONFIG, noise_seed=1).normalized_noisy_update
    b = apply_reduced_dp_mechanism(iter([]), D, CONFIG, noise_seed=1).normalized_noisy_update
    c = apply_reduced_dp_mechanism(iter([]), D, CONFIG, noise_seed=2).normalized_noisy_update
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert float(a.std()) == pytest.approx(0.95 * 0.2 / 50, rel=0.01)  # sigma C / (qN) = 0.0038
    assert abs(float(a.mean())) < 1e-4


def test_seed_and_sampler_are_exclusive():
    with pytest.raises(ValueError):
        apply_reduced_dp_mechanism(iter([]), 3, CONFIG, noise_seed=1,
                                   noise_sampler=lambda d, s, dt, dv: torch.zeros(d))


def test_accountant_values_of_the_paper():
    assert len(default_rdp_orders()) == 221
    expected = {0.788: 16.0184, 0.95: 10.7948, 1.11: 8.0039, 1.687: 4.0017}
    for sigma, epsilon in expected.items():
        result = compute_privacy_accounting(PrivacyAccountantConfig(0.1, sigma, 300, 500))
        assert round(result.epsilon, 4) == epsilon and result.delta == 0.002
    main = compute_privacy_accounting(PrivacyAccountantConfig(0.1, 0.95, 300, 500))
    assert main.optimal_order == pytest.approx(2.1)


def test_private_release_record_composes_every_attempt():
    config = PrivacyAccountantConfig(0.1, 0.95, 300, 500)
    benchmark = public_privacy_record(config, PrivacyConfiguration("benchmark_reproducible"))
    private = public_privacy_record(config, PrivacyConfiguration("private_release"), mechanism_invocations=302)
    assert "NOT an operationally private release" in benchmark["epsilon_semantics"]
    assert private["aborted_round_attempts"] == 2 and private["epsilon"] > benchmark["epsilon"]


def test_private_release_plan_refuses_seeded_randomness():
    plan = PrivateReleaseSeedPlan(RunSeedPlan.create(0), OSRandomness())
    with pytest.raises(PrivateModeViolation):
        plan.noise_seed(0)
    with pytest.raises(PrivateModeViolation):
        _ = plan.sampling_seed
    assert plan.local_training_seed(1, 2) == RunSeedPlan.create(0).local_training_seed(1, 2)
    subset = plan.sample_round(0, 10_000, 0.1)
    assert 800 < len(subset) < 1200
