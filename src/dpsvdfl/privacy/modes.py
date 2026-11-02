"""Privacy modes of a run and the public (epsilon, delta) record (paper Sec. V-E; Supplementary Sec. S8.2).

benchmark_reproducible  (used for EVERY result reported in the paper)
    Client sampling and DP noise derive from the experiment seed through ``dpsvdfl.federated.seeds``. The benchmark
    seeds (0, 1, 2) and every derivation are PUBLICLY RECONSTRUCTIBLE: anyone who knows them can regenerate the sampled
    sets and subtract the noise. This mode reproduces experiments exactly; it does NOT provide an operationally private
    release. Its epsilon is the NOMINAL budget of the ideal randomized mechanism.

private_release
    Client sampling and DP noise come from the operating-system CSPRNG at the round that uses them
    (``dpsvdfl.privacy.randomness``): never seeded, never derived from a public seed, never stored. Model
    initialisation, data partition and local training keep their public seeds. Every round attempt is recorded before
    its randomness is drawn, and the accountant composes every recorded attempt. The paper makes no claim about the use
    of this mode.

In both modes the accountant is the same (``dpsvdfl.privacy.accountant``) and the guarantee concerns the noisy
aggregates and the released models of a trusted-server deployment; floating-point and timing side channels are not
covered.
"""

import dataclasses
from dataclasses import dataclass

from dpsvdfl.privacy.accountant import (
    ACCOUNTANT_LABEL,
    ACCOUNTANT_VERSION,
    PrivacyAccountantConfig,
    compute_privacy_accounting,
)

BENCHMARK_REPRODUCIBLE = "benchmark_reproducible"
PRIVATE_RELEASE = "private_release"
PRIVACY_MODES = (BENCHMARK_REPRODUCIBLE, PRIVATE_RELEASE)

PUBLIC_SEED_RANDOMNESS = "seed-derived (public, reconstructible)"
PRIVATE_RELEASE_BACKEND = "os-csprng/box-muller-float64-v1"  # dpsvdfl.privacy.randomness

BENCHMARK_EPSILON_SEMANTICS = (
    "nominal theoretical (epsilon, delta) of the ideal Poisson-subsampled Gaussian mechanism (real-valued noise); in "
    "benchmark_reproducible mode the client sampling and the DP noise derive from public, reconstructible seeds, so "
    "this run is NOT an operationally private release")
PRIVATE_EPSILON_SEMANTICS = (
    "(epsilon, delta) of the ideal Poisson-subsampled Gaussian mechanism over every recorded round attempt; client "
    "sampling and DP noise drawn from the operating-system CSPRNG (never seeded, derived or stored); floating-point "
    "sampling, timing and other side channels are not covered")


@dataclass(frozen=True)
class PrivacyConfiguration:
    """The privacy mode of a run (the accountant is always ``ACCOUNTANT_VERSION``)."""

    mode: str = BENCHMARK_REPRODUCIBLE

    def __post_init__(self) -> None:
        if self.mode not in PRIVACY_MODES:
            raise ValueError(f"unknown privacy mode {self.mode!r} (expected one of {PRIVACY_MODES})")

    @property
    def is_private(self) -> bool:
        return self.mode == PRIVATE_RELEASE

    def record(self) -> dict:
        return {"mode": self.mode, "accountant_version": ACCOUNTANT_VERSION,
                "dp_randomness": PRIVATE_RELEASE_BACKEND if self.is_private else PUBLIC_SEED_RANDOMNESS}


def privacy_label(privacy: PrivacyConfiguration) -> dict:
    """Public description of a privacy configuration."""
    return {"privacy_mode": privacy.mode, "accountant_version": ACCOUNTANT_VERSION,
            "epsilon_semantics": PRIVATE_EPSILON_SEMANTICS if privacy.is_private else BENCHMARK_EPSILON_SEMANTICS}


def public_privacy_record(config: PrivacyAccountantConfig, privacy: PrivacyConfiguration,
                          mechanism_invocations: int | None = None) -> dict:
    """The "privacy" field of a public run summary.

    benchmark_reproducible runs are accounted for exactly T rounds. private_release accounts every recorded round
    attempt: ``mechanism_invocations`` (>= T) replaces T, and the committed / aborted counts are recorded.
    """
    committed = config.num_rounds
    if privacy.is_private:
        if isinstance(mechanism_invocations, bool) or not isinstance(mechanism_invocations, int):
            raise ValueError("private_release accounting needs the number of recorded round attempts")
        if mechanism_invocations < committed:
            raise ValueError(f"{mechanism_invocations} recorded round attempts for {committed} committed rounds")
        config = dataclasses.replace(config, num_rounds=mechanism_invocations)
    elif mechanism_invocations is not None and mechanism_invocations != committed:
        raise ValueError("benchmark_reproducible runs are accounted for exactly T rounds")
    result = compute_privacy_accounting(config)
    record = {"epsilon": result.epsilon, "delta": result.delta, "optimal_order": result.optimal_order,
              "accountant": ACCOUNTANT_LABEL, **privacy_label(privacy), "accountant_metadata": result.record()}
    if privacy.is_private:
        record.update(committed_rounds=committed, mechanism_invocations=mechanism_invocations,
                      aborted_round_attempts=mechanism_invocations - committed)
    return record
