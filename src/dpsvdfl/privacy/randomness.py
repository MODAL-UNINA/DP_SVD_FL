"""Secret randomness of the ``private_release`` mode: client sampling and DP noise from the operating-system CSPRNG.

Why not PyTorch / NumPy generators. torch.Generator(...).manual_seed(secret) is NOT a cryptographic generator: the CPU
generator is the Mersenne Twister (its state is recoverable from its outputs), the CUDA generator is Philox4x32-10 (a
counter-based statistical generator without a security claim) and either is keyed by a single 64-bit seed. NumPy's bit
generators (PCG64, Philox, ...) make no security claim either. This module therefore never seeds anything and does not
implement a generator of its own: every random bit comes from os.urandom, which Python documents as suitable for
cryptographic use (Linux: the getrandom(2) system call, i.e. the kernel CSPRNG; it blocks until the pool is initialised).

Backend PRIVATE_RELEASE_BACKEND ("os-csprng/box-muller-float64-v1"):
    uniform53(n)    k / 2^53 with k uniform on {0, ..., 2^53 - 1}: the top 53 bits of 8 CSPRNG bytes
    poisson_subset  slot i is included iff uniform53 < q: P(include) = ceil(q 2^53) / 2^53 (= q up to 2^-53), independent
                    across slots and rounds (fresh bytes every call)
    gaussian_noise  Box-Muller in float64 from two 53-bit uniforms (u1 in (0, 1], u2 in [0, 1)), times sigma C, cast to
                    the update dtype and copied to the update's device, chunk by chunk (bounded host memory; nothing is
                    drawn on the GPU)

Properties and assumptions (stated, not "verified by looking at numbers"): the bits are as unpredictable as the kernel
CSPRNG (an assumption on the operating system); nothing is derived from, or reconstructible from, a public seed; nothing
is stored, so a resumed run can never re-release a noise realisation. Box-Muller is exact in real arithmetic; in float64
the samples are a discrete approximation of N(0, 1) with |z| <= sqrt(2 ln 2^53) ~ 8.57. Like every floating-point
Gaussian sampler (torch.randn included) this is NOT covered by the (epsilon, delta) of the ideal mechanism (Mironov 2012;
Holohan & Braghin 2021): floating-point and timing side channels remain an OPEN limitation of private_release.

Cost: 8 CSPRNG bytes per coordinate and round (D = 11e6: ~88 MB from the kernel, typically well under a second), plus
a float64 transform on the host; host memory is bounded by CHUNK_SIZE coordinates.
"""

import os
from collections.abc import Callable

import numpy as np
import torch

from dpsvdfl.federated.seeds import RunSeedPlan
from dpsvdfl.privacy.modes import PRIVATE_RELEASE, PRIVATE_RELEASE_BACKEND

CHUNK_SIZE = 1 << 22  # coordinates per host chunk (~32 MB of float64)
_TWO_PI = 2.0 * np.pi


class PrivateRandomnessUnavailable(RuntimeError):
    """The private_release backend cannot be used here: the run is refused (never downgraded to seeded randomness)."""


class PrivateModeViolation(RuntimeError):
    """Seed-derived DP randomness was requested in private_release mode (a programming error, refused)."""


def check_private_backend() -> dict:
    """Refuse (PrivateRandomnessUnavailable) unless the OS CSPRNG answers; return the backend description."""
    try:
        sample = os.urandom(32)
    except NotImplementedError as error:  # Python raises this when no OS randomness source exists
        raise PrivateRandomnessUnavailable(f"private_release needs os.urandom (the OS CSPRNG): {error}") from error
    if not isinstance(sample, bytes) or len(sample) != 32:
        raise PrivateRandomnessUnavailable("os.urandom returned an unexpected value; private_release is refused")
    return {"backend": PRIVATE_RELEASE_BACKEND,
            "entropy_source": "os.urandom (getrandom(2))" if hasattr(os, "getrandom") else "os.urandom",
            "seeded": False, "stored": False}


class OSRandomness:
    """Uniform, Bernoulli and Gaussian draws from os.urandom (module doc). Holds no state and no seed."""

    def __init__(self, source: Callable[[int], bytes] | None = None) -> None:
        self._source = os.urandom if source is None else source  # tests may inject a failing source

    def uniform53(self, count: int) -> np.ndarray:
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if count == 0:
            return np.empty(0, dtype=np.float64)
        raw = self._source(8 * count)
        if not isinstance(raw, bytes) or len(raw) != 8 * count:
            raise PrivateRandomnessUnavailable("the OS CSPRNG returned too few bytes")
        bits = np.frombuffer(raw, dtype=np.uint64) >> np.uint64(11)
        return bits.astype(np.float64) * (2.0 ** -53)

    def poisson_subset(self, num_slots: int, sampling_rate: float) -> np.ndarray:
        """Sorted slot IDs i in [0, N) with uniform53_i < q (one fresh draw per slot)."""
        if num_slots < 1:
            raise ValueError(f"num_slots must be at least 1, got {num_slots}")
        if not 0.0 <= sampling_rate <= 1.0:
            raise ValueError(f"sampling rate q must be in [0, 1], got {sampling_rate}")
        return np.flatnonzero(self.uniform53(num_slots) < sampling_rate).astype(np.int64)

    def standard_normal(self, count: int) -> np.ndarray:
        """`count` float64 Box-Muller samples (pairs from two fresh 53-bit uniforms)."""
        pairs = (count + 1) // 2
        radius = np.sqrt(-2.0 * np.log1p(-self.uniform53(pairs)))  # log(u1), u1 = 1 - uniform in (0, 1]
        angle = _TWO_PI * self.uniform53(pairs)
        samples = np.empty(2 * pairs, dtype=np.float64)
        samples[0::2] = radius * np.cos(angle)
        samples[1::2] = radius * np.sin(angle)
        return samples[:count]

    def gaussian_noise(self, dimension: int, standard_deviation: float, dtype: torch.dtype,
                       device: torch.device) -> torch.Tensor:
        """z ~ N(0, standard_deviation^2 I_dimension) (float64 Box-Muller, cast to dtype) on `device`."""
        noise = torch.empty(dimension, dtype=dtype, device=device)
        if standard_deviation == 0:
            return noise.zero_()
        for start in range(0, dimension, CHUNK_SIZE):
            stop = min(dimension, start + CHUNK_SIZE)
            chunk = self.standard_normal(stop - start) * float(standard_deviation)
            noise[start:stop] = torch.from_numpy(chunk).to(device=device, dtype=dtype)
        return noise


class PrivateReleaseSeedPlan:
    """The seed plan of a private_release run: public streams from the public RunSeedPlan, DP randomness from the OS.

    Delegates exactly the declared-public streams (classifier, partition, local training) and the noise namespace (an
    identifier). Every seed-derived DP stream (sampling seed, noise master seed, per-round noise seeds,
    common_streams) RAISES PrivateModeViolation, so no code path can fall back to seeded sampling or noise.
    ``attempt_recorder(round_index)`` is called before any randomness of a round is drawn (write-ahead attempt ledger
    of ``dpsvdfl.federated.experiment``); without one the attempts are only counted in memory.
    """

    privacy_mode = PRIVATE_RELEASE

    def __init__(self, public: RunSeedPlan, randomness: OSRandomness | None = None,
                 attempt_recorder: Callable[[int], None] | None = None) -> None:
        if not isinstance(public, RunSeedPlan):
            raise TypeError(f"public must be a RunSeedPlan, got {public!r}")
        self.public = public
        self.randomness = OSRandomness() if randomness is None else randomness
        self.attempt_recorder = attempt_recorder
        self.attempts: list[int] = []  # round indices whose randomness was requested by this process

    # declared-public streams and identifiers
    @property
    def experiment_seed(self) -> int:
        return self.public.experiment_seed

    @property
    def method_noise_namespace(self) -> str:
        return self.public.method_noise_namespace

    @property
    def classifier_seed(self) -> int:
        return self.public.classifier_seed

    @property
    def partition_seed(self) -> int:
        return self.public.partition_seed

    @property
    def local_training_master_seed(self) -> int:
        return self.public.local_training_master_seed

    def local_training_seed(self, round_index: int, slot_id: int) -> int:
        return self.public.local_training_seed(round_index, slot_id)

    # seed-derived DP randomness: refused
    @property
    def sampling_seed(self) -> int:
        raise PrivateModeViolation("private_release: client sampling never derives from a seed")

    @property
    def noise_master_seed(self) -> int:
        raise PrivateModeViolation("private_release: DP noise never derives from a seed")

    def noise_seed(self, round_index: int) -> int:
        raise PrivateModeViolation("private_release: DP noise never derives from a seed")

    def common_streams(self):
        raise PrivateModeViolation("private_release: the sampling stream is not a seed stream")

    # secret randomness of one round
    def sample_round(self, round_index: int, num_slots: int, sampling_rate: float) -> np.ndarray:
        """Record the attempt (write-ahead), then draw S_t from the OS CSPRNG."""
        if self.attempt_recorder is not None:
            self.attempt_recorder(round_index)
        self.attempts.append(round_index)
        return self.randomness.poisson_subset(num_slots, sampling_rate)

    def gaussian_noise(self, dimension: int, standard_deviation: float, dtype: torch.dtype,
                       device: torch.device) -> torch.Tensor:
        return self.randomness.gaussian_noise(dimension, standard_deviation, dtype, torch.device(device))


def is_private_release_plan(seed_plan) -> bool:
    return isinstance(seed_plan, PrivateReleaseSeedPlan)
