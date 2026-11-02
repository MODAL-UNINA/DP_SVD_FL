"""Joint clipping and server-side Gaussian mechanism on the reduced vectors (paper Sec. III-C, Mechanism 1; Algorithm 1 lines 19-23).

For the reduced client vectors s_i in R^D of the sampled slots S_t:

    s_bar_i = s_i * min(1, C / ||s_i||_2)               ONE joint clip per client over all blocks; clip(0) = 0
    A_t     = sum_{i in S_t} s_bar_i                     unnormalised sum
    Z_t     = A_t + xi_t,  xi_t ~ N(0, sigma^2 C^2 I_D)  noise added ONCE, to the unnormalised sum
    output  = Z_t / (q N)                                fixed public normaliser, applied ONCE

Adding or removing one slot changes A_t by s_bar_i, whose norm is at most C: the L2 sensitivity of A_t is C (paper
Theorem 1). Clipping every block separately would give sqrt(L) C instead (paper Remark 2(a)) and is not offered.

The output stays a reduced vector: the reconstruction U Z V^T and the model update are in
``dpsvdfl.federated.server_update``. Only the noisy normalised vector leaves the trusted server.

Noise source. In ``benchmark_reproducible`` mode the noise comes from a ``torch.Generator`` seeded with a per-round seed
of the seed plan, so that experiments can be replayed exactly; whoever knows the seed can regenerate and subtract the
noise, hence such a run is NOT an operationally private release. In ``private_release`` mode the caller passes a
``noise_sampler`` that draws from the operating-system CSPRNG instead (``dpsvdfl.privacy.randomness``).

Device. The whole mechanism (float64 joint norm, clip, running sum, Gaussian draw, division by qN) runs on the device of
the reduced vectors. PyTorch's CPU and CUDA generators produce DIFFERENT samples for the same seed: the distribution
and the privacy accounting are identical, but a CPU run and a CUDA run release different noise realisations.
"""

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

from dpsvdfl.reproducibility import torch_generator

# Values used in the main experiments of the paper (Table II).
CLIPPING_NORM = 0.2
NOISE_MULTIPLIER = 0.95
SAMPLING_RATE = 0.1
NUM_CLIENT_SLOTS = 500


@dataclass(frozen=True)
class DPMechanismConfig:
    """Public parameters of the Gaussian mechanism (defaults: base configuration)."""

    clipping_norm: float = CLIPPING_NORM  # C
    noise_multiplier: float = NOISE_MULTIPLIER  # sigma
    sampling_rate: float = SAMPLING_RATE  # q
    num_client_slots: int = NUM_CLIENT_SLOTS  # N

    def __post_init__(self) -> None:
        _check_real(self.clipping_norm, "clipping_norm")
        _check_real(self.noise_multiplier, "noise_multiplier")
        _check_real(self.sampling_rate, "sampling_rate")
        if not self.clipping_norm > 0:
            raise ValueError(f"clipping_norm must be positive, got {self.clipping_norm}")
        if not self.noise_multiplier >= 0:
            raise ValueError(f"noise_multiplier must be non-negative, got {self.noise_multiplier}")
        if not 0 < self.sampling_rate <= 1:
            raise ValueError(f"sampling_rate must be in (0, 1], got {self.sampling_rate}")
        if isinstance(self.num_client_slots, bool) or not isinstance(self.num_client_slots, int):
            raise TypeError(f"num_client_slots must be an integer, got {self.num_client_slots!r}")
        if self.num_client_slots < 1:
            raise ValueError(f"num_client_slots must be at least 1, got {self.num_client_slots}")

    @property
    def normalizer(self) -> float:
        """qN, the fixed public denominator (not rounded; independent of the realised |S_t|)."""
        return self.sampling_rate * self.num_client_slots

    @property
    def noise_std(self) -> float:
        """sigma * C: per-coordinate standard deviation of the noise added to the unnormalised sum."""
        return self.noise_multiplier * self.clipping_norm

    @property
    def normalized_noise_std(self) -> float:
        """sigma * C / (qN): per-coordinate noise standard deviation of the normalised output."""
        return self.noise_std / self.normalizer


@dataclass(frozen=True)
class ClippingResult:
    """One joint clip. `vector` is a new detached tensor with the input dtype, on the input device.

    clipped_norm = original_norm * scale (the float64 norm of s * scale, derived from the scalars
    instead of rescanning the vector; C up to rounding for a clipped vector).
    """

    vector: torch.Tensor
    original_norm: float
    clipped_norm: float
    scale: float
    was_clipped: bool


@dataclass(frozen=True)
class ClippingObservation:
    """INTERNAL scalar diagnostics of the clip of the index-th vector of a stream (NOT a DP release).

    Carries no coordinate and no client identifier: `index` is the position in the iterable.
    """

    index: int
    original_norm: float  # ||s_i||_2, float64
    clipped_norm: float  # original_norm * scale
    scale: float  # min(1, C / ||s_i||_2); 1 for a zero vector
    was_clipped: bool  # scale < 1


@dataclass(frozen=True)
class DPMechanismResult:
    """Released output of one round of the mechanism: the noisy normalised reduced update and public constants."""

    normalized_noisy_update: torch.Tensor
    dimension: int
    normalizer: float
    noise_std: float
    normalized_noise_std: float


def joint_clip_reduced_update(reduced_update: torch.Tensor, clipping_norm: float) -> ClippingResult:
    """s_bar = s * min(1, C / ||s||_2) with ONE scalar for the whole vector; clip(0) = 0.

    The norm is computed in float64 over all coordinates of s together (all blocks), on the
    device of s. The input is never modified; the result never aliases it.
    """
    _check_clipping_norm(clipping_norm)
    _check_reduced_vector(reduced_update, check_finite=False)
    update = reduced_update.detach()
    original_norm, scale = _clip_scale(update, clipping_norm)  # the only full-vector norm
    if original_norm == 0.0:
        vector = torch.zeros_like(update)
    elif scale == 1.0:
        vector = update.clone()
    else:
        vector = _rescale(update, scale)
    return ClippingResult(
        vector=vector,
        original_norm=original_norm,
        clipped_norm=original_norm * scale,
        scale=scale,
        was_clipped=scale < 1.0,
    )


def sum_clipped_updates(
    clipped_updates: Iterable[torch.Tensor],
    dimension: int,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """A_t = sum of the (already clipped) vectors, accumulated into one zero vector on `device`.

    No averaging, no normalisation, no noise. An empty iterable gives the zero vector.
    """
    _check_dimension(dimension)
    device = torch.device(device)
    total = torch.zeros(dimension, dtype=dtype, device=device)
    for update in clipped_updates:
        _check_reduced_vector(update, dimension, dtype, device)
        total += update.detach()
    return total


def sample_gaussian_noise(
    dimension: int,
    standard_deviation: float,
    seed: int,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """z ~ N(0, standard_deviation^2 I_dimension) from a local generator on `device` seeded with `seed`.

    Equivalent to torch.randn(dimension, generator=torch.Generator(device).manual_seed(seed), dtype=dtype,
    device=device) * standard_deviation; the global RNG is not used. A zero standard deviation gives
    exact zeros. The same seed gives the same draw on the same device type, but CPU and CUDA generators
    give different (equally distributed) draws (module doc).
    `seed` is internal randomness: whoever knows it can regenerate z, so it must not be released.
    """
    _check_dimension(dimension)
    _check_real(standard_deviation, "standard_deviation")
    if not standard_deviation >= 0:
        raise ValueError(f"standard_deviation must be non-negative, got {standard_deviation}")
    device = torch.device(device)
    generator = torch_generator(seed, device=device)
    if standard_deviation == 0:
        return torch.zeros(dimension, dtype=dtype, device=device)
    return torch.randn(dimension, generator=generator, dtype=dtype, device=device) * standard_deviation


def apply_reduced_dp_mechanism(
    reduced_updates: Iterable[torch.Tensor],
    dimension: int,
    config: DPMechanismConfig,
    noise_seed: int | None,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    on_clip: Callable[[ClippingObservation], None] | None = None,
    *,
    noise_sampler: Callable[[int, float, torch.dtype, torch.device], torch.Tensor] | None = None,
) -> DPMechanismResult:
    """Clip every raw s_i once, sum, add N(0, sigma^2 C^2 I_D) to the sum, divide once by qN.

    `reduced_updates` are the raw encoded vectors s_i of the sampled slots, as any iterable
    (empty slots give zero vectors; an empty round is an empty iterable). `dimension`
    (D of the round), `dtype` and `device` are explicit so that an empty round is well defined:
    it releases noise / (qN). Every vector must live on `device` (default CPU; CUDA in production
    on a GPU), and the whole computation, noise included, runs there (module doc).
    `noise_seed` is an internal per-round reproducibility seed. Use a fresh seed for
    every round and do not expose it as part of the released DP output.

    The iterable is consumed in a single streaming pass: each raw vector is clipped and
    added to the running sum, and no list of uploads is kept, so a generator that trains
    and encodes one client at a time holds only O(1) vectors. Every vector is validated
    (type, 1-D, device, length D, dtype; finite through its float64 norm); an invalid vector
    aborts the call before any noise is drawn or anything is released.

    Each ||s_i||_2 is computed ONCE (float64, on `device`). `on_clip`, if given, is called after
    each vector has been added to the sum with its ClippingObservation (INTERNAL scalars only;
    an exception raised there aborts the call before any noise is drawn).

    noise_sampler (keyword-only; private_release mode, dpsvdfl.privacy.randomness): a callable
    (dimension, sigma C, dtype, device) -> z that draws the noise from secret randomness; noise_seed must
    then be None (a seed is never combined with it). Default None: the seeded noise above (benchmark_reproducible).
    Clipping, the sum, sigma C and the division by qN are the same in both cases.
    """
    _check_dimension(dimension)
    if noise_sampler is not None and noise_seed is not None:
        raise ValueError("a noise sampler (private_release) is never combined with a noise seed")
    if noise_sampler is None and noise_seed is None:
        raise TypeError("the seeded mechanism needs an integer noise seed")
    device = torch.device(device)
    clipped_sum = torch.zeros(dimension, dtype=dtype, device=device)  # A_t, unnormalised
    for index, update in enumerate(reduced_updates):
        _check_reduced_vector(update, dimension, dtype, device, check_finite=False)
        clipped = joint_clip_reduced_update(update, config.clipping_norm)  # the only full-vector norm of s_i
        del update  # streaming: only the running sum survives the iteration
        clipped_sum += clipped.vector
        if on_clip is not None:
            on_clip(ClippingObservation(index=index, original_norm=clipped.original_norm,
                                        clipped_norm=clipped.clipped_norm, scale=clipped.scale,
                                        was_clipped=clipped.was_clipped))
        del clipped
    if noise_sampler is None:
        noise = sample_gaussian_noise(dimension, config.noise_std, noise_seed, dtype, device)
    else:
        noise = noise_sampler(dimension, config.noise_std, dtype, device)
        if (not isinstance(noise, torch.Tensor) or noise.shape != (dimension,) or noise.dtype != dtype
                or not _on_device(noise, device)):
            raise ValueError("the noise sampler must return a 1-D tensor of length D with the update dtype and device")
    noisy_sum = clipped_sum + noise
    normalized_noisy_update = noisy_sum / config.normalizer  # the only division by qN

    return DPMechanismResult(
        normalized_noisy_update=normalized_noisy_update,
        dimension=dimension,
        normalizer=config.normalizer,
        noise_std=config.noise_std,
        normalized_noise_std=config.normalized_noise_std,
    )


def expected_squared_noise_norm(dimension: int, config: DPMechanismConfig, normalized: bool = False) -> float:
    """Theoretical E||z||_2^2 of the Gaussian noise of one round (a pure scalar of public constants).

    normalized=False: noise added to the unnormalised sum, D * (sigma C)^2;
    normalized=True:  noise in the released vector,        D * (sigma C / (qN))^2.
    The per-coordinate standard deviation sigma C does not depend on D, so representations with the
    same (C, sigma, q, N) differ only through D. Never inspects a noise realisation.
    """
    _check_dimension(dimension)
    standard_deviation = config.normalized_noise_std if normalized else config.noise_std
    return dimension * standard_deviation * standard_deviation


def _clip_scale(update: torch.Tensor, clipping_norm: float) -> tuple[float, float]:
    """(||s||_2 in float64, min(1, C / ||s||_2)) with ONE reduction on the device of s.

    A non-finite norm means a non-finite coordinate (or, for float64 inputs only, a norm that
    overflows); the vector is rejected either way, before it reaches the sum.
    """
    original_norm = float(torch.linalg.vector_norm(update.double()))  # one scalar device->host copy
    if not math.isfinite(original_norm):
        raise ValueError("a reduced update contains non-finite values (its float64 norm is not finite)")
    if original_norm <= clipping_norm:  # zero vector included: clip(0) = 0
        return original_norm, 1.0
    return original_norm, clipping_norm / original_norm


def _rescale(update: torch.Tensor, scale: float) -> torch.Tensor:
    """s * scale computed in float64 and cast back to the dtype of s (on its device)."""
    return (update.double() * scale).to(update.dtype)


def _check_clipping_norm(clipping_norm: float) -> None:
    _check_real(clipping_norm, "clipping_norm")
    if not clipping_norm > 0:
        raise ValueError(f"clipping_norm must be positive, got {clipping_norm}")


def _check_real(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")


def _check_dimension(dimension: int) -> None:
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError(f"dimension must be a positive integer, got {dimension!r}")


def _on_device(tensor: torch.Tensor, device: torch.device) -> bool:
    """True if `tensor` lives on `device` (a device without an index matches any index of its type)."""
    return tensor.device.type == device.type and (device.index is None or tensor.device.index == device.index)


def _check_reduced_vector(
    vector: torch.Tensor,
    dimension: int | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    check_finite: bool = True,
) -> None:
    """Metadata checks (no data scan), then an optional finiteness scan."""
    if not isinstance(vector, torch.Tensor):
        raise TypeError(f"a reduced update must be a torch.Tensor, got {type(vector).__name__}")
    if not vector.is_floating_point():
        raise TypeError(f"a reduced update must be floating-point, got {vector.dtype}")
    if vector.ndim != 1:
        raise ValueError(f"a reduced update must be 1-D, got shape {tuple(vector.shape)}")
    if device is not None and not _on_device(vector, device):
        raise ValueError(f"a reduced update is on device {vector.device}, but the mechanism runs on {device}")
    if dimension is not None and vector.numel() != dimension:
        raise ValueError(f"a reduced update has length {vector.numel()}, expected D = {dimension}")
    if dtype is not None and vector.dtype != dtype:
        raise TypeError(f"a reduced update has dtype {vector.dtype}, expected {dtype}")
    if check_finite and not torch.isfinite(vector).all():
        raise ValueError("a reduced update contains non-finite values")
