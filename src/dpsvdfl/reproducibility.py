"""Seed helpers: every random stream of an experiment is derived from explicit integer seeds.

All seeds are internal, data-independent experiment metadata. What a seed builds may be public (for example the
classifier initialisation), whereas the DP-noise seeds drive randomness that must never be released in a real
deployment (see ``dpsvdfl.privacy.modes`` for the difference between the two privacy modes).
"""

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch


def _check_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer, got {seed!r}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")


def numpy_generator(seed: int) -> np.random.Generator:
    """A new NumPy generator driven only by ``seed`` (``None`` is rejected: it would draw OS entropy)."""
    _check_seed(seed)
    return np.random.default_rng(seed)


def torch_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    """A new ``torch.Generator`` on ``device`` seeded with ``seed``.

    The same seed gives the same stream on the same device type, but the CPU and CUDA generators produce different
    streams (this is why a CPU run and a CUDA run release different, equally distributed, noise realisations).
    """
    _check_seed(seed)
    return torch.Generator(device=device).manual_seed(int(seed))


def derive_indexed_seed(base_seed: int, *components: int) -> int:
    """Seed of position ``components`` of the stream ``base_seed`` (NumPy ``SeedSequence`` spawn key).

    A pure function of its arguments: e.g. ``derive_indexed_seed(local_master, round_index, slot_id)`` does not depend
    on which other seeds were derived before. Returns a non-negative 63-bit integer.
    """
    _check_seed(base_seed)
    for component in components:
        _check_seed(component)
    sequence = np.random.SeedSequence(int(base_seed), spawn_key=tuple(int(c) for c in components))
    return int(sequence.generate_state(1, dtype=np.uint64)[0]) & ((1 << 63) - 1)


def stable_namespace_id(namespace: str) -> int:
    """Stable 63-bit integer ID of a string (first 8 bytes of its SHA-256); unlike ``hash()`` it never changes."""
    if not isinstance(namespace, str) or not namespace:
        raise ValueError(f"namespace must be a non-empty string, got {namespace!r}")
    digest = hashlib.sha256(namespace.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def derive_seeds(seed: int, count: int) -> list[int]:
    """``count`` independent 32-bit seeds derived from one seed (NumPy ``SeedSequence``)."""
    _check_seed(seed)
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    return [int(value) for value in np.random.SeedSequence(int(seed)).generate_state(count)]


@contextmanager
def seeded_cpu_rng(seed: int) -> Iterator[None]:
    """Run a block with the default CPU generator seeded to ``seed``; the previous state is restored on exit."""
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        yield
