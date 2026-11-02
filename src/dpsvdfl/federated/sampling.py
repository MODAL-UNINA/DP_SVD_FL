"""Poisson client-slot sampling (Algorithm 1, line 9).

In every round each of the N public slots is included independently with probability q: S_t = {i : b_i = 1} with
b_i ~ Bernoulli(q). The number of participants is random (mean qN), S_t may be empty, and empty slots are sampled
exactly like the others. The sampler never looks at the data.

The sampled sets are internal to the trusted server: the Poisson-amplified accountant certifies the release only for
observers who do not see S_t (paper, Remark 2(e)).
"""

import numpy as np

from dpsvdfl.reproducibility import numpy_generator


def sample_poisson_clients(num_clients: int, q: float, rng: np.random.Generator) -> np.ndarray:
    """One round: the sorted IDs i in [0, N) with b_i = 1, where b_i ~ Bernoulli(q) i.i.d."""
    if num_clients < 1:
        raise ValueError(f"num_clients must be at least 1, got {num_clients}")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"sampling rate q must be in [0, 1], got {q}")
    # rng.random() is uniform on [0, 1), so P(u_i < q) = q.
    inclusion = rng.random(num_clients) < q
    return np.flatnonzero(inclusion)


def generate_sampling_schedule(num_clients: int, q: float, num_rounds: int, seed: int) -> list[np.ndarray]:
    """[S_1, ..., S_T] (list index t - 1 holds round t), drawn from ONE generator built from ``seed``.

    The schedule depends only on (N, q, T, sampling seed), so the adaptive and fixed-rank variants reuse exactly the
    same sequence of sampled sets at the same experiment seed.
    """
    if num_rounds < 0:
        raise ValueError(f"num_rounds must be non-negative, got {num_rounds}")
    rng = numpy_generator(seed)
    return [sample_poisson_clients(num_clients, q, rng) for _ in range(num_rounds)]


def validate_sampling_schedule(schedule: list[np.ndarray], num_clients: int, num_rounds: int) -> None:
    """Check a schedule: T rounds, each a sorted 1-D array of distinct slot IDs in [0, N)."""
    if len(schedule) != num_rounds:
        raise ValueError(f"the sampling schedule has {len(schedule)} rounds, expected T = {num_rounds}")
    for round_index, sampled in enumerate(schedule):
        values = np.asarray(sampled)
        if values.ndim != 1 or (values.size > 0 and not np.issubdtype(values.dtype, np.integer)):
            raise ValueError(f"round index {round_index}: sampled slot IDs must be a 1-D integer array")
        if values.size > 0 and (values.min() < 0 or values.max() >= num_clients):
            raise ValueError(f"round index {round_index}: sampled slot IDs must lie in [0, {num_clients})")
        if np.any(np.diff(values) <= 0):
            raise ValueError(f"round index {round_index}: sampled slot IDs must be sorted and distinct")
