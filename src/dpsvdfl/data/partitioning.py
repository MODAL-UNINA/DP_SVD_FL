"""Distribution of the training set over N public client slots (paper Sec. V-A; Supplementary Sec. S6.2).

A partition is a list of exactly N int64 arrays: element i holds the training-set
indices of slot i (public ID i). Empty arrays are valid slots and are never removed.

All randomness comes from one local NumPy generator built from ``seed`` (the partition seed of the seed plan),
independently of the learning method and of the sampling seed.
"""

import math

import numpy as np

from dpsvdfl.reproducibility import numpy_generator

NUM_CLIENT_SLOTS = 500
DIRICHLET_ALPHA = 0.6

PARTITION_SCHEMES = ("iid", "dirichlet")


def iid_partition(num_examples: int, num_clients: int, seed: int) -> list[np.ndarray]:
    """IID partition: shuffle all indices once, then split them as evenly as possible.

    Slot sizes differ by at most one (the first num_examples mod N slots get one
    extra example). If num_examples < N, the last slots are empty.
    """
    if num_examples < 0:
        raise ValueError(f"num_examples must be non-negative, got {num_examples}")
    _check_num_clients(num_clients)
    rng = numpy_generator(seed)

    shuffled_indices = rng.permutation(num_examples)
    return list(np.array_split(shuffled_indices, num_clients))


def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float, seed: int) -> list[np.ndarray]:
    """Dirichlet label partition with a symmetric per-class Dirichlet(alpha, ..., alpha) over all N slots.

    For every class c present in `labels` (in increasing label order):
      1. collect the indices of class c and shuffle them;
      2. draw p_c ~ Dirichlet(alpha * 1_N);
      3. draw integer counts ~ Multinomial(n_c, p_c), which sum exactly to n_c;
      4. give the next counts[i] shuffled indices of class c to slot i.
    Finally the indices inside every slot are shuffled.

    No minimum slot size is enforced, nothing is redrawn or rebalanced: empty
    slots are a valid outcome and stay in the partition.
    """
    labels = _check_labels(labels)
    _check_num_clients(num_clients)
    if not (math.isfinite(alpha) and alpha > 0):
        raise ValueError(f"Dirichlet alpha must be a finite positive number, got {alpha}")
    rng = numpy_generator(seed)

    empty = np.empty(0, dtype=np.int64)
    pieces_per_client: list[list[np.ndarray]] = [[empty] for _ in range(num_clients)]

    for label in np.unique(labels):
        class_indices = rng.permutation(np.flatnonzero(labels == label))
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        counts = rng.multinomial(len(class_indices), proportions)

        boundaries = np.cumsum(counts)[:-1]
        for client_id, piece in enumerate(np.split(class_indices, boundaries)):
            pieces_per_client[client_id].append(piece)

    return [rng.permutation(np.concatenate(pieces)) for pieces in pieces_per_client]


def make_partition(
    labels: np.ndarray,
    scheme: str,
    num_clients: int,
    seed: int,
    alpha: float | None = None,
) -> list[np.ndarray]:
    """Build the training-set partition for `scheme` in {"iid", "dirichlet"}."""
    if scheme == "iid":
        if alpha is not None:
            raise ValueError("alpha is only used by the Dirichlet partition")
        return iid_partition(len(_check_labels(labels)), num_clients, seed)
    if scheme == "dirichlet":
        if alpha is None:
            raise ValueError("the Dirichlet partition requires alpha")
        return dirichlet_partition(labels, num_clients, alpha, seed)
    raise ValueError(f"unknown partition scheme {scheme!r}; expected one of {PARTITION_SCHEMES}")


def validate_partition(partition: list[np.ndarray], num_examples: int, num_clients: int) -> None:
    """Check that there are exactly N slots and that every index in [0, num_examples) occurs exactly once."""
    if len(partition) != num_clients:
        raise ValueError(f"expected {num_clients} client slots, got {len(partition)}")
    for client_id, indices in enumerate(partition):
        if not isinstance(indices, np.ndarray) or indices.ndim != 1:
            raise ValueError(f"slot {client_id}: expected a 1-D numpy array of indices")
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"slot {client_id}: indices must be integers, got dtype {indices.dtype}")

    all_indices = np.concatenate([np.asarray(indices, dtype=np.int64) for indices in partition])
    out_of_range = (all_indices < 0) | (all_indices >= num_examples)
    if np.any(out_of_range):
        raise ValueError(
            f"{int(out_of_range.sum())} indices are out of range [0, {num_examples}), "
            f"e.g. {int(all_indices[out_of_range][0])}"
        )
    occurrences = np.bincount(all_indices, minlength=num_examples)
    duplicated = np.flatnonzero(occurrences > 1)
    if len(duplicated) > 0:
        raise ValueError(f"{len(duplicated)} indices are duplicated, e.g. {int(duplicated[0])}")
    missing = np.flatnonzero(occurrences == 0)
    if len(missing) > 0:
        raise ValueError(f"{len(missing)} indices are missing, e.g. {int(missing[0])}")


def _check_num_clients(num_clients: int) -> None:
    if num_clients < 1:
        raise ValueError(f"num_clients must be at least 1, got {num_clients}")


def _check_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be a 1-D array, got shape {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"labels must be integers, got dtype {labels.dtype}")
    return labels
