"""Local minibatch policy ``fixed_padded_last`` (Supplementary Sec. S7.1).

Each local epoch draws one shuffled permutation of the client's n examples and traverses it once in batches of B.
Complete batches are unchanged. A final incomplete batch of r < B examples keeps those r examples and is completed
with B - r examples drawn uniformly WITH REPLACEMENT from the same client's examples. Every optimizer step therefore
uses exactly B examples, every original example appears once per epoch before any padding (nothing is dropped),
padding may repeat local examples, and no padding is drawn when n mod B = 0. An epoch has ceil(n / B) steps.

RNG. The permutation comes from ``torch.utils.data.RandomSampler`` and its shuffle generator, iterated exactly once per
epoch, so padding never alters an epoch's permutation. Padding positions come from a dedicated generator seeded with
``padding_seed_for(training_seed)``; it touches no other stream. Padded examples are ordinary dataset accesses: they
receive fresh random augmentations and therefore consume additional augmentation draws.
"""

import hashlib
import math

import torch
from torch.utils.data import RandomSampler, Sampler

from dpsvdfl.reproducibility import derive_indexed_seed, stable_namespace_id

FIXED_PADDED_LAST = "fixed_padded_last"
LOCAL_BATCHING_POLICIES = (FIXED_PADDED_LAST,)

# FROZEN: the padding stream of every experiment of the paper was derived from this exact namespace string (its
# historical name is kept on purpose: changing it would change every padding draw).
PADDING_RNG_NAMESPACE = "diagnostic-fixed-batch-padding"


def check_batching_policy(policy: str) -> None:
    if policy not in LOCAL_BATCHING_POLICIES:
        raise ValueError(f"batching_policy must be one of {LOCAL_BATCHING_POLICIES}, got {policy!r}")


def padding_seed_for(training_seed: int, namespace: str = PADDING_RNG_NAMESPACE) -> int:
    """Seed of the padding generator of one client call (a pure function of its training seed)."""
    return derive_indexed_seed(training_seed, stable_namespace_id(namespace))


def values_fingerprint(values) -> str:
    """Short SHA-256 fingerprint of a sequence of values (e.g. one epoch's padding positions)."""
    return hashlib.sha256(",".join(str(value) for value in values).encode()).hexdigest()[:16]


class PaddedLastBatchSampler(Sampler[list[int]]):
    """Batches of one epoch of ``sampler``, with only a final incomplete batch padded to ``batch_size``.

    Each epoch the wrapped RandomSampler is iterated exactly once, as ``BatchSampler`` does, so the permutation stream
    is unchanged. Padding positions (into the client's own Subset) are drawn with replacement from
    ``padding_generator`` only. Per-epoch scalar statistics are appended to ``epochs`` (diagnostics only).
    """

    def __init__(self, sampler: RandomSampler, batch_size: int, padding_generator: torch.Generator,
                 record_padding_indices: bool = False) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self.sampler = sampler
        self.batch_size = batch_size
        self.padding_generator = padding_generator
        self.record_padding_indices = record_padding_indices
        self.epochs: list[dict] = []

    def __len__(self) -> int:
        return math.ceil(len(self.sampler) / self.batch_size)

    def __iter__(self):
        order = [int(position) for position in self.sampler]  # this epoch's permutation (one pass)
        size = self.batch_size
        batches = [order[start:start + size] for start in range(0, len(order), size)]
        padding = []
        if batches and len(batches[-1]) < size:
            padding = torch.randint(len(order), (size - len(batches[-1]),), generator=self.padding_generator).tolist()
            batches[-1] = batches[-1] + padding
        record = {
            "epoch": len(self.epochs) + 1,
            "n": len(order),
            "num_batches": len(batches),
            "num_padding_occurrences": len(padding),
            "padding_fingerprint": values_fingerprint(padding) if padding else None,
        }
        if self.record_padding_indices:
            record["padding_positions"] = padding
        self.epochs.append(record)
        yield from batches
