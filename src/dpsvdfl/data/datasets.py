"""Benchmark datasets: CIFAR-10, CIFAR-100 and EMNIST Balanced (paper Sec. V-A).

The official training split is distributed over the client slots; the official test split is a single centralised
evaluation set and is never partitioned (Supplementary Sec. S6).
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets

from dpsvdfl.data.transforms import eval_transform, train_transform

DATASET_NAMES = ("cifar10", "cifar100", "emnist_balanced")

NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "emnist_balanced": 47,
}

# Fixed EMNIST split (47 classes); ByClass is not used.
EMNIST_SPLIT = "balanced"


def num_classes(dataset_name: str) -> int:
    """Number of classes of a benchmark dataset (known without loading it)."""
    if dataset_name not in NUM_CLASSES:
        raise ValueError(f"unknown dataset {dataset_name!r}; expected one of {DATASET_NAMES}")
    return NUM_CLASSES[dataset_name]


def load_dataset(dataset_name: str, root: str | Path, train: bool, download: bool = False) -> Dataset:
    """Load one split with its preprocessing: training transforms if `train`, evaluation transforms otherwise.

    Nothing is downloaded unless `download=True`.
    """
    expected_classes = num_classes(dataset_name)
    transform = train_transform(dataset_name) if train else eval_transform(dataset_name)
    root = Path(root)
    try:
        if dataset_name == "cifar10":
            dataset = datasets.CIFAR10(root, train=train, transform=transform, download=download)
        elif dataset_name == "cifar100":
            dataset = datasets.CIFAR100(root, train=train, transform=transform, download=download)
        else:
            dataset = datasets.EMNIST(root, split=EMNIST_SPLIT, train=train, transform=transform, download=download)
    except RuntimeError as error:
        if download:
            raise
        raise RuntimeError(
            f"{dataset_name} ({'train' if train else 'test'}) was not found or is corrupted under {root}. "
            f"Pass download=True (CLI: --download) to download it. Original error: {error}"
        ) from error

    if len(dataset.classes) != expected_classes:
        raise RuntimeError(f"{dataset_name}: expected {expected_classes} classes, found {len(dataset.classes)}")
    return dataset


def load_benchmark(dataset_name: str, root: str | Path, download: bool = False) -> tuple[Dataset, Dataset]:
    """Return (training set, test set) of a benchmark dataset."""
    train_set = load_dataset(dataset_name, root, train=True, download=download)
    test_set = load_dataset(dataset_name, root, train=False, download=download)
    return train_set, test_set


def dataset_labels(dataset: Dataset) -> np.ndarray:
    """Return the labels of a torchvision dataset as a 1-D int64 array, without loading any image.

    CIFAR stores `targets` as a list of ints and EMNIST as a tensor.
    """
    targets = dataset.targets
    if isinstance(targets, torch.Tensor):
        labels = targets.detach().cpu().numpy()
    else:
        labels = np.asarray(targets)
    labels = labels.astype(np.int64, copy=False)
    if labels.ndim != 1:
        raise ValueError(f"expected 1-D labels, got shape {labels.shape}")
    return labels
