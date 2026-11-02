"""Tiny synthetic image datasets for CPU tests (no download, no dpsvdfl import).

Images are 3 x 40 x 40 uniform noise. The training split applies a random 32 x 32 crop and a random horizontal flip
drawn from the global torch RNG, as torchvision's PIL transforms do, so that the augmentation random stream of the
local-training code is exercised. ``targets`` is a list of ints (CIFAR-style) and ``classes`` has ``num_classes``
entries, like the torchvision datasets.
"""

import torch
from torch.utils.data import Dataset


class SyntheticImages(Dataset):
    def __init__(self, num_examples: int, num_classes: int = 10, seed: int = 0, train: bool = True) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.rand(num_examples, 3, 40, 40, generator=generator)
        self.targets = torch.randint(0, num_classes, (num_examples,), generator=generator).tolist()
        self.classes = [str(label) for label in range(num_classes)]
        self.train = train

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        image = self.images[index]
        if self.train:
            top, left = torch.randint(0, 9, (2,)).tolist()
            image = image[:, top:top + 32, left:left + 32]
            if torch.rand(1).item() < 0.5:
                image = image.flip(-1)
        else:
            image = image[:, 4:36, 4:36]
        return image.contiguous(), self.targets[index]


def synthetic_benchmark(seed: int = 0, num_train: int = 72, num_test: int = 40):
    """(training set, test set) with 10 classes."""
    return SyntheticImages(num_train, 10, seed, train=True), SyntheticImages(num_test, 10, seed + 1000, train=False)
