#!/usr/bin/env python
"""Download the public datasets and the ImageNet ResNet-18 checkpoint used by the experiments.

    python scripts/prepare_data.py --data-root data
    python scripts/prepare_data.py --data-root data --datasets cifar10

Datasets go to ``--data-root`` (torchvision layout). The pretrained weights
``ResNet18_Weights.IMAGENET1K_V1`` are cached by torchvision under ``$TORCH_HOME`` (default ``~/.cache/torch``).
The script also prints the number of examples and classes of every split as a sanity check.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from torchvision.models import resnet18  # noqa: E402

from dpsvdfl.data.datasets import DATASET_NAMES, load_benchmark  # noqa: E402
from dpsvdfl.models.resnet import RESNET18_WEIGHTS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CIFAR-10, CIFAR-100, EMNIST Balanced and ResNet-18 weights.")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_NAMES), choices=DATASET_NAMES)
    parser.add_argument("--skip-weights", action="store_true", help="do not download the pretrained ResNet-18")
    args = parser.parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        train_set, test_set = load_benchmark(name, args.data_root, download=True)
        print(f"{name}: {len(train_set)} training / {len(test_set)} test examples, {len(train_set.classes)} classes")
    if not args.skip_weights:
        resnet18(weights=RESNET18_WEIGHTS)
        print(f"pretrained weights available: {RESNET18_WEIGHTS}")


if __name__ == "__main__":
    main()
