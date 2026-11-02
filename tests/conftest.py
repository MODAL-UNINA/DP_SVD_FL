"""Shared fixtures: tiny synthetic data and the pretrained ResNet-18 (skipped when the weights cannot be loaded)."""

import pytest
import torch

from synthetic_data import synthetic_benchmark


@pytest.fixture(scope="session")
def pretrained_available() -> bool:
    try:
        from dpsvdfl.models.resnet import load_pretrained_resnet18

        load_pretrained_resnet18(10, 0)
    except Exception as error:  # no cached weights and no network
        pytest.skip(f"the ImageNet ResNet-18 weights are not available: {error}")
    return True


@pytest.fixture()
def tiny_data():
    return synthetic_benchmark(seed=3)


@pytest.fixture(autouse=True)
def _single_thread_determinism():
    torch.manual_seed(0)
    yield
