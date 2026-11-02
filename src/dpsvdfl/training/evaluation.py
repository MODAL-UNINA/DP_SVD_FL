"""Top-1 test accuracy of a released global model (Supplementary Sec. S6.1)."""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dpsvdfl.training.local import _check_num_workers


@torch.no_grad()
def evaluate_classification_accuracy(
    model: nn.Module,
    dataset: Dataset,
    device: str | torch.device | None = None,
    batch_size: int = 256,
    num_workers: int = 0,
    multiprocessing_context: str | None = None,
) -> float:
    """Top-1 accuracy of `model` on `dataset`, in dataset order (no shuffling).

    The dataset's own transform is used as is (the evaluation transform for a test set).
    The model is evaluated in eval mode on `device` (default: the device of its first
    parameter; the model is never moved). The train/eval flag of every module is
    restored afterwards, and parameters, buffers and the global RNG are left unchanged.
    """
    _check_num_workers(num_workers)
    if len(dataset) == 0:
        raise ValueError("cannot evaluate on an empty dataset")
    if device is None:
        device = next(model.parameters()).device

    modes = [(module, module.training) for module in model.modules()]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        multiprocessing_context=multiprocessing_context if num_workers > 0 else None,
        generator=torch.Generator().manual_seed(0),  # the DataLoader never touches the global RNG
    )
    correct = 0
    total = 0
    model.eval()
    try:
        for inputs, targets in loader:
            predictions = model(inputs.to(device)).argmax(dim=1).cpu()
            correct += int((predictions == targets).sum())
            total += len(targets)
    finally:
        for module, training in modes:
            module.training = training
    return correct / total
