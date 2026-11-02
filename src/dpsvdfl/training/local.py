"""Local client training and the uncompressed client update (Algorithm 1, lines 13-14; Supplementary Sec. S7.1).

A sampled non-empty client i starts from the released global model W^(t-1), runs E local epochs of SGD on its own
data, updating only the trainable blocks B, and returns

    Delta W_i[name] = W_i[name] - W^(t-1)[name]    for every name in B   (native parameter shapes).

The update is NOT projected, clipped, noised or aggregated here.

Local protocol of the paper: plain SGD (no momentum, no weight decay), minibatch size B = 50 with the
``fixed_padded_last`` policy (``dpsvdfl.training.batching``), learning rate lr_t = 0.003 * 0.998^(t-1) where t is the
communication round, a FRESH optimizer for every client call, and every BatchNorm module kept in evaluation mode.
An empty slot performs no optimisation, draws no random number and returns an exact zero update.

Randomness of a client call (all derived from its training seed, common to both variants of the method): the shuffle
generator of the RandomSampler, the augmentation stream (with num_workers = 0 the PIL transforms run inside a scoped CPU
RNG seeded with the augmentation seed) and the padding generator. The minibatches and augmentations are therefore
identical for the adaptive and the fixed-rank variant at the same experiment seed.
"""

import copy
import math
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset

from dpsvdfl.models.partitions import BATCHNORM_TYPES, ModelPartition, set_batchnorm_eval
from dpsvdfl.reproducibility import derive_seeds, seeded_cpu_rng
from dpsvdfl.training.batching import FIXED_PADDED_LAST, PaddedLastBatchSampler, check_batching_policy, padding_seed_for

AUGMENTATION_BACKEND = "cpu"  # the PIL pipelines of dpsvdfl.data.transforms, as in every experiment of the paper
MULTIPROCESSING_CONTEXTS = ("spawn", "forkserver", "fork")


@dataclass(frozen=True)
class LocalTrainingConfig:
    """Local-training hyperparameters (defaults: the protocol of the paper, Table II)."""

    local_epochs: int = 10
    batch_size: int = 50
    base_learning_rate: float = 0.003
    learning_rate_decay: float = 0.998  # per communication round
    momentum: float = 0.0
    weight_decay: float = 0.0
    batching_policy: str = FIXED_PADDED_LAST

    def __post_init__(self) -> None:
        if self.local_epochs < 1:
            raise ValueError(f"local_epochs must be at least 1, got {self.local_epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")
        if not (math.isfinite(self.base_learning_rate) and self.base_learning_rate > 0):
            raise ValueError(f"base_learning_rate must be positive, got {self.base_learning_rate}")
        if not (math.isfinite(self.learning_rate_decay) and self.learning_rate_decay > 0):
            raise ValueError(f"learning_rate_decay must be positive, got {self.learning_rate_decay}")
        if self.momentum < 0 or self.weight_decay < 0:
            raise ValueError("momentum and weight_decay must be non-negative")
        check_batching_policy(self.batching_policy)


@dataclass
class LocalTrainingResult:
    """Outcome of one local-training call.

    delta:         {parameter name: Delta W_i[name]} for every trainable block, native shapes, detached; on CPU by
                   default, on the training device with ``keep_delta_on_device=True``.
    num_examples:  |D_i|, the number of original local examples (0 for an empty slot).
    num_batches:   optimizer steps over all local epochs, E * ceil(|D_i| / B) (0 for an empty slot).
    mean_loss:     occurrence-weighted mean training cross-entropy (diagnostic only; None for an empty slot).
    learning_rate: the local learning rate lr_t of this round.
    timings:       observational wall-clock seconds (never part of a result).
    """

    delta: dict[str, torch.Tensor]
    num_examples: int
    num_batches: int
    mean_loss: float | None
    learning_rate: float
    timings: dict[str, float] = field(default_factory=dict, compare=False)


def round_learning_rate(round_index: int, base_learning_rate: float = 0.003, decay: float = 0.998) -> float:
    """lr_t = base_lr * decay ** round_index, with round_index = 0 for the first communication round."""
    if isinstance(round_index, bool) or not isinstance(round_index, (int, np.integer)):
        raise TypeError(f"round_index must be an integer, got {round_index!r}")
    if round_index < 0:
        raise ValueError(f"round_index must be non-negative (the first round is 0), got {round_index}")
    if not (math.isfinite(base_learning_rate) and base_learning_rate > 0):
        raise ValueError(f"base_learning_rate must be positive, got {base_learning_rate}")
    if not (math.isfinite(decay) and decay > 0):
        raise ValueError(f"decay must be positive, got {decay}")
    return float(base_learning_rate * decay**round_index)


# ---------------------------------------------------------------- client data loader


def seed_dataloader_worker(worker_id: int) -> None:
    """``worker_init_fn``: seed Python ``random`` and NumPy in a DataLoader worker from its torch seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def client_loader_seeds(training_seed: int) -> tuple[int, int, int]:
    """(shuffle_seed, augmentation_seed, padding_seed) of one client call, all derived from its training seed."""
    shuffle_seed, augmentation_seed = derive_seeds(training_seed, 2)
    return shuffle_seed, augmentation_seed, padding_seed_for(training_seed)


def make_client_dataloader(
    train_set: Dataset,
    client_indices: np.ndarray,
    batch_size: int,
    shuffle_seed: int,
    augmentation_seed: int,
    padding_seed: int,
    num_workers: int = 0,
    multiprocessing_context: str | None = None,
    pin_memory: bool = False,
) -> DataLoader:
    """DataLoader over exactly the client's training indices, reshuffled every epoch (fixed_padded_last policy).

    Only local generators are used: the minibatch order comes from a RandomSampler driven by ``shuffle_seed``; the
    DataLoader generator (``augmentation_seed``) only seeds worker processes when num_workers > 0 (with num_workers = 0,
    as in the paper, ``train_client`` drives the transforms with ``seeded_cpu_rng(augmentation_seed)``).
    """
    _check_num_workers(num_workers)
    if len(client_indices) == 0:
        raise ValueError("an empty client slot has no DataLoader; train_client returns a zero update instead")
    subset = Subset(train_set, [int(index) for index in client_indices])
    shuffle_generator = torch.Generator().manual_seed(shuffle_seed)
    worker_generator = torch.Generator().manual_seed(augmentation_seed)
    sampler = RandomSampler(subset, generator=shuffle_generator)  # new permutation every epoch
    batch_sampler = PaddedLastBatchSampler(sampler, batch_size, torch.Generator().manual_seed(padding_seed))
    return DataLoader(
        subset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        worker_init_fn=seed_dataloader_worker,
        generator=worker_generator,
        multiprocessing_context=multiprocessing_context if num_workers > 0 else None,
        pin_memory=pin_memory,
    )


def make_optimizer(model: nn.Module, learning_rate: float, config: LocalTrainingConfig) -> torch.optim.Optimizer:
    """A fresh SGD optimizer over the parameters with ``requires_grad=True``."""
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(trainable_parameters) == 0:
        raise ValueError("the model has no trainable parameter")
    return torch.optim.SGD(trainable_parameters, lr=learning_rate, momentum=config.momentum,
                           weight_decay=config.weight_decay)


def set_local_training_mode(model: nn.Module) -> None:
    """``model.train()`` followed by ``set_batchnorm_eval(model)``: BatchNorm never updates its statistics."""
    model.train()
    set_batchnorm_eval(model)


# ---------------------------------------------------------------- local SGD


def run_local_sgd(
    model: nn.Module,
    dataloader: DataLoader,
    learning_rate: float,
    config: LocalTrainingConfig,
    device: str | torch.device,
    non_blocking: bool = False,
) -> tuple[int, float]:
    """Train ``model`` in place for ``config.local_epochs`` epochs; return (num_batches, mean_loss).

    ``model`` must already be on ``device``; a new optimizer is created here, so no optimizer state survives between
    calls. mean_loss is accumulated on the device in float64 and read back once (no per-step synchronisation).
    """
    set_local_training_mode(model)
    optimizer = make_optimizer(model, learning_rate, config)
    loss_function = nn.CrossEntropyLoss()

    num_batches = 0
    num_processed = 0
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)  # diagnostic accumulator, stays on device
    for _ in range(config.local_epochs):
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)  # plain SGD step
            loss = loss_function(model(inputs), targets)
            loss.backward()
            optimizer.step()
            batch_size = len(targets)
            num_batches += 1
            num_processed += batch_size
            loss_sum += loss.detach().double() * batch_size  # diagnostic only

    for module in model.modules():
        if isinstance(module, BATCHNORM_TYPES) and module.training:
            raise RuntimeError("a BatchNorm layer was switched to training mode during local training")
    return num_batches, loss_sum.item() / num_processed


@torch.no_grad()
def compute_client_delta(global_model: nn.Module, local_model: nn.Module, trainable_names: tuple[str, ...],
                         keep_on_device: bool = False) -> dict[str, torch.Tensor]:
    """Delta W_i[name] = W_local[name] - W_global[name] for the trainable names, as new detached tensors.

    ``keep_on_device=True`` subtracts on the device of the local model and leaves the result there; the default
    subtracts on the CPU. Elementwise subtraction is exactly rounded, so both give the same values.
    """
    global_parameters = dict(global_model.named_parameters())
    local_parameters = dict(local_model.named_parameters())
    delta = {}
    for name in trainable_names:
        local_value = local_parameters[name].detach()
        if keep_on_device:
            global_value = global_parameters[name].detach().to(local_value.device)
        else:
            local_value, global_value = local_value.to("cpu"), global_parameters[name].detach().to("cpu")
        delta[name] = local_value - global_value
    return delta


def zero_client_update(global_model: nn.Module, trainable_names: tuple[str, ...],
                       device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """The update of an empty slot: exact zeros with the keys, shapes and dtypes of a real update."""
    global_parameters = dict(global_model.named_parameters())
    return {name: torch.zeros_like(global_parameters[name], device=device) for name in trainable_names}


def train_client(
    global_model: nn.Module,
    partition: ModelPartition,
    train_set: Dataset,
    client_indices: np.ndarray,
    round_index: int,
    training_seed: int,
    config: LocalTrainingConfig = LocalTrainingConfig(),
    device: str | torch.device = "cpu",
    num_workers: int = 0,
    multiprocessing_context: str | None = None,
    keep_delta_on_device: bool = False,
) -> LocalTrainingResult:
    """Train one client slot from the global model W^(t-1) and return its uncompressed update Delta W_i.

    ``global_model`` is never modified: training runs on a deep copy moved to ``device``. The model's
    ``requires_grad`` flags must match ``partition.trainable`` (see ``apply_partition``). ``training_seed`` is the
    common local-training seed of this slot and round (``RunSeedPlan.local_training_seed``).
    """
    _check_trainable_flags(global_model, partition)
    _check_num_workers(num_workers)
    learning_rate = round_learning_rate(round_index, config.base_learning_rate, config.learning_rate_decay)
    shuffle_seed, augmentation_seed, padding_seed = client_loader_seeds(training_seed)

    if len(client_indices) == 0:
        return LocalTrainingResult(
            delta=zero_client_update(global_model, partition.trainable_parameter_names,
                                     device=device if keep_delta_on_device else "cpu"),
            num_examples=0, num_batches=0, mean_loss=None, learning_rate=learning_rate,
        )

    started = time.perf_counter()  # observational only
    asynchronous_copies = torch.device(device).type == "cuda"  # pinned batches + non-blocking copies on CUDA
    local_model = copy.deepcopy(global_model).to(device)
    dataloader = make_client_dataloader(train_set, client_indices, config.batch_size, shuffle_seed, augmentation_seed,
                                        padding_seed, num_workers, multiprocessing_context,
                                        pin_memory=asynchronous_copies)
    with seeded_cpu_rng(augmentation_seed):
        num_batches, mean_loss = run_local_sgd(local_model, dataloader, learning_rate, config, device,
                                               non_blocking=asynchronous_copies)
    trained = time.perf_counter()
    delta = compute_client_delta(global_model, local_model, partition.trainable_parameter_names,
                                 keep_on_device=keep_delta_on_device)
    return LocalTrainingResult(
        delta=delta, num_examples=len(client_indices), num_batches=num_batches, mean_loss=mean_loss,
        learning_rate=learning_rate,
        timings={"local_training_s": trained - started, "client_delta_s": time.perf_counter() - trained},
    )


def _check_num_workers(num_workers: int) -> None:
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError(f"num_workers must be a non-negative integer, got {num_workers!r}")


def _check_trainable_flags(model: nn.Module, partition) -> None:
    flagged = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if flagged != set(partition.trainable_parameter_names):
        raise ValueError("the model's requires_grad flags do not match the partition; call apply_partition first")
