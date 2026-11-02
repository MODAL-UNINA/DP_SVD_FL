"""Small file helpers of the experiment driver: atomic writes, append-only JSONL logs and fingerprints."""

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from torch import nn


def write_bytes_atomic(path: str | Path, writer: Callable[[BinaryIO], None]) -> None:
    """Write through ``writer(file)`` to a temporary file in the same directory, fsync, then rename onto ``path``."""
    path = Path(path)
    temporary = path.with_name(f".tmp.{path.name}.{os.getpid()}")
    try:
        with open(temporary, "wb") as file:
            writer(file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json_atomic(path: str | Path, content: dict) -> None:
    write_bytes_atomic(path, lambda file: file.write((json.dumps(content, indent=2) + "\n").encode("utf-8")))


def read_json(path: str | Path) -> dict:
    with open(path) as file:
        return json.load(file)


def append_jsonl(path: str | Path, record: dict) -> None:
    """Append one record and fsync, so that a later checkpoint never commits a round whose line is lost."""
    with open(path, "a") as file:
        file.write(json.dumps(record) + "\n")
        file.flush()
        os.fsync(file.fileno())


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as file:
        return [json.loads(line) for line in file if line.strip()]


def truncate_jsonl(path: str | Path, last_committed_round: int, round_key: str = "round") -> None:
    """Keep exactly the records of rounds 1..last_committed_round (resume after an interruption).

    Records of uncommitted rounds and an incomplete last line are removed. The kept records must be exactly rounds
    1..last_committed_round, in order; anything else raises instead of being repaired.
    """
    path = Path(path)
    if not path.exists():
        if last_committed_round > 0:
            raise RuntimeError(f"{path} is missing although {last_committed_round} rounds are committed")
        return
    lines = path.read_text().split("\n")
    kept = []
    for line in lines[:-1]:  # the text after the last newline, if any, is an interrupted write
        if line.strip():
            record = json.loads(line)
            if record[round_key] <= last_committed_round:
                kept.append(record)
    if [record[round_key] for record in kept] != list(range(1, last_committed_round + 1)):
        raise RuntimeError(f"{path}: the committed records are not exactly rounds 1..{last_committed_round}")
    content = "".join(json.dumps(record) + "\n" for record in kept).encode("utf-8")
    write_bytes_atomic(path, lambda file: file.write(content))


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Portable copy of the model state: detached CPU tensors (parameters and buffers)."""
    return {name: value.detach().to("cpu", copy=True) for name, value in model.state_dict().items()}


def state_dict_fingerprint(state_dict: dict[str, torch.Tensor]) -> str:
    """SHA-256 over names, dtypes, shapes and raw bytes of a state dict (identifies W^(0) exactly)."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().to("cpu").contiguous()
        digest.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}|".encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def arrays_fingerprint(arrays: list[np.ndarray]) -> str:
    """SHA-256 of a partition or a sampling schedule (sizes and contents, in order); machine independent."""
    digest = hashlib.sha256()
    for array in arrays:
        values = np.asarray(array, dtype="<i8")
        digest.update(np.array([len(values)], dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()
