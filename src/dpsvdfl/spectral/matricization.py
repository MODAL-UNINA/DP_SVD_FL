"""Matricisation of trainable tensors into blocks W_l in R^(m_l x n_l) (paper Sec. III-A)."""

import torch


def matricize(tensor: torch.Tensor) -> torch.Tensor:
    """Return the matrix block W_l associated with a trainable tensor.

    - convolutional kernel [c_out, c_in, h, w] -> [c_out, c_in * h * w]
      (row index = output channel, columns in row-major (c_in, h, w) order);
    - linear weight [m, n] -> unchanged;
    - bias vector [m] -> [m, 1].

    The result is a view whenever the input is contiguous, so no data is copied.
    """
    if tensor.ndim == 4:
        c_out = tensor.shape[0]
        return tensor.reshape(c_out, -1)
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 1:
        return tensor.reshape(-1, 1)
    raise ValueError(f"Cannot matricise a tensor of shape {tuple(tensor.shape)}")


def matrix_shape(tensor_shape: tuple[int, ...] | torch.Size) -> tuple[int, int]:
    """Return (m_l, n_l) of the block obtained by `matricize` from a tensor of the given shape."""
    shape = tuple(tensor_shape)
    if len(shape) == 4:
        c_out, c_in, h, w = shape
        return c_out, c_in * h * w
    if len(shape) == 2:
        return shape[0], shape[1]
    if len(shape) == 1:
        return shape[0], 1
    raise ValueError(f"Cannot matricise a tensor of shape {shape}")
