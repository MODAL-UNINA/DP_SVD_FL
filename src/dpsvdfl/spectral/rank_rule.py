"""Public rank-selection rule of DP-SVD-FL (paper Sec. III-B and Definition 1).

Notation follows the paper: the singular values of a projected weight matrix are s_1 >= s_2 >= ... >= 0 with 1-based
indices. In code, the Python list ``s`` stores s_j at position ``s[j - 1]``.

For an m x n matrix the rule selects

    k = k*                      if r_{k*} = s_{k*} / s_{k*+1} >= tau  and  k* / sqrt(m n) < rho_max     (gap branch)
    k = min{k : sum_{j<=k} s_j^2 / sum_j s_j^2 >= gamma}                otherwise                      (energy branch)
    k = 1                       for an all-zero matrix                                                  (fallback)

where k* is the SMALLEST index attaining the maximum consecutive ratio over J = {j < min(m, n) : s_{j+1} > 0}.
The parameters are public and fixed a priori (paper: tau = 2, rho_max = 0.05, gamma = 0.95). The energy criterion
refers to the spectrum of the WEIGHTS, not of the client updates (paper Remark 1).
"""

import math
from dataclasses import dataclass

import torch

# Values used in every experiment of the paper (Table II).
TAU = 2.0
RHO_MAX = 0.05
GAMMA = 0.95

# The branch names recorded for every projected layer (released round diagnostics, initial rank snapshot).
GAP_BRANCH = "gap"
ENERGY_BRANCH = "energy"
FALLBACK_BRANCH = "fallback"  # zero spectrum only
RANK_RULE_BRANCHES = (GAP_BRANCH, ENERGY_BRANCH, FALLBACK_BRANCH)


@dataclass(frozen=True)
class RankRuleResult:
    """Outcome of the rank rule for one projected layer.

    selected_rank:     k_l, the rank actually used.
    branch:            "gap", "energy" or "fallback".
    gap_rank:          k*, smallest j in J attaining max s_j / s_(j+1); None if J is empty.
    max_gap_ratio:     s_k* / s_(k*+1); None if J is empty.
    gap_rank_fraction: k* / sqrt(m n), compared with rho_max; None if J is empty.
    energy_rank:       smallest kappa with sum_{j<=kappa} s_j^2 / sum_j s_j^2 >= gamma;
                       None if the spectrum is zero. With gamma = 0.95 this is k95.
    selected_energy_fraction: sum_{j<=k_l} s_j^2 / sum_j s_j^2 (NaN if the spectrum is zero).
    """

    selected_rank: int
    branch: str
    gap_rank: int | None
    max_gap_ratio: float | None
    gap_rank_fraction: float | None
    energy_rank: int | None
    selected_energy_fraction: float


def largest_gap(s: list[float]) -> tuple[int, float] | None:
    """Return (k*, s_k* / s_(k*+1)) over J = {j : s_(j+1) > 0}, or None if J is empty.

    Ties are broken towards the smaller rank: a later j replaces the current
    maximiser only if its ratio is strictly larger.
    """
    best_rank = None
    best_ratio = -math.inf
    for j in range(1, len(s)):  # j = 1, ..., r - 1 (1-based), so that s_(j+1) exists
        s_j = s[j - 1]
        s_next = s[j]  # s_(j+1)
        if s_next > 0:
            ratio = s_j / s_next
            if ratio > best_ratio:
                best_rank = j
                best_ratio = ratio
    if best_rank is None:
        return None
    return best_rank, best_ratio


def energy_rank(s: list[float], gamma: float) -> int | None:
    """Return min{kappa : sum_{j<=kappa} s_j^2 / sum_j s_j^2 >= gamma}, or None if sum_j s_j^2 = 0."""
    squared = [value * value for value in s]
    total_energy = sum(squared)
    if total_energy == 0:
        return None
    # The running sum accumulates in the same order as `total_energy`, so at
    # kappa = r it equals total_energy exactly and the loop always returns.
    cumulative_energy = 0.0
    for kappa in range(1, len(s) + 1):
        cumulative_energy += squared[kappa - 1]
        if cumulative_energy / total_energy >= gamma:
            return kappa
    raise AssertionError("unreachable: the full spectrum retains all of the energy")


def energy_fraction(s: list[float], k: int) -> float:
    """Return sum_{j<=k} s_j^2 / sum_j s_j^2 (NaN if the spectrum is zero)."""
    squared = [value * value for value in s]
    total_energy = sum(squared)
    if total_energy == 0:
        return math.nan
    return sum(squared[:k]) / total_energy


def select_rank(
    singular_values: torch.Tensor,
    m: int,
    n: int,
    tau: float = TAU,
    rho_max: float = RHO_MAX,
    gamma: float = GAMMA,
) -> RankRuleResult:
    """Apply the DP-SVD-FL rank rule to the spectrum of an m x n matrix.

    1. Gap branch:    if J is non-empty, s_k*/s_(k*+1) >= tau and k*/sqrt(m n) < rho_max,
                      then k = k*.
    2. Energy branch: otherwise, if sum_j s_j^2 > 0, k = min{kappa : energy fraction >= gamma}.
    3. Fallback:      otherwise (zero spectrum), k = 1.
    """
    if singular_values.ndim != 1:
        raise ValueError("singular_values must be a 1-D tensor")
    if singular_values.numel() != min(m, n):
        raise ValueError(
            f"expected min(m, n) = {min(m, n)} singular values, got {singular_values.numel()}"
        )

    # Work with Python floats (double precision) so that every comparison is explicit.
    s = singular_values.detach().to(device="cpu", dtype=torch.float64).tolist()
    if any(value < 0 for value in s):
        raise ValueError("singular values must be non-negative")
    if any(s[i] < s[i + 1] for i in range(len(s) - 1)):
        raise ValueError("singular values must be sorted in non-increasing order")

    gap = largest_gap(s)
    k_energy = energy_rank(s, gamma)

    gap_rank = None
    max_gap_ratio = None
    gap_rank_fraction = None
    if gap is not None:
        gap_rank, max_gap_ratio = gap
        gap_rank_fraction = gap_rank / math.sqrt(m * n)

    if gap is not None and max_gap_ratio >= tau and gap_rank_fraction < rho_max:
        selected_rank = gap_rank
        branch = GAP_BRANCH
    elif k_energy is not None:
        selected_rank = k_energy
        branch = ENERGY_BRANCH
    else:
        selected_rank = 1
        branch = FALLBACK_BRANCH

    return RankRuleResult(
        selected_rank=selected_rank,
        branch=branch,
        gap_rank=gap_rank,
        max_gap_ratio=max_gap_ratio,
        gap_rank_fraction=gap_rank_fraction,
        energy_rank=k_energy,
        selected_energy_fraction=energy_fraction(s, selected_rank),
    )
