"""Spectral bases, projection P_l and reconstruction R_l (paper Sec. III-B, Algorithm 1 lines 3-6).

For a projected layer with previously released weights W_l = U diag(s) V^T:

    U_k = U[:, :k_l],  V_k = V[:, :k_l],
    P_l(X) = U_k^T X V_k   in R^(k_l x k_l)     (encoding of a client update block)
    R_l(S) = U_k S V_k^T   in R^(m_l x n_l)     (decoding of the released coefficients)

U_k and V_k have orthonormal columns, so R_l is an isometry (||R_l(S)||_F = ||S||_F) and R_l P_l is the orthogonal
projector Pi_l (paper Lemma 1). A basis records its rank source: "adaptive" (the rank rule of the current round,
DP-SVD-FL) or "initial_fixed" (k_l^(0) selected once on W^(0), the fixed-rank spectral ablation).
"""

from dataclasses import dataclass

import torch

from dpsvdfl.spectral.rank_rule import GAMMA, RHO_MAX, TAU, RankRuleResult, select_rank

BASIS_SPECTRAL = "spectral"
RANK_ADAPTIVE = "adaptive"
RANK_INITIAL_FIXED = "initial_fixed"
RANK_SOURCES = (RANK_ADAPTIVE, RANK_INITIAL_FIXED)

# The SVD is computed in double precision so that the rank rule, which compares ratios of singular values, is
# evaluated on an accurate spectrum of W_l. The bases are then cast back to the dtype of the weights (float32).
SVD_DTYPE = torch.float64


@dataclass(frozen=True)
class ProjectionBasis:
    """Basis of one projected layer.

    U_k:             [m, k] left singular vectors (orthonormal columns; dtype/device of the weight matrix).
    V_k:             [n, k] right singular vectors (columns of V, not rows of V^T).
    singular_values: all min(m, n) singular values s_1 >= s_2 >= ... (float64) of the current weight.
    rank_rule:       outcome of the rank rule for an adaptive rank (k = rank_rule.selected_rank); None for a fixed rank.
    basis_source:    always "spectral" in this release.
    rank_source:     "adaptive" or "initial_fixed".
    """

    U_k: torch.Tensor
    V_k: torch.Tensor
    singular_values: torch.Tensor
    rank_rule: RankRuleResult | None
    basis_source: str = BASIS_SPECTRAL
    rank_source: str = RANK_ADAPTIVE

    def __post_init__(self) -> None:
        if self.rank_source not in RANK_SOURCES:
            raise ValueError(f"rank_source must be one of {RANK_SOURCES}, got {self.rank_source!r}")
        if (self.rank_rule is not None) != (self.rank_source == RANK_ADAPTIVE):
            raise ValueError("a rank-rule result is present exactly for an adaptive rank")
        if self.U_k.ndim != 2 or self.V_k.ndim != 2 or self.U_k.shape[1] != self.V_k.shape[1]:
            raise ValueError(f"U_k {tuple(self.U_k.shape)} and V_k {tuple(self.V_k.shape)} must both have k columns")

    @property
    def rank(self) -> int:
        return self.rank_rule.selected_rank if self.rank_rule is not None else int(self.U_k.shape[1])


def reduced_svd(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(U, s, V) with matrix = U diag(s) V^T; U: [m, r], s: [r], V: [n, r], r = min(m, n).

    ``torch.linalg.svd`` returns Vh = V^T; it is transposed here so that the columns of V are the right singular
    vectors, as in the paper.
    """
    if matrix.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {tuple(matrix.shape)}")
    U, s, Vh = torch.linalg.svd(matrix, full_matrices=False)
    V = Vh.transpose(-2, -1)
    return U, s, V


@torch.no_grad()
def compute_projection_basis(
    weight_matrix: torch.Tensor,
    tau: float = TAU,
    rho_max: float = RHO_MAX,
    gamma: float = GAMMA,
) -> ProjectionBasis:
    """One float64 SVD of W_l, the rank rule, and the top-k singular vectors (Algorithm 1, lines 4-6)."""
    m, n = weight_matrix.shape
    U, s, V = reduced_svd(weight_matrix.to(SVD_DTYPE))
    rank_rule = select_rank(s, m, n, tau=tau, rho_max=rho_max, gamma=gamma)
    k = rank_rule.selected_rank
    U_k = U[:, :k].to(weight_matrix.dtype)
    V_k = V[:, :k].to(weight_matrix.dtype)
    return ProjectionBasis(U_k=U_k, V_k=V_k, singular_values=s, rank_rule=rank_rule)


def check_fixed_rank(k: int, matrix_shape: tuple[int, int], name: str = "layer") -> None:
    """A fixed rank must be an integer (not bool) with 1 <= k <= min(m, n)."""
    m, n = matrix_shape
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"{name}: a fixed rank must be an integer, got {k!r}")
    if not 1 <= k <= min(m, n):
        raise ValueError(f"{name}: a fixed rank must satisfy 1 <= k <= min(m, n) = {min(m, n)}, got {k}")


@torch.no_grad()
def compute_spectral_basis_at_fixed_rank(weight_matrix: torch.Tensor, k: int) -> ProjectionBasis:
    """Top-k singular vectors of the CURRENT weight for a rank fixed beforehand; the rank rule is not called.

    Same float64 SVD and truncation as ``compute_projection_basis``: with k equal to the adaptive rank both return
    identical U_k and V_k.
    """
    if weight_matrix.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {tuple(weight_matrix.shape)}")
    check_fixed_rank(k, tuple(weight_matrix.shape))
    U, s, V = reduced_svd(weight_matrix.to(SVD_DTYPE))
    U_k = U[:, :k].to(weight_matrix.dtype)
    V_k = V[:, :k].to(weight_matrix.dtype)
    return ProjectionBasis(U_k=U_k, V_k=V_k, singular_values=s, rank_rule=None, rank_source=RANK_INITIAL_FIXED)


def project_update(delta_w: torch.Tensor, U_k: torch.Tensor, V_k: torch.Tensor) -> torch.Tensor:
    """P_l(X) = U_k^T X V_k: coefficients of an [m, n] matrix in the spectral basis, shape [k, k]."""
    _check_basis(U_k, V_k, delta_w.shape)
    return U_k.T @ delta_w @ V_k


def reconstruct_update(coefficients: torch.Tensor, U_k: torch.Tensor, V_k: torch.Tensor) -> torch.Tensor:
    """R_l(S) = U_k S V_k^T: the [m, n] matrix represented by a [k, k] coefficient matrix."""
    m, k = U_k.shape
    n = V_k.shape[0]
    _check_basis(U_k, V_k, (m, n))
    if tuple(coefficients.shape) != (k, k):
        raise ValueError(f"expected coefficients of shape ({k}, {k}), got {tuple(coefficients.shape)}")
    return U_k @ coefficients @ V_k.T


def _check_basis(U_k: torch.Tensor, V_k: torch.Tensor, matrix_shape: tuple[int, ...] | torch.Size) -> None:
    m, n = matrix_shape
    if U_k.ndim != 2 or V_k.ndim != 2 or U_k.shape[1] != V_k.shape[1]:
        raise ValueError(f"U_k {tuple(U_k.shape)} and V_k {tuple(V_k.shape)} must both have k columns")
    if U_k.shape[0] != m or V_k.shape[0] != n:
        raise ValueError(f"basis shapes U_k {tuple(U_k.shape)}, V_k {tuple(V_k.shape)} do not match a ({m}, {n}) matrix")


def representation_dimension(projected_ranks: dict[str, int], unprojected_matrix_shapes: dict[str, tuple[int, int]]) -> int:
    """D = sum_{projected} k_l^2 + sum_{unprojected} m_l n_l (paper Sec. III-C)."""
    projected = sum(k * k for k in projected_ranks.values())
    unprojected = sum(m * n for m, n in unprojected_matrix_shapes.values())
    return projected + unprojected


def full_dimension(trainable_matrix_shapes: dict[str, tuple[int, int]]) -> int:
    """D_full = sum_{trainable} m_l n_l (paper Sec. III-C)."""
    return sum(m * n for m, n in trainable_matrix_shapes.values())
