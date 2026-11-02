import pytest
import torch

from dpsvdfl.models.partitions import ModelPartition
from dpsvdfl.spectral.matricization import matricize
from dpsvdfl.spectral.projection import (
    compute_projection_basis,
    compute_spectral_basis_at_fixed_rank,
    project_update,
    reconstruct_update,
)
from dpsvdfl.spectral.rank_rule import select_rank
from dpsvdfl.spectral.representation import (
    build_round_spectral_state,
    encode_client_update,
)
from dpsvdfl.federated.server_update import reconstruct_server_update


def test_gap_branch_takes_smallest_maximiser():
    s = torch.tensor([8.0, 4.0, 2.0, 1.0, 0.5] + [0.4] * 395, dtype=torch.float64)
    result = select_rank(s, 400, 500)
    assert result.branch == "gap" and result.selected_rank == 1  # ratios 2, 2, 2, 2: ties -> smallest index


def test_gap_rejected_by_rho_max_falls_back_to_energy():
    s = torch.tensor([1.0] * 30 + [0.1] * 10, dtype=torch.float64)  # k* = 30, 30 / sqrt(40 * 40) >= 0.05
    result = select_rank(s, 40, 40)
    assert result.gap_rank == 30 and result.max_gap_ratio == pytest.approx(10.0)
    assert result.branch == "energy" and result.selected_energy_fraction >= 0.95


def test_zero_spectrum_fallback():
    result = select_rank(torch.zeros(5, dtype=torch.float64), 5, 7)
    assert result.branch == "fallback" and result.selected_rank == 1


def test_energy_rank_is_minimal():
    s = torch.linspace(10, 1, 50, dtype=torch.float64)
    result = select_rank(s, 50, 60, tau=1e9)
    k = result.selected_rank
    energy = (s ** 2).cumsum(0) / (s ** 2).sum()
    assert energy[k - 1] >= 0.95 and energy[k - 2] < 0.95


def test_bases_are_orthonormal_and_reconstruction_is_an_isometry():
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(64, 288, generator=generator)
    basis = compute_projection_basis(weight)
    k = basis.rank
    eye = torch.eye(k)
    assert torch.allclose(basis.U_k.T @ basis.U_k, eye, atol=1e-5)
    assert torch.allclose(basis.V_k.T @ basis.V_k, eye, atol=1e-5)
    S = torch.randn(k, k, generator=generator)
    assert torch.linalg.norm(reconstruct_update(S, basis.U_k, basis.V_k)) == pytest.approx(torch.linalg.norm(S), rel=1e-5)
    X = torch.randn(64, 288, generator=generator)
    P = project_update(X, basis.U_k, basis.V_k)
    assert torch.linalg.norm(P) <= torch.linalg.norm(X) + 1e-5  # non-expansive (Lemma 1)
    projected = reconstruct_update(P, basis.U_k, basis.V_k)
    again = reconstruct_update(project_update(projected, basis.U_k, basis.V_k), basis.U_k, basis.V_k)
    assert torch.allclose(projected, again, atol=1e-5)  # Pi is idempotent


def test_fixed_rank_basis_equals_adaptive_basis_at_the_same_rank():
    weight = torch.randn(32, 72, generator=torch.Generator().manual_seed(1))
    adaptive = compute_projection_basis(weight)
    fixed = compute_spectral_basis_at_fixed_rank(weight, adaptive.rank)
    assert torch.equal(adaptive.U_k, fixed.U_k) and torch.equal(adaptive.V_k, fixed.V_k)


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, 3, bias=False)
        self.fc = torch.nn.Linear(8, 3)


def test_encode_decode_layout_and_dimension():
    torch.manual_seed(0)
    model = _Toy()
    partition = ModelPartition(projected=("conv.weight",), unprojected=("fc.weight", "fc.bias"), frozen=())
    state = build_round_spectral_state(model, partition)
    k = state.ranks["conv.weight"]
    assert state.dimension == k * k + 8 * 3 + 3  # D = sum k^2 + sum m n
    delta = {name: torch.randn_like(p) for name, p in model.named_parameters()}
    s = encode_client_update(delta, partition, state)
    assert s.shape == (state.dimension,)
    # the unprojected blocks are transmitted unchanged, after the projected coefficients
    assert torch.equal(s[k * k:k * k + 24], delta["fc.weight"].reshape(-1))
    decoded = reconstruct_server_update(s, partition, state)
    basis = state.bases["conv.weight"]
    expected = basis.U_k @ basis.U_k.T @ matricize(delta["conv.weight"]) @ basis.V_k @ basis.V_k.T
    assert torch.allclose(matricize(decoded["conv.weight"]), expected, atol=1e-5)
    assert torch.equal(decoded["fc.bias"], delta["fc.bias"])
