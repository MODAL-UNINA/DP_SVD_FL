"""Client-level privacy accountant of DP-SVD-FL: RDP upper bounds of the Poisson-subsampled Gaussian mechanism.

This is the reference accountant ``dpsvdfl-sgm-rdp-certified-v2`` of the paper (Sec. IV-C;
Supplementary Sec. S8).

Mechanism (paper Theorem 2). Conditionally on the released history, every round of the Gaussian mechanism is a
Poisson-sampled Gaussian mechanism with L2 sensitivity C and noise standard deviation sigma C. After rescaling by 1/C it
is the Sampled Gaussian Mechanism (SGM) with sensitivity 1, noise multiplier sigma and sampling rate q, under add/remove
adjacency of ONE client's whole data. The accountant depends only on (q, sigma, T, delta): NOT on C, D or D_full, the
ranks or bases, the model, or the realised |S_t| (the Poisson probability q is used, never an empirical rate).

Per-round RDP (both divergence directions; Mironov, Talwar, Zhang 2019):
    Q = N(0, 1) (in units of sigma),   r(z) = (1 - q) + q exp(z / sigma - 1 / (2 sigma^2))   (= P / Q),
    I(beta) = E_{Z ~ N(0,1)}[r(Z)^beta],
    forward  D_alpha(P || Q) = log I(alpha)     / (alpha - 1),
    reverse  D_alpha(Q || P) = log I(1 - alpha) / (alpha - 1),
    rho(alpha) = max(forward, reverse).
Every value used here is an UPPER bound on the exact quantity (never an estimate):

  * q = 0: P = Q, rho = 0 exactly. q = 1: I(beta) = exp(beta (beta - 1) / (2 sigma^2)) exactly, so both directions
    equal alpha / (2 sigma^2).
  * forward direction, integer alpha >= 2: the exact binomial sum (all terms positive)
        I(alpha) = sum_{j=0}^{alpha} binom(alpha, j) (1-q)^(alpha-j) q^j exp((j^2 - j) / (2 sigma^2)).
    The reverse direction has no finite expansion (a negative power of a mixture) and is integrated as below.
  * every other (order, direction): a certified bracket [L, U] of J = int exp(g(z)) dz, I = J / sqrt(2 pi), with
        g(z) = beta log r(z) - z^2 / 2,   g'(z) = (beta / sigma) s(z) - z,   g''(z) = kappa s (1 - s) - 1,
        s(z) = expit(z / sigma - 1 / (2 sigma^2) + logit q),   kappa = beta / sigma^2.
    (i) Curvature pieces. s(1 - s) <= 1/4; where |kappa| s(1 - s) <= 1 the curvature lies in [-1, 0] (kappa > 0) or
        [-2, -1] (kappa < 0). If |kappa| > 4 the complementary "band" {|kappa| s(1-s) > 1} is the explicit interval
        |logit s| < log((1 + w) / (1 - w)), w = sqrt(1 - 4 / |kappa|), mapped to z in closed form (and widened by a
        relative 1e-9 so that rounding cannot move a point of the band outside it). Inside the band g'' lies in
        [-1, kappa/4 - 1] (kappa > 0) or [kappa/4 - 1, -1] (kappa < 0). g is CONCAVE outside the band; for
        kappa > 0 it is convex inside it: this is where the forward integrand can be BIMODAL. At
        most three pieces: (-inf, b-], band, [b+, +inf); no mode can be missed because no mode is searched globally.
    (ii) Cells. On a cell [m - eta, m + eta] with Gmin <= g'' <= Gmax (the piece bounds), Taylor's theorem gives
            g(m + y) <= g(m) + g'(m) y + max(0, Gmax) eta^2 / 2,   g(m + y) >= g(m) + g'(m) y + min(0, Gmin) eta^2 / 2,
        and int_{-eta}^{eta} exp(a y) dy = 2 eta sinh(a eta) / (a eta) in closed form. eta is chosen per piece so
        that (max(0, Gmax) - min(0, Gmin)) eta^2 / 2 = relative_tolerance: each cell's upper / lower ratio is then
        exp(relative_tolerance). The upper-bound cells are widened by a few ulps so that they cover their interval
        despite the rounding of the cell centres.
    (iii) Where cells are used. On a concave piece, g' is monotone, so its maximiser is found by bisection, and so are
        the two points where g falls SUPERLEVEL_DEPTH nats below that maximum; that superlevel interval is covered by
        cells. Outside it (still inside the concave piece) the integrand is bounded by the tangent line at the
        interval end (concavity), which integrates in closed form; an infinite tail needs a strictly negative
        (resp. positive) slope, otherwise the computation is REFUSED. The band is always covered by cells.
    (iv) U = sum of the cell upper bounds + the tangent bounds; L = sum of the cell lower bounds. U is returned; the
        certified log gap log U - log L is recorded (tightness, never validity). A floating-point allowance
        (1e-12 + 64 eps_mach x the magnitude of the evaluated exponents) is added to log U: the bounds are exact
        theorems in real arithmetic; the float64 evaluation is covered by that argued allowance, not by interval
        arithmetic (stated as a limitation).

Validation, conversion and non-negativity are SEPARATE steps:
    1. RDP-curve validation: every forward / reverse / maximum value must be finite and >= 0 (a negative upper bound on
       a Renyi divergence is an anomaly and raises AccountantError; it is never clamped);
    2. conversion (Balle et al. 2020, Thm 21), minimised over the order grid:
           raw epsilon = min_alpha [T rho(alpha) + log((alpha - 1) / alpha) - (log delta + log alpha) / (alpha - 1)];
       it can be negative for tiny T rho and large delta, which is reported as raw_epsilon;
    3. epsilon = max(0, raw epsilon): a negative value certifies (0, delta)-DP a fortiori; epsilon_clamped_to_zero
       records when this happened.
The basic conversion T rho(alpha) + log(1 / delta) / (alpha - 1) is a diagnostic only. delta = None means 1 / N and is
refused at configuration time when it would not lie in (0, 1) (N = 1).

Threat model: unchanged (the guarantee covers observers of the released aggregates and their post-processing, not of
the sampled sets or of the randomness). What the randomness of a run guarantees in practice depends on the privacy mode
(dpsvdfl.privacy.modes): with benchmark_reproducible seeds the epsilon is the NOMINAL budget of the ideal mechanism.
"""

import math
import platform
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import scipy
from scipy.special import expit, gammaln, logsumexp

if TYPE_CHECKING:
    from dpsvdfl.dp.mechanism import DPMechanismConfig

ACCOUNTANT_FAMILY = "RDP of the Poisson-subsampled Gaussian mechanism, add/remove adjacency, both directions"
ACCOUNTANT_VERSION = "dpsvdfl-sgm-rdp-certified-v2"
ACCOUNTANT_LABEL = "Poisson SGM, add/remove adjacency, bidirectional RDP"  # the label every summary carries
CONVERSION = ("Balle et al. 2020 (Thm 21): epsilon = min over the order grid of T rho(alpha) + log((alpha - 1) / alpha) "
              "- (log delta + log alpha) / (alpha - 1); then max(0, .)")
INTEGRATION_METHOD = ("exact binomial sum (forward, integer orders); certified bracket elsewhere: second-order Taylor "
                      "cells with explicit curvature bounds on at most three pieces + concave tangent tail bounds")
VALIDITY_LIMITS = (
    "upper bounds on the RDP of the Poisson-subsampled Gaussian mechanism with sensitivity 1 (after scaling by 1 / C), "
    "add/remove adjacency of one client, adaptive composition of T identical rounds",
    "exact in real arithmetic; float64 evaluation covered by an argued rounding allowance (no interval arithmetic)",
    "orders restricted to the grid: a finer or wider grid can only give a smaller (still valid) epsilon",
    "the ideal mechanism with real-valued Gaussian noise: floating-point sampling is not covered",
)

DEFAULT_RELATIVE_TOLERANCE = 1e-7  # per-cell upper / lower log ratio; the certified log gap of a curve value
SUPERLEVEL_DEPTH = 50.0  # nats below a concave piece's maximum covered by cells (the rest: tangent bounds)
MAX_CELLS_PER_PIECE = 2_000_000  # beyond this the cells grow (still a valid bound) and the gap is recorded
GAP_FALLBACK_THRESHOLD = 1e-4  # a recorded log gap above this is listed among the conservative fallbacks
_BAND_WIDENING = 1e-9
_BISECTION_STEPS = 200
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_EPS = float(np.finfo(np.float64).eps)

CLOSED_FORM = "closed_form"  # exact (q = 0, q = 1, forward at integer orders)
CERTIFIED_INTEGRATION = "certified_integration"


class AccountantError(ArithmeticError):
    """No reliable bound can be computed for this input (an anomaly is refused, never clamped or approximated)."""


def default_rdp_orders() -> tuple[float, ...]:
    """The 221 orders of the paper (Supplementary Sec. S8.1): {1.02, 1.04, ..., 2.98} U {3.0, 3.5, ..., 63.5}.

    Built from integer ratios so that every order is the double closest to its decimal value.
    """
    fine = tuple((50 + k) / 50 for k in range(1, 100))  # 1.02 ... 2.98 (99 orders)
    coarse = tuple((6 + k) / 2 for k in range(0, 122))  # 3.0 ... 63.5 (122 orders)
    return fine + coarse


def check_delta_resolution(num_client_slots: int, delta) -> float:
    """The delta of a run (explicit, or 1 / N for None); refused at configuration time if not in (0, 1)."""
    if delta is None:
        if isinstance(num_client_slots, bool) or not isinstance(num_client_slots, (int, np.integer)):
            raise TypeError(f"num_client_slots must be an integer, got {num_client_slots!r}")
        if num_client_slots < 2:
            raise ValueError(f"delta = None means delta = 1 / N, which is not in (0, 1) for N = {num_client_slots}: "
                             "pass an explicit delta in (0, 1) for a single-slot run")
        return 1.0 / num_client_slots
    _check_delta(delta)
    return float(delta)


@dataclass(frozen=True)
class PrivacyAccountantConfig:
    """Parameters of the accountant (defaults: the main configuration of the paper, Table II).

    delta = None means delta = 1 / N; refused here when N = 1.
    """

    sampling_rate: float = 0.1  # q
    noise_multiplier: float = 0.95  # sigma
    num_rounds: int = 300  # T
    num_client_slots: int = 500  # N
    delta: float | None = None

    def __post_init__(self) -> None:
        _check_real(self.sampling_rate, "sampling_rate")
        if not 0.0 <= self.sampling_rate <= 1.0:
            raise ValueError(f"sampling_rate must be in [0, 1], got {self.sampling_rate}")
        _check_noise_multiplier(self.noise_multiplier)
        _check_integer(self.num_rounds, "num_rounds", minimum=0)
        _check_integer(self.num_client_slots, "num_client_slots", minimum=1)
        check_delta_resolution(self.num_client_slots, self.delta)

    @property
    def target_delta(self) -> float:
        """The delta actually used: the explicit value, or 1 / N."""
        return 1.0 / self.num_client_slots if self.delta is None else float(self.delta)

    @classmethod
    def from_mechanism_config(
        cls, mechanism_config: "DPMechanismConfig", num_rounds: int, delta: float | None = None
    ) -> "PrivacyAccountantConfig":
        """Accountant for T rounds of a Gaussian mechanism configuration: q, sigma and N are taken over; C is not needed."""
        return cls(
            sampling_rate=mechanism_config.sampling_rate,
            noise_multiplier=mechanism_config.noise_multiplier,
            num_rounds=num_rounds,
            num_client_slots=mechanism_config.num_client_slots,
            delta=delta,
        )


@dataclass(frozen=True)
class RDPBounds:
    """Certified per-round bounds at one order: forward / reverse upper (used) and lower (tightness only)."""

    order: float
    forward_upper: float
    reverse_upper: float
    forward_lower: float
    reverse_lower: float
    forward_method: str
    reverse_method: str

    @property
    def rdp(self) -> float:
        return max(self.forward_upper, self.reverse_upper)

    @property
    def log_gap(self) -> float:
        """Largest certified gap (upper - lower) of the two directions, in units of (alpha - 1) rho, i.e. log I."""
        return (self.order - 1.0) * max(self.forward_upper - self.forward_lower,
                                        self.reverse_upper - self.reverse_lower)


@dataclass(frozen=True)
class SGMRDPCurve:
    """Per-round certified RDP of the SGM on an order grid: both directions and their maximum (the curve used)."""

    orders: tuple[float, ...]
    forward_rdp: tuple[float, ...]  # upper bounds on D_alpha(P || Q)
    reverse_rdp: tuple[float, ...]  # upper bounds on D_alpha(Q || P)
    rdp: tuple[float, ...]  # max of the two
    forward_lower: tuple[float, ...]
    reverse_lower: tuple[float, ...]
    methods: tuple[str, ...]  # "<forward method>/<reverse method>" per order
    max_log_gap: float
    relative_tolerance: float


@dataclass(frozen=True)
class PrivacyAccountantResult:
    """Client-level (epsilon, delta) after T rounds with the certified curve and its metadata.

    For T = 0 or q = 0 no data-dependent release happens: epsilon = basic_epsilon = 0, raw_epsilon and the optimal
    orders are None.
    """

    epsilon: float
    delta: float
    optimal_order: float | None
    basic_epsilon: float
    basic_optimal_order: float | None
    sampling_rate: float
    noise_multiplier: float
    num_rounds: int
    num_client_slots: int
    delta_source: str  # "explicit" or "1/N"
    raw_epsilon: float | None
    epsilon_clamped_to_zero: bool
    orders: tuple[float, ...]
    forward_rdp: tuple[float, ...]
    reverse_rdp: tuple[float, ...]
    per_round_rdp: tuple[float, ...]
    composed_rdp: tuple[float, ...]
    rdp_methods: tuple[str, ...]
    max_certified_log_gap: float
    relative_tolerance: float
    conservative_fallbacks: tuple[str, ...]
    validity_notes: tuple[str, ...]
    integration_method: str
    numpy_version: str
    scipy_version: str
    python_version: str
    accountant_family: str = ACCOUNTANT_FAMILY
    accountant_version: str = ACCOUNTANT_VERSION
    conversion: str = CONVERSION

    def record(self, include_curve: bool = False) -> dict:
        """Metadata of the result as JSON primitives (public-safe key names: q, sigma, T, N, delta).

        include_curve adds the per-order curves (INTERNAL validation reports; never needed in a public summary).
        """
        record = {
            "accountant_family": self.accountant_family,
            "accountant_version": self.accountant_version,
            "q": self.sampling_rate,
            "sigma": self.noise_multiplier,
            "T": self.num_rounds,
            "N": self.num_client_slots,
            "delta": self.delta,
            "delta_source": self.delta_source,
            "order_grid": list(self.orders),
            "optimal_order": self.optimal_order,
            "raw_epsilon": self.raw_epsilon,
            "epsilon": self.epsilon,
            "epsilon_clamped_to_zero": self.epsilon_clamped_to_zero,
            "basic_epsilon_diagnostic": self.basic_epsilon,
            "conversion": self.conversion,
            "integration_method": self.integration_method,
            "relative_tolerance": self.relative_tolerance,
            "max_certified_log_gap": self.max_certified_log_gap,
            "conservative_fallbacks": list(self.conservative_fallbacks),
            "validity_limits": list(VALIDITY_LIMITS) + list(self.validity_notes),
            "numpy_version": self.numpy_version,
            "scipy_version": self.scipy_version,
            "python_version": self.python_version,
        }
        if include_curve:
            record.update(forward_rdp=list(self.forward_rdp), reverse_rdp=list(self.reverse_rdp),
                          per_round_rdp=list(self.per_round_rdp), rdp_methods=list(self.rdp_methods))
        return record


# ---------------------------------------------------------------- per-round bounds


def integer_sgm_rdp_closed_form(sampling_rate: float, noise_multiplier: float, order: int) -> float:
    """Exact forward D_alpha(P || Q) at an integer order alpha >= 2 (theory eq. (2); Mironov et al. 2019).

        rho(alpha) = log sum_{j=0}^{alpha} binom(alpha, j) (1-q)^(alpha-j) q^j exp((j^2 - j) / (2 sigma^2)) / (alpha - 1)

    Evaluated with log-binomials and logsumexp (positive terms only); no rounding allowance (see sgm_rdp_bounds).
    """
    _check_sampling_rate(sampling_rate)
    _check_noise_multiplier(noise_multiplier)
    if isinstance(order, bool) or not float(order).is_integer() or order < 2:
        raise ValueError(f"order must be an integer >= 2, got {order!r}")
    log_value, _ = _integer_forward_log_moment(sampling_rate, noise_multiplier, int(order))
    return log_value / (int(order) - 1)


def sgm_rdp_bounds(sampling_rate: float, noise_multiplier: float, order: float,
                   relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE) -> RDPBounds:
    """Certified per-round RDP bounds of both directions at one order (module doc)."""
    _check_sampling_rate(sampling_rate)
    _check_noise_multiplier(noise_multiplier)
    _check_order(order)
    _check_tolerance(relative_tolerance)
    return _bounds(float(sampling_rate), float(noise_multiplier), float(order), float(relative_tolerance))


def sgm_rdp_directions(sampling_rate: float, noise_multiplier: float, order: float,
                       relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE) -> tuple[float, float]:
    """(upper bound on D_alpha(P || Q), upper bound on D_alpha(Q || P)) of one SGM round."""
    bounds = sgm_rdp_bounds(sampling_rate, noise_multiplier, order, relative_tolerance)
    return bounds.forward_upper, bounds.reverse_upper


def sgm_rdp_at_order(sampling_rate: float, noise_multiplier: float, order: float,
                     relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE) -> float:
    """Upper bound on rho_{q,sigma}(alpha) = max of the two directions."""
    return sgm_rdp_bounds(sampling_rate, noise_multiplier, order, relative_tolerance).rdp


def compute_sgm_rdp_curve(
    sampling_rate: float,
    noise_multiplier: float,
    orders: tuple[float, ...] | None = None,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> SGMRDPCurve:
    """Certified per-round RDP curve on `orders` (default: the paper's order grid), validated (step 1)."""
    _check_sampling_rate(sampling_rate)
    _check_noise_multiplier(noise_multiplier)
    _check_tolerance(relative_tolerance)
    orders = default_rdp_orders() if orders is None else _checked_orders(orders)
    return _cached_curve(float(sampling_rate), float(noise_multiplier), tuple(orders), float(relative_tolerance))


@lru_cache(maxsize=256)
def _cached_curve(q: float, sigma: float, orders: tuple[float, ...], tolerance: float) -> SGMRDPCurve:
    bounds = [_bounds(q, sigma, order, tolerance) for order in orders]
    curve = SGMRDPCurve(
        orders=orders,
        forward_rdp=tuple(b.forward_upper for b in bounds),
        reverse_rdp=tuple(b.reverse_upper for b in bounds),
        rdp=tuple(b.rdp for b in bounds),
        forward_lower=tuple(b.forward_lower for b in bounds),
        reverse_lower=tuple(b.reverse_lower for b in bounds),
        methods=tuple(f"{b.forward_method}/{b.reverse_method}" for b in bounds),
        max_log_gap=max(b.log_gap for b in bounds),
        relative_tolerance=tolerance,
    )
    validate_rdp_curve(curve)
    return curve


def validate_rdp_curve(curve: SGMRDPCurve) -> None:
    """Step 1 (validation): every bound finite and >= 0, rdp = max of the directions, lower <= upper (else AccountantError).

    A Renyi divergence is >= 0, so a negative UPPER bound can only be a defect: it is reported, never clamped.
    """
    for index, order in enumerate(curve.orders):
        values = (curve.forward_rdp[index], curve.reverse_rdp[index], curve.rdp[index])
        if not all(math.isfinite(value) for value in values):
            raise AccountantError(f"non-finite RDP bound at order {order}: {values}")
        if min(values) < 0.0:
            raise AccountantError(f"negative RDP upper bound at order {order}: {values} (an anomaly, never clamped)")
        if curve.rdp[index] != max(curve.forward_rdp[index], curve.reverse_rdp[index]):
            raise AccountantError(f"the curve at order {order} is not the maximum of its two directions")
        if curve.forward_lower[index] > curve.forward_rdp[index] or curve.reverse_lower[index] > curve.reverse_rdp[index]:
            raise AccountantError(f"a certified lower bound exceeds its upper bound at order {order}")


def _bounds(q: float, sigma: float, order: float, tolerance: float) -> RDPBounds:
    """Both directions at one order (inputs already validated)."""
    if q == 0.0:  # P = Q
        return RDPBounds(order, 0.0, 0.0, 0.0, 0.0, CLOSED_FORM, CLOSED_FORM)
    if q == 1.0:  # the Gaussian mechanism: alpha / (2 sigma^2) in both directions
        value = order / (2.0 * sigma * sigma)
        return RDPBounds(order, value, value, value, value, CLOSED_FORM, CLOSED_FORM)
    scale = order - 1.0
    if order.is_integer():
        exact, magnitude = _integer_forward_log_moment(q, sigma, int(order))
        allowance = 1e-12 + 64.0 * _EPS * magnitude
        forward_up, forward_low, forward_method = exact + allowance, exact - allowance, CLOSED_FORM
    else:
        forward_up, forward_low = _certified_log_moment(order, q, sigma, tolerance)
        forward_method = CERTIFIED_INTEGRATION
    reverse_up, reverse_low = _certified_log_moment(1.0 - order, q, sigma, tolerance)
    return RDPBounds(order, forward_up / scale, reverse_up / scale, max(0.0, forward_low) / scale,
                     max(0.0, reverse_low) / scale, forward_method, CERTIFIED_INTEGRATION)


def _integer_forward_log_moment(q: float, sigma: float, alpha: int) -> tuple[float, float]:
    """(log I(alpha) by the exact binomial sum, magnitude of its largest exponent) for 0 < q < 1."""
    if q == 1.0:
        value = alpha * (alpha - 1) / (2.0 * sigma * sigma)
        return value, abs(value)
    j = np.arange(alpha + 1, dtype=np.float64)
    log_binomial = gammaln(alpha + 1.0) - gammaln(j + 1.0) - gammaln(alpha - j + 1.0)
    log_terms = (log_binomial + (alpha - j) * math.log1p(-q) + j * math.log(q)
                 + (j * j - j) / (2.0 * sigma * sigma))
    return float(logsumexp(log_terms)), float(np.max(np.abs(log_terms))) + float(np.max(np.abs(log_binomial)))


# ---------------------------------------------------------------- certified integration of E[r(Z)^beta]


class _LogIntegrand:
    """g(z) = beta log r(z) - z^2 / 2 and g'(z) for 0 < q < 1 (vectorised and scalar forms)."""

    def __init__(self, beta: float, q: float, sigma: float) -> None:
        self.beta, self.q, self.sigma = beta, q, sigma
        self.log_one_minus_q = math.log1p(-q)
        self.shift = -1.0 / (2.0 * sigma * sigma) + math.log(q) - math.log1p(-q)  # t(z) = z / sigma + shift
        self.kappa = beta / (sigma * sigma)
        self.slope = beta / sigma  # g'(z) = slope * s(z) - z

    def arrays(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = z / self.sigma + self.shift
        log_r = self.log_one_minus_q + np.logaddexp(0.0, t)
        return self.beta * log_r - 0.5 * z * z, self.slope * expit(t) - z

    def g(self, z: float) -> float:
        t = z / self.sigma + self.shift
        return self.beta * (self.log_one_minus_q + _softplus(t)) - 0.5 * z * z

    def dg(self, z: float) -> float:
        return self.slope * _expit(z / self.sigma + self.shift) - z

    def z_of_t(self, t: float) -> float:
        return self.sigma * (t - self.shift)


@dataclass(frozen=True)
class _Piece:
    lower: float  # may be -inf
    upper: float  # may be +inf
    curvature_min: float
    curvature_max: float
    band: bool  # the band is always fully covered by cells; the other pieces are concave


def _pieces(integrand: _LogIntegrand) -> list[_Piece]:
    """The at most three curvature pieces of g (module doc, (i))."""
    kappa = integrand.kappa
    outside = (-1.0, 0.0) if kappa > 0 else (-2.0, -1.0)
    if abs(kappa) <= 4.0:
        return [_Piece(-math.inf, math.inf, *outside, band=False)]
    w = math.sqrt(1.0 - 4.0 / abs(kappa))
    half_width = (math.log1p(w) - math.log1p(-w)) * (1.0 + _BAND_WIDENING) + _BAND_WIDENING  # in t; widened
    left, right = integrand.z_of_t(-half_width), integrand.z_of_t(half_width)
    inside = (-1.0, kappa / 4.0 - 1.0) if kappa > 0 else (kappa / 4.0 - 1.0, -1.0)
    return [_Piece(-math.inf, left, *outside, band=False), _Piece(left, right, *inside, band=True),
            _Piece(right, math.inf, *outside, band=False)]


def _certified_log_moment(beta: float, q: float, sigma: float, tolerance: float) -> tuple[float, float]:
    """(upper, lower) bounds on log E[r(Z)^beta] for 0 < q < 1 (module doc, (ii)-(iv))."""
    integrand = _LogIntegrand(beta, q, sigma)
    upper_terms, lower_terms, magnitude = [], [], 1.0
    for piece in _pieces(integrand):
        if piece.band:
            start, stop = piece.lower, piece.upper
        else:
            peak = _concave_argmax(integrand, piece.lower, piece.upper)
            target = integrand.g(peak) - SUPERLEVEL_DEPTH
            start = _level_point(integrand, piece.lower, peak, target, direction=-1.0)
            stop = _level_point(integrand, piece.upper, peak, target, direction=1.0)
            if stop < piece.upper:  # right tangent bound (concavity of g on [stop, upper])
                upper_terms.append(integrand.g(stop) + _log_exp_integral(integrand.dg(stop), piece.upper - stop))
            if start > piece.lower:  # left tangent bound, in y = start - z >= 0
                upper_terms.append(integrand.g(start) + _log_exp_integral(-integrand.dg(start), start - piece.lower))
        cells_upper, cells_lower, cells_magnitude = _cells(integrand, start, stop, piece.curvature_min,
                                                           piece.curvature_max, tolerance)
        upper_terms.append(cells_upper)
        lower_terms.append(cells_lower)
        magnitude = max(magnitude, cells_magnitude)
    log_upper = float(logsumexp(upper_terms)) - _LOG_SQRT_2PI
    log_lower = float(logsumexp(lower_terms)) - _LOG_SQRT_2PI
    allowance = 1e-12 + 64.0 * _EPS * magnitude
    if not (math.isfinite(log_upper) and math.isfinite(log_lower)):
        raise AccountantError(f"no finite certified bound for E[r^beta], beta = {beta}, q = {q}, sigma = {sigma}")
    return log_upper + allowance, log_lower - allowance


def _cells(integrand: _LogIntegrand, start: float, stop: float, curvature_min: float, curvature_max: float,
           tolerance: float) -> tuple[float, float, float]:
    """(log upper, log lower, magnitude) of int_start^stop exp(g) from second-order Taylor cells (module doc, (ii))."""
    width = stop - start
    if not width > 0.0:
        return -math.inf, -math.inf, 1.0
    spread = max(0.0, curvature_max) - min(0.0, curvature_min)  # > 0: curvature_min <= -1 on every piece
    half = math.sqrt(2.0 * tolerance / spread)
    count = min(MAX_CELLS_PER_PIECE, max(1, math.ceil(width / (2.0 * half))))
    eta = width / (2.0 * count)
    centres = start + (2.0 * np.arange(count, dtype=np.float64) + 1.0) * eta
    values, slopes = integrand.arrays(centres)
    # upper-bound cells widened by a few ulps of the coordinates: they cover [start, stop] despite rounding
    eta_cover = eta + 4.0 * _EPS * (abs(start) + abs(stop) + 1.0)
    upper = (values + math.log(2.0 * eta_cover) + _log_sinhc(slopes * eta_cover)
             + max(0.0, curvature_max) * eta_cover * eta_cover / 2.0)
    lower = values + math.log(2.0 * eta) + _log_sinhc(slopes * eta) + min(0.0, curvature_min) * eta * eta / 2.0
    magnitude = float(np.max(np.abs(values))) + 0.5 * max(start * start, stop * stop) + 1.0
    return float(logsumexp(upper)), float(logsumexp(lower)), magnitude


def _log_sinhc(x: np.ndarray) -> np.ndarray:
    """log(sinh(x) / x) (even; >= 0), stable for small and large |x|."""
    a = np.abs(x)
    out = np.empty_like(a)
    small = a < 1e-4
    out[small] = a[small] * a[small] / 6.0  # >= the exact value minus a^4 / 180 < 1e-17
    big = ~small
    ab = a[big]
    out[big] = ab + np.log(-np.expm1(-2.0 * ab)) - np.log(2.0 * ab)
    return out


def _log_exp_integral(slope: float, length: float) -> float:
    """log int_0^length exp(slope y) dy (length may be +inf, then slope < 0 is required)."""
    if not length > 0.0:
        return -math.inf
    if math.isinf(length):
        if not slope < 0.0:
            raise AccountantError(f"an infinite tail with slope {slope} >= 0 cannot be bounded")
        return -math.log(-slope)
    if slope == 0.0:
        return math.log(length)
    x = slope * length
    if slope > 0.0:
        return x + math.log(-math.expm1(-x)) - math.log(slope)
    return math.log(-math.expm1(x)) - math.log(-slope)


def _concave_argmax(integrand: _LogIntegrand, lower: float, upper: float) -> float:
    """Maximiser of g on a concave piece [lower, upper] by bisection on the decreasing g' (bracket from g' bounds).

    For z < min(0, beta / sigma), g'(z) > 0; for z > max(0, beta / sigma), g'(z) < 0 (0 <= s <= 1).
    """
    left = lower if math.isfinite(lower) else min(upper, 0.0, integrand.slope) - 1.0
    right = upper if math.isfinite(upper) else max(lower, 0.0, integrand.slope) + 1.0
    if integrand.dg(right) >= 0.0:
        return right
    if integrand.dg(left) <= 0.0:
        return left
    for _ in range(_BISECTION_STEPS):
        middle = 0.5 * (left + right)
        if middle <= left or middle >= right:
            break
        if integrand.dg(middle) > 0.0:
            left = middle
        else:
            right = middle
    return 0.5 * (left + right)


def _level_point(integrand: _LogIntegrand, end: float, peak: float, target: float, direction: float) -> float:
    """From the maximiser `peak` towards `end` (direction +1: right, -1: left) on a concave piece: the piece end if g
    stays >= target up to it, else a point where g < target (g is monotone there, so bisection is valid)."""
    if math.isfinite(end) and integrand.g(end) >= target:
        return end
    step = 1.0
    far = peak + direction * step
    for _ in range(_BISECTION_STEPS):
        if math.isfinite(end) and (far - end) * direction >= 0.0:
            far = end  # g(end) < target
            break
        if integrand.g(far) < target:
            break
        step *= 2.0
        far = peak + direction * step
    else:
        raise AccountantError("the superlevel interval of a concave piece could not be bracketed")
    near = peak
    for _ in range(_BISECTION_STEPS):
        middle = 0.5 * (near + far)
        if middle == near or middle == far:
            break
        if integrand.g(middle) >= target:
            near = middle
        else:
            far = middle
    return far


def _softplus(t: float) -> float:
    return t + math.log1p(math.exp(-t)) if t > 0.0 else math.log1p(math.exp(t))


def _expit(t: float) -> float:
    if t >= 0.0:
        return 1.0 / (1.0 + math.exp(-t))
    e = math.exp(t)
    return e / (1.0 + e)


# ---------------------------------------------------------------- composition and conversion


def compose_rdp(per_round_rdp: tuple[float, ...], num_rounds: int) -> tuple[float, ...]:
    """Adaptive composition of T identical rounds: rho_total(alpha) = T rho(alpha)."""
    _check_integer(num_rounds, "num_rounds", minimum=0)
    return tuple(num_rounds * value for value in per_round_rdp)


def rdp_to_epsilon(orders: tuple[float, ...], rdp: tuple[float, ...], delta: float) -> tuple[float, float]:
    """Step 2 (conversion): (RAW epsilon, optimal order), raw epsilon = min_alpha [rdp + log((alpha-1)/alpha)
    - (log delta + log alpha)/(alpha-1)]. It may be negative (no clamping here). Ties: smallest order."""
    alphas, values = _conversion_inputs(orders, rdp, delta)
    epsilons = values + np.log((alphas - 1.0) / alphas) - (math.log(delta) + np.log(alphas)) / (alphas - 1.0)
    index = int(np.argmin(epsilons))
    return float(epsilons[index]), float(alphas[index])


def rdp_to_basic_epsilon(orders: tuple[float, ...], rdp: tuple[float, ...], delta: float) -> tuple[float, float]:
    """Diagnostic only: (epsilon, optimal order) with epsilon = min_alpha [rdp + log(1/delta)/(alpha-1)]."""
    alphas, values = _conversion_inputs(orders, rdp, delta)
    epsilons = values + math.log(1.0 / delta) / (alphas - 1.0)
    index = int(np.argmin(epsilons))
    return float(epsilons[index]), float(alphas[index])


def compute_privacy_accounting(
    config: PrivacyAccountantConfig, relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE
) -> PrivacyAccountantResult:
    """epsilon_SGM(q, sigma, T, delta) on the paper's order grid: validation, conversion, non-negativity."""
    orders = default_rdp_orders()
    curve = compute_sgm_rdp_curve(config.sampling_rate, config.noise_multiplier, orders, relative_tolerance)
    composed = compose_rdp(curve.rdp, config.num_rounds)
    delta = config.target_delta
    fallbacks, notes = [], []

    if config.num_rounds == 0 or config.sampling_rate == 0.0:
        # No data-dependent release: the conversion bound would be positive but meaningless.
        epsilon, optimal_order, basic_epsilon, basic_optimal_order, raw_epsilon = 0.0, None, 0.0, None, None
    else:
        raw_epsilon, optimal_order = rdp_to_epsilon(orders, composed, delta)
        basic_epsilon, basic_optimal_order = rdp_to_basic_epsilon(orders, composed, delta)
        epsilon = max(0.0, raw_epsilon)
        if optimal_order in (orders[0], orders[-1]):
            notes.append(f"the optimal order {optimal_order} is at the edge of the grid: a wider grid could give a "
                         "smaller epsilon (the reported value remains a valid upper bound)")
        if epsilon > raw_epsilon:
            notes.append(f"the conversion gave raw epsilon {raw_epsilon!r} < 0: (0, delta)-DP holds a fortiori and "
                         "epsilon = 0 is reported (the RDP curve itself was validated as non-negative)")
    if curve.max_log_gap > GAP_FALLBACK_THRESHOLD:
        fallbacks.append(f"certified bracket wider than {GAP_FALLBACK_THRESHOLD} (log gap {curve.max_log_gap!r}); "
                         "its upper end is used, so epsilon may be conservative")

    return PrivacyAccountantResult(
        epsilon=epsilon,
        delta=delta,
        optimal_order=optimal_order,
        basic_epsilon=basic_epsilon,
        basic_optimal_order=basic_optimal_order,
        sampling_rate=config.sampling_rate,
        noise_multiplier=config.noise_multiplier,
        num_rounds=config.num_rounds,
        num_client_slots=config.num_client_slots,
        delta_source="1/N" if config.delta is None else "explicit",
        raw_epsilon=raw_epsilon,
        epsilon_clamped_to_zero=raw_epsilon is not None and raw_epsilon < 0.0,
        orders=orders,
        forward_rdp=curve.forward_rdp,
        reverse_rdp=curve.reverse_rdp,
        per_round_rdp=curve.rdp,
        composed_rdp=composed,
        rdp_methods=curve.methods,
        max_certified_log_gap=curve.max_log_gap,
        relative_tolerance=curve.relative_tolerance,
        conservative_fallbacks=tuple(fallbacks),
        validity_notes=tuple(notes),
        integration_method=INTEGRATION_METHOD,
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        python_version=platform.python_version(),
    )


# ---------------------------------------------------------------- validation


def _conversion_inputs(orders, rdp, delta) -> tuple[np.ndarray, np.ndarray]:
    _check_delta(delta)
    alphas = np.asarray(_checked_orders(orders), dtype=np.float64)
    values = np.asarray(rdp, dtype=np.float64)
    if values.shape != alphas.shape:
        raise ValueError(f"got {values.size} RDP values for {alphas.size} orders")
    if not np.all(np.isfinite(values)):
        raise ValueError("RDP values must be finite")
    if np.any(values < 0.0):
        raise ValueError("RDP values must be non-negative (a negative RDP is an anomaly, never converted)")
    return alphas, values


def _checked_orders(orders) -> tuple[float, ...]:
    orders = tuple(float(order) for order in orders)
    if len(orders) == 0:
        raise ValueError("at least one order is required")
    for order in orders:
        _check_order(order)
    if any(b <= a for a, b in zip(orders, orders[1:])):
        raise ValueError("orders must be strictly increasing")
    return orders


def _check_real(value, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def _check_integer(value, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")


def _check_sampling_rate(sampling_rate: float) -> None:
    _check_real(sampling_rate, "sampling_rate")
    if not 0.0 <= sampling_rate <= 1.0:
        raise ValueError(f"sampling_rate must be in [0, 1], got {sampling_rate}")


def _check_noise_multiplier(noise_multiplier: float) -> None:
    _check_real(noise_multiplier, "noise_multiplier")
    if not noise_multiplier > 0.0:
        raise ValueError(f"noise_multiplier must be positive, got {noise_multiplier}")


def _check_order(order: float) -> None:
    _check_real(order, "order")
    if not order > 1.0:
        raise ValueError(f"RDP orders must be > 1, got {order}")


def _check_tolerance(tolerance: float) -> None:
    _check_real(tolerance, "relative_tolerance")
    if not 0.0 < tolerance <= 1e-2:
        raise ValueError(f"relative_tolerance must be in (0, 1e-2], got {tolerance}")


def _check_delta(delta: float) -> None:
    _check_real(delta, "delta")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
