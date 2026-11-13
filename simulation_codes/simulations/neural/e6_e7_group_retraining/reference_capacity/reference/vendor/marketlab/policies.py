"""Auction-precommitted encoders and price-affine Gaussian forecasting heads.

At time t, encoder input u_t=(p_{t-1},D_t,macro_t) is fixed BEFORE the
auction. The candidate p_t may enter the head mean only via chi(z)*p_t;
the head's variance v(z) and all other coefficients exclude p_t.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


class ZeroEncoder:
    def reset(self):
        pass
    def step(self, u):
        return np.zeros(0)


@dataclass
class SelectPublicEncoder:
    indices: tuple[int, ...]
    def reset(self):
        pass
    def step(self, u):
        return np.asarray(u, float)[list(self.indices)].copy()


@dataclass
class RecurrentEncoder:
    A: np.ndarray
    B: np.ndarray
    nonlinear: bool = True
    state: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.A = np.asarray(self.A, float)
        self.B = np.asarray(self.B, float)
        d = self.A.shape[0]
        if self.A.shape != (d,d) or self.B.shape[0] != d:
            raise ValueError('Encoder shapes are inconsistent')
        if max(abs(np.linalg.eigvals(self.A))) >= 1:
            raise ValueError('Isolated recurrent encoder must be stable')
        self.reset()

    def reset(self):
        self.state = np.zeros(self.A.shape[0])

    def step(self, u):
        u = np.asarray(u, float)
        if len(u) != self.B.shape[1]:
            raise ValueError('Public input dimension mismatch')
        proposal = self.A @ self.state + self.B @ u
        self.state = np.tanh(proposal) if self.nonlinear else proposal
        return self.state.copy()


@dataclass(frozen=True)
class HeadOutput:
    intercept: float
    private_coef: float
    price_coef: float
    variance: float


@dataclass(frozen=True)
class AffineGaussianHead:
    """State-dependent mean/variance; constant affine is zero state coefficients.

    If nonlinear_state=True, state is passed through tanh for bounded coeffs.
    Identity state is useful for the analytically Gaussian witness.
    """
    intercept: float
    private_coef: float
    price_coef: float
    log_variance: float
    intercept_weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    private_weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    price_weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    logvar_weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    nonlinear_state: bool = True

    def evaluate(self, state) -> HeadOutput:
        z = np.asarray(state,float)
        if self.nonlinear_state:
            z = np.tanh(z)
        weights = [np.asarray(getattr(self, s), float) for s in
                   ('intercept_weights','private_weights','price_weights','logvar_weights')]
        if any(len(a) != len(z) for a in weights):
            raise ValueError('Head state/weight dimension mismatch')
        a = self.intercept + weights[0] @ z
        beta = self.private_coef + weights[1] @ z
        chi = self.price_coef + weights[2] @ z
        v = float(np.exp(self.log_variance + weights[3] @ z))
        if not np.isfinite(v) or v <= 0 or not np.isfinite(a+beta+chi):
            raise ValueError('Invalid Gaussian head')
        return HeadOutput(float(a),float(beta),float(chi),v)


@dataclass
class InvestorPolicy:
    name: str
    mass: float
    gamma: float
    signal_loading: np.ndarray
    signal_noise_sd: float
    encoder: object
    head: AffineGaussianHead

    def __post_init__(self):
        self.signal_loading = np.asarray(self.signal_loading,float)
        assert self.mass > 0 and self.gamma > 0 and self.signal_noise_sd > 0
