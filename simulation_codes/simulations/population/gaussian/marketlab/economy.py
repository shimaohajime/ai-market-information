"""Exogenous primitives and reproducible innovation banks for one-period asset markets.

Timing: f_t and pre-auction public macro x_t are realized before trading;
D_t is the PREVIOUS asset's matured dividend; D_{t+1} (from f_t and a fresh
shock) is realized only AFTER the auction. Future payoffs are stored in an
innovation bank for scoring, never passed to trading policies.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_discrete_lyapunov


@dataclass(frozen=True)
class Economy:
    name: str
    F: np.ndarray
    G: np.ndarray
    payoff_loading: np.ndarray
    target_loading: np.ndarray
    public_loading: np.ndarray
    public_noise_sd: np.ndarray
    payoff_noise_sd: float
    supply_noise_sd: float
    supply_mean: float = 0.0
    risk_free_gross: float = 1.0
    volatility_loading: np.ndarray | None = None

    def __post_init__(self):
        F = np.asarray(self.F, float)
        G = np.asarray(self.G, float)
        m = F.shape[0]
        assert F.shape == (m, m) and G.shape[0] == m
        assert np.asarray(self.payoff_loading).shape == (m,)
        assert np.asarray(self.target_loading).shape == (m,)
        assert np.asarray(self.public_loading).ndim == 2
        assert np.asarray(self.public_loading).shape[1] == m
        assert np.asarray(self.public_noise_sd).shape == (np.asarray(self.public_loading).shape[0],)
        assert self.risk_free_gross > 0 and self.supply_noise_sd > 0 and self.payoff_noise_sd > 0
        if self.volatility_loading is not None:
            assert np.asarray(self.volatility_loading).shape == (m,)
        if max(abs(np.linalg.eigvals(F))) >= 1:
            raise ValueError('Exogenous latent law must be stationary (spectral radius < 1).')

    @property
    def dim(self):
        return self.F.shape[0]

    @property
    def n_public(self):
        return self.public_loading.shape[0]

    @property
    def stationary_cov(self):
        return solve_discrete_lyapunov(self.F, self.G @ self.G.T)

    def draw_bank(self, periods: int, burn: int = 400, seed: int = 17,
                  n_private_streams: int = 8) -> 'ShockBank':
        rng = np.random.default_rng(seed)
        n = periods + burn
        cov = self.stationary_cov
        f = np.zeros((n, self.dim))
        f_prev = rng.multivariate_normal(np.zeros(self.dim), cov)
        innovations = rng.standard_normal((n, self.G.shape[1]))
        for t in range(n):
            f[t] = self.F @ f_prev + self.G @ innovations[t]
            f_prev = f[t]
        pub = f @ self.public_loading.T + rng.standard_normal((n, self.n_public))*self.public_noise_sd
        eps = rng.standard_normal(n)
        if self.volatility_loading is None:
            sd = np.full(n, self.payoff_noise_sd)
        else:
            # A bounded but nonconstant volatility; payoff is then generally non-Gaussian.
            log_scale = np.clip(f @ self.volatility_loading, -2.0, 2.0)
            sd = self.payoff_noise_sd * np.exp(log_scale)
        future = f @ self.payoff_loading + sd*eps
        supply_shock = self.supply_noise_sd * rng.standard_normal(n)
        private_noise = rng.standard_normal((n, n_private_streams))
        lag_dividend = np.r_[0.0, future[:-1]]
        return ShockBank(f, pub, lag_dividend, future, supply_shock,
                         private_noise, burn=burn, seed=seed)


@dataclass(frozen=True)
class ShockBank:
    factor: np.ndarray
    public: np.ndarray
    lag_dividend: np.ndarray
    future_dividend: np.ndarray
    supply_shock: np.ndarray
    private_stdnormal: np.ndarray
    burn: int
    seed: int

    @property
    def n(self):
        return len(self.factor)
