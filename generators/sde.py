"""Regime-switching Ornstein-Uhlenbeck process (Doob's exact simulation).

References: Uhlenbeck & Ornstein (1930), Hamilton (1989), Gillespie (1996).
"""

import logging
import numpy as np
from .base import BaseSynthesizer

logger = logging.getLogger(__name__)


class SDESynthesizer(BaseSynthesizer):
    """Regime-switching OU process generator."""

    def __init__(self, length: int = 1024, random_seed: int = 42) -> None:
        super().__init__(length=length, random_seed=random_seed)

    def _sample_transition_matrix(self, n_regimes: int, rng: np.random.Generator) -> np.ndarray:
        diag = rng.uniform(15.0, 50.0)
        P = np.zeros((n_regimes, n_regimes))
        for i in range(n_regimes):
            alpha = np.full(n_regimes, 1.0)
            alpha[i] = diag
            P[i] = rng.dirichlet(alpha)
        return P

    def _sample_regime_params(self, n_regimes, rng):
        thetas = rng.uniform(0.05, 5.0, size=n_regimes)
        sigmas = rng.uniform(0.05, 2.0, size=n_regimes)
        base_mu = rng.uniform(-3.0, 3.0)
        if n_regimes == 1:
            return thetas, np.array([base_mu]), sigmas
        mus = np.empty(n_regimes)
        mus[0] = base_mu
        for i in range(1, n_regimes):
            mus[i] = mus[i - 1] + rng.choice([-1.0, 1.0]) * rng.uniform(1.0, 4.0)
        return thetas, mus, sigmas

    def _simulate_ou_exact(self, regimes, thetas, mus, sigmas, rng):
        dt = 1.0
        x = np.empty(self.length, dtype=np.float64)
        k0 = regimes[0]
        x[0] = mus[k0] + (sigmas[k0] / np.sqrt(2.0 * thetas[k0])) * rng.standard_normal()
        decay = np.exp(-thetas * dt)
        t_std = np.sqrt(np.maximum((sigmas**2 / (2 * thetas)) * (1 - np.exp(-2 * thetas * dt)), 1e-15))
        noise = rng.standard_normal(self.length - 1)
        for t in range(self.length - 1):
            k = regimes[t + 1]
            x[t + 1] = mus[k] + (x[t] - mus[k]) * decay[k] + t_std[k] * noise[t]
        return x

    def _gen_single_series(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed=seed)
        n_regimes = int(rng.choice([2, 3, 4], p=[0.5, 0.3, 0.2]))
        thetas, mus, sigmas = self._sample_regime_params(n_regimes, rng)
        P = self._sample_transition_matrix(n_regimes, rng)
        regimes = np.empty(self.length, dtype=np.int32)
        regimes[0] = rng.integers(0, n_regimes)
        for t in range(1, self.length):
            regimes[t] = rng.choice(n_regimes, p=P[regimes[t - 1]])
        series = self._simulate_ou_exact(regimes, thetas, mus, sigmas, rng)
        if not np.all(np.isfinite(series)):
            series = self._simulate_ou_exact(
                np.zeros(self.length, dtype=np.int32),
                np.array([1.0]), np.array([0.0]), np.array([0.5]), rng,
            )
        return series.astype(np.float32, copy=False)[None, :]