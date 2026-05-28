"""ARIMA/SARIMA generator.

Covers random walks (d=1), stationary ARMA, and seasonal ARIMA.
Reference: Box & Jenkins (1970), Hamilton (1994) Ch. 3.
"""

import logging
import numpy as np
from .base import BaseSynthesizer

logger = logging.getLogger(__name__)


class ARIMASynthesizer(BaseSynthesizer):
    """ARIMA(p,d,q) x (P,D,Q,m) generator with randomized orders."""

    def __init__(self, length: int = 1024, random_seed: int = 42) -> None:
        super().__init__(length=length, random_seed=random_seed)
        self.burn_in = 500

    @staticmethod
    def _check_stationarity(ar_coeffs: np.ndarray) -> bool:
        """Check AR polynomial roots lie outside the unit circle."""
        if len(ar_coeffs) == 0:
            return True
        poly = np.r_[1, -ar_coeffs]
        roots = np.abs(np.roots(poly))
        return bool(np.all(roots > 1.05))

    @staticmethod
    def _check_invertibility(ma_coeffs: np.ndarray) -> bool:
        """Check MA polynomial roots lie outside the unit circle."""
        if len(ma_coeffs) == 0:
            return True
        poly = np.r_[1, ma_coeffs]
        roots = np.abs(np.roots(poly))
        return bool(np.all(roots > 1.05))

    def _sample_ar_coeffs(self, p: int, rng: np.random.Generator) -> np.ndarray:
        """Sample stationary AR coefficients with rejection."""
        for _ in range(100):
            coeffs = rng.uniform(-0.8, 0.8, size=p)
            if self._check_stationarity(coeffs):
                return coeffs
        # fallback: small safe coefficients
        return rng.uniform(-0.3, 0.3, size=p)

    def _sample_ma_coeffs(self, q: int, rng: np.random.Generator) -> np.ndarray:
        """Sample invertible MA coefficients with rejection."""
        for _ in range(100):
            coeffs = rng.uniform(-0.8, 0.8, size=q)
            if self._check_invertibility(coeffs):
                return coeffs
        return rng.uniform(-0.3, 0.3, size=q)

    def _simulate_arma(
        self, ar: np.ndarray, ma: np.ndarray, sigma: float, n: int, rng: np.random.Generator,
    ) -> np.ndarray:
        """Simulate ARMA(p,q) via direct recursion."""
        p, q = len(ar), len(ma)
        eps = rng.normal(0, sigma, size=n)
        y = np.zeros(n)
        for t in range(n):
            y[t] = eps[t]
            for j in range(min(p, t)):
                y[t] += ar[j] * y[t - j - 1]
            for j in range(min(q, t)):
                y[t] += ma[j] * eps[t - j - 1]
        return y

    def _gen_single_series(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed=seed)
        total_len = self.length + self.burn_in

        # Sample orders
        p = int(rng.choice([0, 1, 2, 3], p=[0.1, 0.3, 0.4, 0.2]))
        d = int(rng.choice([0, 1, 2], p=[0.4, 0.4, 0.2]))
        q = int(rng.choice([0, 1, 2, 3], p=[0.1, 0.3, 0.4, 0.2]))
        if p == 0 and q == 0:
            p = 1  # avoid trivial white noise

        # Seasonal component (30% chance)
        seasonal = rng.random() < 0.3
        m = int(rng.choice([4, 7, 12, 24, 52])) if seasonal else 0
        P = int(rng.choice([0, 1])) if seasonal else 0
        D = int(rng.choice([0, 1])) if seasonal else 0
        Q = int(rng.choice([0, 1])) if seasonal else 0

        sigma = float(np.exp(rng.uniform(np.log(0.01), np.log(2.0))))

        # Non-seasonal ARMA
        ar = self._sample_ar_coeffs(p, rng) if p > 0 else np.array([])
        ma = self._sample_ma_coeffs(q, rng) if q > 0 else np.array([])

        y = self._simulate_arma(ar, ma, sigma, total_len, rng)

        if seasonal and m > 0:
            if P > 0:
                sar = self._sample_ar_coeffs(P, rng)
                for t in range(m * P, total_len):
                    for j in range(P):
                        y[t] += sar[j] * y[t - m * (j + 1)]
            if Q > 0:
                sma = self._sample_ma_coeffs(Q, rng)
                eps_s = rng.normal(0, sigma * 0.5, size=total_len)
                for t in range(m * Q, total_len):
                    for j in range(Q):
                        y[t] += sma[j] * eps_s[t - m * (j + 1)]
            for _ in range(D):
                y = y[m:] - y[:-m]

        for _ in range(d):
            y = np.cumsum(y)

        y = y[-self.length:]

        if not np.all(np.isfinite(y)):
            y = rng.normal(0, sigma, size=self.length)

        return y.astype(np.float32, copy=False)[None, :]