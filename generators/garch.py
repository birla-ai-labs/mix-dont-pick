"""GARCH-family generator: GARCH(1,1), GJR-GARCH, EGARCH.

References: Bollerslev (1986), Glosten et al. (1993), Nelson (1991).
"""

import logging
import numpy as np
from .base import BaseSynthesizer

logger = logging.getLogger(__name__)


class GARCHSynthesizer(BaseSynthesizer):

    def __init__(self, length: int = 1024, random_seed: int = 42) -> None:
        super().__init__(length=length, random_seed=random_seed)
        self.burn_in = 500
        self.model_types = ["garch11", "gjr_garch", "egarch"]
        self.mean_types = ["zero", "constant", "ar1"]
        self.innov_types = ["normal", "student_t"]

    def _sample_innovations(self, n, innov_type, doff, rng):
        if innov_type == "normal":
            return rng.standard_normal(n)
        raw = rng.standard_t(doff, size=n)
        return raw / np.sqrt(doff / (doff - 2.0))

    def _simulate_garch11(self, omega, alpha, beta, mu, phi, z):
        L = len(z)
        sigma2, eps, r = np.empty(L), np.empty(L), np.empty(L)
        sigma2[0] = omega / (1 - alpha - beta)
        eps[0] = np.sqrt(sigma2[0]) * z[0]
        r[0] = mu + eps[0]
        for t in range(1, L):
            sigma2[t] = max(omega + alpha * eps[t-1]**2 + beta * sigma2[t-1], 1e-8)
            eps[t] = np.sqrt(sigma2[t]) * z[t]
            r[t] = mu + phi * (r[t-1] - mu) + eps[t]
        return r

    def _simulate_gjr_garch(self, omega, alpha, beta, gamma, mu, phi, z):
        L = len(z)
        sigma2, eps, r = np.empty(L), np.empty(L), np.empty(L)
        sigma2[0] = omega / max(1 - alpha - gamma / 2 - beta, 1e-6)
        eps[0] = np.sqrt(sigma2[0]) * z[0]
        r[0] = mu + eps[0]
        for t in range(1, L):
            ind = 1.0 if eps[t-1] < 0 else 0.0
            sigma2[t] = max(omega + (alpha + gamma * ind) * eps[t-1]**2 + beta * sigma2[t-1], 1e-8)
            eps[t] = np.sqrt(sigma2[t]) * z[t]
            r[t] = mu + phi * (r[t-1] - mu) + eps[t]
        return r

    def _simulate_egarch(self, omega, alpha, beta, gamma, mu, phi, z):
        L = len(z)
        log_s2, eps, r = np.empty(L), np.empty(L), np.empty(L)
        E_abs_z = np.sqrt(2 / np.pi)
        log_s2[0] = omega / (1 - beta)
        eps[0] = np.sqrt(np.exp(log_s2[0])) * z[0]
        r[0] = mu + eps[0]
        for t in range(1, L):
            log_s2[t] = np.clip(omega + alpha * (abs(z[t-1]) - E_abs_z) + gamma * z[t-1] + beta * log_s2[t-1], -20, 20)
            eps[t] = np.sqrt(np.exp(log_s2[t])) * z[t]
            r[t] = mu + phi * (r[t-1] - mu) + eps[t]
        return r

    def _gen_single_series(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed=seed)
        total_len = self.length + self.burn_in
        model = rng.choice(self.model_types)
        innov = rng.choice(self.innov_types)
        mean_type = rng.choice(self.mean_types)
        z = self._sample_innovations(total_len, innov, float(rng.uniform(3, 10)), rng)
        mu = float(rng.normal(0, 0.05)) if mean_type == "constant" else 0.0
        phi = float(rng.uniform(-0.3, 0.3)) if mean_type == "ar1" else 0.0
        sigma_unc = float(np.exp(rng.uniform(np.log(0.005), np.log(0.5))))

        try:
            if model == "garch11":
                pers = float(rng.uniform(0.70, 0.97))
                a_share = float(rng.uniform(0.02, 0.25))
                a, b = pers * a_share, pers - pers * a_share
                r = self._simulate_garch11(sigma_unc**2 * (1 - a - b), a, b, mu, phi, z)
            elif model == "gjr_garch":
                g = float(rng.uniform(0.01, 0.15))
                pers = float(rng.uniform(0.70, 0.97))
                a_share = float(rng.uniform(0.02, 0.20))
                a = pers * a_share
                b = max(pers - a - g / 2, 0.01)
                r = self._simulate_gjr_garch(sigma_unc**2 * max(1 - a - g/2 - b, 1e-4), a, b, g, mu, phi, z)
            else:
                b = float(rng.uniform(0.70, 0.97))
                a = float(rng.uniform(0.05, 0.25))
                g = float(rng.uniform(-0.20, 0.20))
                r = self._simulate_egarch(np.log(sigma_unc**2) * (1 - b), a, b, g, mu, phi, z)
            r = r[self.burn_in:]
            if not np.all(np.isfinite(r)):
                raise ValueError("non-finite")
        except Exception:
            z_safe = self._sample_innovations(total_len, "normal", 5.0, rng)
            r = self._simulate_garch11(1e-4, 0.05, 0.90, 0.0, 0.0, z_safe)[self.burn_in:]

        if rng.random() < 0.5:
            r = np.cumsum(r)
        return r.astype(np.float32, copy=False)[None, :]