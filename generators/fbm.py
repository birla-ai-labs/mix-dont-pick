"""Fractional Brownian Motion / fractional Gaussian noise generator.

Uses the `fbm` package (circulant embedding, O(L log L)) for speed.
Fallback to Cholesky if unavailable.
Reference: Mandelbrot & Van Ness (1968), Dietrich & Newsam (1997).
"""

import logging
import numpy as np
from .base import BaseSynthesizer

logger = logging.getLogger(__name__)

try:
    from fbm import FBM as _FBM
    _HAS_FBM = True
except ImportError:
    _HAS_FBM = False
    logger.warning("fbm package not installed; falling back to Cholesky (slow for L>512).")


class FBMSynthesizer(BaseSynthesizer):
    """Fractional Brownian motion with H sampled uniformly in (0.1, 0.9)."""

    def __init__(self, length: int = 1024, random_seed: int = 42) -> None:
        super().__init__(length=length, random_seed=random_seed)

    @staticmethod
    def _fbm_covariance(n: int, H: float) -> np.ndarray:
        """Exact fBm covariance matrix for Cholesky fallback."""
        idx = np.arange(1, n + 1, dtype=np.float64)
        # C(i,j) = 0.5 * (|i|^{2H} + |j|^{2H} - |i-j|^{2H})
        i, j = np.meshgrid(idx, idx)
        return 0.5 * (np.abs(i)**(2*H) + np.abs(j)**(2*H) - np.abs(i - j)**(2*H))

    def _gen_single_series(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed=seed)
        H = float(rng.uniform(0.1, 0.9))
        output_type = rng.choice(["fbm", "fgn"])

        if _HAS_FBM:
            f = _FBM(n=self.length, hurst=H, length=1.0, method="daviesharte")
            # fbm package uses its own RNG; we re-seed it
            np.random.seed(seed)
            sample = f.fbm() if output_type == "fbm" else f.fgn()
            # fbm() returns length n+1 (includes t=0), fgn() returns length n
            if len(sample) == self.length + 1:
                sample = sample[1:]
            sample = sample[:self.length]
        else:
            cov = self._fbm_covariance(self.length, H)
            cov += 1e-10 * np.eye(self.length)  # jitter
            try:
                L_chol = np.linalg.cholesky(cov)
                sample = L_chol @ rng.standard_normal(self.length)
            except np.linalg.LinAlgError:
                sample = rng.standard_normal(self.length)
            if output_type == "fgn":
                sample = np.diff(sample, prepend=0.0)

        # Random scale
        scale = float(np.exp(rng.uniform(np.log(0.1), np.log(5.0))))
        sample = sample * scale

        if not np.all(np.isfinite(sample)):
            sample = rng.standard_normal(self.length)

        return sample.astype(np.float32, copy=False)[None, :]