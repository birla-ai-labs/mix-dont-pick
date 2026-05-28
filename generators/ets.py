"""ETS (Error-Trend-Seasonality) state-space generator.

Covers multiplicative error AND multiplicative seasonality — absent from
all existing TSFM synthetic pipelines.
Reference: Hyndman et al. (2008), "Forecasting with Exponential Smoothing", Ch. 2-3.
"""

import logging
import numpy as np
from .base import BaseSynthesizer

logger = logging.getLogger(__name__)


class ETSSynthesizer(BaseSynthesizer):
    """ETS state-space generator with add/mult error, trend, seasonality."""

    def __init__(self, length: int = 1024, random_seed: int = 42) -> None:
        super().__init__(length=length, random_seed=random_seed)
        self.burn_in = 200

    def _gen_single_series(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed=seed)
        total_len = self.length + self.burn_in

        error = rng.choice(["A", "M"])
        trend = rng.choice(["N", "A", "Ad"])
        season = rng.choice(["N", "A", "M"])
        period = int(rng.choice([4, 7, 12, 24, 52])) if season != "N" else 1

        alpha = float(rng.uniform(0.01, 0.3))
        beta = float(rng.uniform(0.001, 0.1)) if trend != "N" else 0.0
        gamma = float(rng.uniform(0.001, 0.15)) if season != "N" else 0.0
        phi = float(rng.uniform(0.8, 0.98)) if trend == "Ad" else 1.0

        if error == "M":
            sigma = float(rng.uniform(0.005, 0.05))
        else:
            sigma = float(np.exp(rng.uniform(np.log(0.01), np.log(0.3))))

        level = float(rng.uniform(10, 50))
        b = float(rng.uniform(-0.1, 0.1)) if trend != "N" else 0.0
        if season == "A":
            s = rng.normal(0, level * 0.05, size=period)
            s -= s.mean()
        elif season == "M":
            s = rng.uniform(0.85, 1.15, size=period)
            s = s / s.mean()
        else:
            s = np.ones(period)

        y = np.empty(total_len)
        LEVEL_FLOOR = 1e-2
        LEVEL_CEIL = 1e6

        for t in range(total_len):
            si = s[t % period]

            if season == "M":
                yhat = (level + phi * b) * si
            elif season == "A":
                yhat = level + phi * b + si
            else:
                yhat = level + phi * b

            if error == "A":
                e = rng.normal(0, sigma * abs(level))
                y[t] = yhat + e
            else:
                e = rng.normal(0, sigma)
                y[t] = yhat * (1 + e)

            if error == "A":
                err_term = e
            else:
                err_term = e * yhat if abs(yhat) > 1e-10 else e

            level_old = level
            if season == "M":
                level = level + phi * b + alpha * err_term / max(abs(si), 1e-10)
            else:
                level = level + phi * b + alpha * err_term

            level = np.clip(level, -LEVEL_CEIL, LEVEL_CEIL)
            if abs(level) < LEVEL_FLOOR and error == "M":
                level = np.sign(level) * LEVEL_FLOOR if level != 0 else LEVEL_FLOOR

            if trend != "N":
                b = phi * b + beta * (level - level_old)
                b = np.clip(b, -10.0, 10.0)

            if season == "A":
                s[t % period] = si + gamma * err_term
            elif season == "M":
                denom = max(abs(level_old + phi * b), 1e-10)
                s[t % period] = si + gamma * err_term / denom
                s[t % period] = np.clip(s[t % period], 0.1, 10.0)

        y = y[self.burn_in:]

        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e8:
            # Fallback: simple additive ETS
            level = float(rng.uniform(10, 50))
            y = np.empty(self.length)
            for t in range(self.length):
                e = rng.normal(0, 0.1 * level)
                y[t] = level + e
                level = level + 0.05 * e
            
        return y.astype(np.float32, copy=False)[None, :]