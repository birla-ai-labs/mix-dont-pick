"""Trend-Seasonality-Irregularity decomposition generator (Chronos-2 style)."""

import numpy as np
from .base import BaseSynthesizer


class TSISynthesizer(BaseSynthesizer):
    """TSI composition with 8 models (additive, multiplicative, mixed)."""

    def __init__(self, length: int = 1024, random_seed: int = 42) -> None:
        super().__init__(length=length, random_seed=random_seed)
        self.trend_types = ["linear", "nonlinear", "none"]
        self.season_types = ["sinusoidal", "step", "triangular", "impulsive", "none"]
        self.irreg_types = ["fgn", "fbm", "white"]
        self.models = list(range(1, 9))

    def _generate_trend(self, trend_type: str) -> np.ndarray:
        t = np.linspace(0, 1, self.length)
        if trend_type == "linear":
            return self.rng.uniform(-2, 2) * t + self.rng.uniform(-1, 1)
        if trend_type == "nonlinear":
            a, b, c, d = self.rng.uniform(-1, 1), self.rng.uniform(-1, 1), self.rng.uniform(-2, 2), self.rng.uniform(-1, 1)
            return (a * t + b) * np.sin(2 * np.pi * t) + c * t + d
        return np.zeros(self.length)

    def _generate_seasonality(self, season_type: str) -> np.ndarray:
        t = np.arange(self.length)
        if season_type == "sinusoidal":
            period = self.rng.integers(4, max(8, self.length // 8))
            return float(self.rng.uniform(0.5, 2.0)) * np.sin(2 * np.pi / period * t + self.rng.uniform(0, 2 * np.pi))
        if season_type == "step":
            period = int(self.rng.integers(4, max(8, self.length // 8)))
            amp = float(self.rng.uniform(0.5, 2.0))
            return np.array([amp if (i % (2 * period)) < period else 0.0 for i in range(self.length)])
        if season_type == "triangular":
            period = int(self.rng.integers(4, max(8, self.length // 8)))
            amp = float(self.rng.uniform(0.5, 2.0))
            s = np.empty(self.length)
            for i in range(self.length):
                cp = i % (2 * period)
                s[i] = amp * (cp / period) if cp < period else amp * (2 - cp / period)
            return s
        if season_type == "impulsive":
            period = int(self.rng.integers(4, max(8, self.length // 8)))
            amp = float(self.rng.uniform(0.5, 2.0))
            s = np.zeros(self.length)
            s[::period] = amp
            return np.convolve(s, np.ones(3) / 3, mode="same")
        return np.zeros(self.length)

    def _generate_irregularity(self, irreg_type: str) -> np.ndarray:
        if irreg_type == "fgn":
            H = self.rng.uniform(0.3, 0.7)
            noise = self.rng.normal(0, 0.1, self.length)
            for i in range(1, self.length):
                noise[i] += H * noise[i - 1]
            return noise
        if irreg_type == "fbm":
            noise = self.rng.normal(0, 0.1, self.length)
            cumul = np.cumsum(noise)
            return cumul * (self.rng.uniform(0.5, 0.9) / np.std(cumul))
        if irreg_type == "white":
            return self.rng.normal(0, 0.2, self.length)
        return np.zeros(self.length)

    def _combine(self, T: np.ndarray, S: np.ndarray, N: np.ndarray, model: int) -> np.ndarray:
        if model == 1:
            return T + S + N
        if model == 2:
            return (T + S) * (1 + 0.1 * N)
        if model == 3:
            return (T + N) * (1 + 0.1 * S)
        if model == 4:
            return (S + N) * (1 + 0.1 * T)
        if model == 5: 
            return T * (1 + 0.1 * S) + N
        if model == 6: 
            return T * (1 + 0.1 * N) + S
        if model == 7: 
            return S * (1 + 0.1 * N) + T
        if model == 8: 
            return T * (1 + 0.1 * S) * (1 + 0.1 * N)
        return T + S + N

    def _gen_single_series(self, seed: int) -> np.ndarray:
        # TSI uses self.rng directly (original pattern — sequential, not per-seed)
        T = self._generate_trend(self.rng.choice(self.trend_types))
        S = self._generate_seasonality(self.rng.choice(self.season_types))
        N = self._generate_irregularity(self.rng.choice(self.irreg_types))
        return self._combine(T, S, N, self.rng.choice(self.models)).astype(np.float32)[None, :]

    def generate_series(self, n: int) -> np.ndarray:
        from tqdm import tqdm
        return np.vstack([self._gen_single_series(i) for i in tqdm(range(n), desc="Generating TSI")])