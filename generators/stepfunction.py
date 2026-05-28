"""Piecewise-constant generator with hard/ramp/sigmoid transitions."""

import logging
import numpy as np
from .base import BaseSynthesizer

logger = logging.getLogger(__name__)


class StepFunctionSynthesizer(BaseSynthesizer):

    def __init__(self, length: int = 1024, random_seed: int = 42) -> None:
        super().__init__(length=length, random_seed=random_seed)
        self.transition_types = ["hard", "ramp", "sigmoid"]
        self.level_modes = ["uniform", "random_walk", "clustered"]

    def _sample_segment_lengths(self, n_seg, rng):
        proportions = rng.dirichlet(rng.uniform(0.5, 2.0) * np.ones(n_seg))
        lengths = np.maximum(np.floor(proportions * self.length).astype(int), 2)
        deficit = self.length - lengths.sum()
        if deficit > 0:
            for idx in rng.choice(n_seg, size=deficit, replace=True):
                lengths[idx] += 1
        elif deficit < 0:
            for _ in range(-deficit):
                longest = np.argmax(lengths)
                if lengths[longest] > 2:
                    lengths[longest] -= 1
        return lengths

    def _sample_levels(self, n_seg, mode, rng):
        if mode == "uniform":
            return rng.uniform(-5, 5, size=n_seg)
        if mode == "random_walk":
            start = rng.uniform(-3, 3)
            return np.cumsum(rng.normal(0, rng.uniform(0.3, 2.0), size=n_seg)) - rng.normal(0, rng.uniform(0.3, 2.0)) + start
        n_clusters = rng.integers(2, min(6, n_seg + 1))
        centers = rng.uniform(-5, 5, size=n_clusters)
        return centers[rng.choice(n_clusters, size=n_seg)] + rng.normal(0, 0.1, size=n_seg)

    def _apply_transition(self, series, cps, levels, ttype, rng):
        if ttype == "hard":
            return series
        result = series.copy()
        for i, cp in enumerate(cps):
            if i + 1 >= len(levels):
                break
            window = max(3, int(rng.uniform(0.01, 0.05) * self.length))
            half_w = window // 2
            start, end = max(0, cp - half_w), min(self.length, cp + half_w)
            if end - start < 2:
                continue
            t_norm = np.linspace(0, 1, end - start)
            blend = t_norm if ttype == "ramp" else 1 / (1 + np.exp(-rng.uniform(5, 15) * (t_norm - 0.5)))
            result[start:end] = levels[i] + (levels[i + 1] - levels[i]) * blend
        return result

    def _gen_single_series(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed=seed)
        n_seg = min(int(rng.geometric(p=0.15)) + 2, max(3, self.length // 4))
        level_mode = rng.choice(self.level_modes)
        ttype = rng.choice(self.transition_types, p=[0.5, 0.3, 0.2])
        seg_lengths = self._sample_segment_lengths(n_seg, rng)
        levels = self._sample_levels(n_seg, level_mode, rng)
        series = np.empty(self.length, dtype=np.float64)
        pos, cps = 0, []
        for i, sl in enumerate(seg_lengths):
            end = min(pos + sl, self.length)
            series[pos:end] = levels[i]
            if i > 0:
                cps.append(pos)
            pos = end
        series = self._apply_transition(series, cps, levels, ttype, rng)
        if rng.random() < 0.7:
            series += rng.normal(0, rng.uniform(0.01, 0.3), size=self.length)
        if not np.all(np.isfinite(series)):
            series = np.zeros(self.length, dtype=np.float64)
        return series.astype(np.float32, copy=False)[None, :]