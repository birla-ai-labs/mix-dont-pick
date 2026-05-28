"""GP compositional kernel generator.

Upgraded from Chronos (Ansari et al., 2024) KernSynth with Matérn kernels
(Rasmussen & Williams, 2006, Ch. 4) and non-zero mean functions.
"""

import functools
import logging

import numpy as np
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Kernel,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)

from .base import BaseSynthesizer

logger = logging.getLogger(__name__)

NON_REPEAT_KERNELS = [
    DotProduct(sigma_0=0.0),
    DotProduct(sigma_0=1.0),
    DotProduct(sigma_0=10.0),
    RBF(length_scale=0.1),
    RBF(length_scale=1.0),
    RBF(length_scale=10.0),
    RationalQuadratic(alpha=0.1),
    RationalQuadratic(alpha=1.0),
    RationalQuadratic(alpha=10.0),
    Matern(nu=0.5, length_scale=0.1),
    Matern(nu=0.5, length_scale=1.0),
    Matern(nu=0.5, length_scale=10.0),
    Matern(nu=1.5, length_scale=0.1),
    Matern(nu=1.5, length_scale=1.0),
    Matern(nu=1.5, length_scale=10.0),
    Matern(nu=2.5, length_scale=0.1),
    Matern(nu=2.5, length_scale=1.0),
    Matern(nu=2.5, length_scale=10.0),
    WhiteKernel(noise_level=0.1),
    ConstantKernel(),
]


class KernelSynthesizer(BaseSynthesizer):
    """GP compositional kernel generator with Matérn + mean functions."""

    def __init__(
        self,
        length: int = 1024,
        max_kernels: int = 5,
        random_seed: int = 42,
    ) -> None:
        super().__init__(length=length, random_seed=random_seed)
        self.max_kernels = max_kernels
        self.kernel_bank = self._build_kernel_bank()
        self.x = np.linspace(0, 1, self.length)
        self.p_mean_function = 0.5

    def _build_kernel_bank(self) -> list[Kernel]:
        partial_bank = [
            ExpSineSquared(periodicity=24 / self.length),
            ExpSineSquared(periodicity=48 / self.length),
            ExpSineSquared(periodicity=96 / self.length),
            ExpSineSquared(periodicity=24 * 7 / self.length),
            ExpSineSquared(periodicity=48 * 7 / self.length),
            ExpSineSquared(periodicity=96 * 7 / self.length),
            ExpSineSquared(periodicity=7 / self.length),
            ExpSineSquared(periodicity=14 / self.length),
            ExpSineSquared(periodicity=30 / self.length),
            ExpSineSquared(periodicity=60 / self.length),
            ExpSineSquared(periodicity=365 / self.length),
            ExpSineSquared(periodicity=365 * 2 / self.length),
            ExpSineSquared(periodicity=4 / self.length),
            ExpSineSquared(periodicity=26 / self.length),
            ExpSineSquared(periodicity=52 / self.length),
            ExpSineSquared(periodicity=4 / self.length),
            ExpSineSquared(periodicity=6 / self.length),
            ExpSineSquared(periodicity=12 / self.length),
            ExpSineSquared(periodicity=4 / self.length),
            ExpSineSquared(periodicity=4 * 10 / self.length),
            ExpSineSquared(periodicity=10 / self.length),
        ]
        return partial_bank + NON_REPEAT_KERNELS

    def _random_binary_map(self, a: Kernel, b: Kernel, rng: np.random.Generator) -> Kernel:
        return a + b if rng.random() < 0.8 else a * b

    def _sample_from_gp_prior(
        self, kernel: Kernel, x: np.ndarray | None = None, random_seed: int | None = None,
    ) -> np.ndarray:
        if x is None:
            x = self.x
        if x.ndim == 1:
            x = x[:, None]
        cov = kernel(x)
        rng = np.random.default_rng(seed=random_seed) if random_seed is not None else self.rng
        return rng.multivariate_normal(mean=np.zeros(x.shape[0]), cov=cov, method="eigh")

    def _sample_mean_function(self, rng: np.random.Generator) -> np.ndarray:
        if rng.random() > self.p_mean_function:
            return np.zeros(self.length)
        t = self.x
        mean_type = rng.choice(["linear", "quadratic", "sinusoidal", "log_linear", "random_walk"])
        if mean_type == "linear":
            return rng.uniform(-3, 3) * t + rng.uniform(-2, 2)
        if mean_type == "quadratic":
            return rng.uniform(-2, 2) * t**2 + rng.uniform(-2, 2) * t + rng.uniform(-1, 1)
        if mean_type == "sinusoidal":
            return rng.uniform(0.5, 3) * np.sin(2 * np.pi * rng.uniform(1, 8) * t + rng.uniform(0, 2 * np.pi))
        if mean_type == "log_linear":
            return rng.uniform(-3, 3) * np.log1p(rng.uniform(1, 20) * t) + rng.uniform(-1, 1)
        step_std = rng.uniform(0.01, 0.1)
        return np.cumsum(rng.normal(0, step_std, size=self.length))

    def _gen_single_series(self, seed: int) -> np.ndarray:
        local_rng = np.random.default_rng(seed=seed)
        num_kernels = local_rng.integers(1, self.max_kernels + 1)
        selected = local_rng.choice(self.kernel_bank, num_kernels, replace=True)
        kernel = functools.reduce(lambda a, b: self._random_binary_map(a, b, local_rng), selected)
        try:
            gp_sample = self._sample_from_gp_prior(kernel=kernel, random_seed=seed)
            if len(gp_sample) == 0:
                raise ValueError("Empty series")
        except np.linalg.LinAlgError:
            gp_sample = self._sample_from_gp_prior(
                kernel=ConstantKernel() + WhiteKernel(noise_level=0.1), random_seed=seed,
            )
        series = gp_sample + self._sample_mean_function(local_rng)
        return series.astype(np.float32, copy=False)[None, :]