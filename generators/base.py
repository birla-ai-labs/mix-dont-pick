"""Base class for all synthetic time series generators."""

import logging
import time
from tqdm import tqdm
import sys
import numpy as np

logger = logging.getLogger(__name__)


class BaseSynthesizer:
    """ABC for synthetic time series generators.

    Subclasses must implement:
        _gen_single_series(seed: int) -> np.ndarray of shape (1, length)

    Convention from tsfm-synthetic repo:
        - __init__ takes (length, random_seed, **kwargs)
        - self.rng = np.random.default_rng(seed=random_seed)
        - self.length = length
    """

    def __init__(self, length: int = 1024, random_seed: int = 42) -> None:
        self.length = length
        self.random_seed = random_seed
        self.rng = np.random.default_rng(seed=random_seed)

    def _gen_single_series(self, seed: int) -> np.ndarray:
        raise NotImplementedError

    def generate_series(self, n: int) -> np.ndarray:
        """Generate n series, shape (n, length), float32."""
        seeds = self.rng.integers(0, 2**31, size=n)
        if n == 1:
            return self._gen_single_series(int(seeds[0]))

        if sys.stderr.isatty():
            iterator = tqdm(seeds, desc=f"Generating {self.__class__.__name__}")
        else:
            iterator = seeds
            logger.info("Generating %d series (%s)...", n, self.__class__.__name__)

        results = [self._gen_single_series(int(s)) for s in iterator]
        return np.vstack(results).astype(np.float32, copy=False)

    def generate_corpus(
        self,
        n: int = 1_000_000,
        seed: int | None = None,
    ) -> tuple[np.ndarray, float]:
        """Generate a full corpus with timing.

        Parameters
        ----------
        n : int
            Number of series.
        seed : int, optional
            If provided, re-seeds self.rng before generation.

        Returns
        -------
        corpus : np.ndarray, shape (n, self.length), float32
        elapsed : float, wall-clock seconds
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
        t0 = time.perf_counter()
        corpus = self.generate_series(n)
        elapsed = time.perf_counter() - t0
        logger.info(
            "%s: generated %d x %d in %.1fs",
            self.__class__.__name__, n, self.length, elapsed,
        )
        return corpus, elapsed