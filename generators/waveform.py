"""Non-smooth periodic waveform generator (sawtooth, square, triangle).

High harmonic content absent from all GP-based smooth generators.
Reference: scipy.signal.sawtooth, scipy.signal.square.
"""

import numpy as np
from scipy import signal as sp_signal
from .base import BaseSynthesizer


class WaveformSynthesizer(BaseSynthesizer):
    """Sawtooth, square, triangle waveforms with randomized params."""

    def __init__(self, length: int = 1024, random_seed: int = 42) -> None:
        super().__init__(length=length, random_seed=random_seed)
        self.waveform_types = ["sawtooth", "square", "triangle"]

    def _gen_single_series(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed=seed)

        # Mix 1-3 waveforms
        n_waves = int(rng.integers(1, 4))
        t = np.linspace(0, 1, self.length, endpoint=False)
        combined = np.zeros(self.length)

        for _ in range(n_waves):
            wtype = rng.choice(self.waveform_types)
            freq = float(rng.uniform(1, 50))  # cycles over [0,1]
            amplitude = float(rng.uniform(0.3, 3.0))
            phase = float(rng.uniform(0, 2 * np.pi))
            duty = float(rng.uniform(0.2, 0.8))

            angle = 2 * np.pi * freq * t + phase

            if wtype == "sawtooth":
                # width=1 -> sawtooth, width=0.5 -> triangle
                wave = sp_signal.sawtooth(angle, width=float(rng.choice([0.0, 1.0])))
            elif wtype == "square":
                wave = sp_signal.square(angle, duty=duty)
            else:  # triangle
                wave = sp_signal.sawtooth(angle, width=0.5)

            combined += amplitude * wave

        # Optional modifications
        if rng.random() < 0.3:
            mod_freq = float(rng.uniform(0.5, 5))
            combined *= 1 + 0.5 * np.sin(2 * np.pi * mod_freq * t)

        if rng.random() < 0.7:
            combined += rng.normal(0, float(rng.uniform(0.01, 0.3)), size=self.length)

        if rng.random() < 0.3:
            combined += rng.uniform(-2, 2) * t

        return combined.astype(np.float32, copy=False)[None, :]