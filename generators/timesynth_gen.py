"""TimeSynth signal-mixing generator."""

import logging

import numpy as np
import timesynth as ts

from .base import BaseSynthesizer

logger = logging.getLogger(__name__)


class TimeSynthesizer(BaseSynthesizer):
    """TimeSynth-based randomized signal-mixing generator."""

    def __init__(
        self,
        length: int = 1024,
        max_signals: int = 3,
        sampling_frequency: float = 1.0,
        random_seed: int = 42,
    ) -> None:
        super().__init__(length=length, random_seed=random_seed)
        self.max_signals = max_signals
        self.sampling_frequency = sampling_frequency
        self.random_seed = random_seed
        self.irregular_prob = self.rng.uniform(0.1, 0.3)
        self.signal_types = ["sinusoidal", "car", "narma", "pseudoperiodic", "autoregressive"]
        self.noise_bank = (
            [{"type": "gaussian", "std": s} for s in [0.02, 0.05, 0.1, 0.2, 0.3]]
            + [{"type": "red", "std": s, "tau": t} for s in [0.05, 0.15, 0.25] for t in [2.0, 5.0, 10.0]]
        )

    def _create_signal(self, signal_type: str, rng: np.random.Generator):
        if signal_type == "sinusoidal":
            p = int(rng.integers(4, max(8, self.length // 8)))
            stop_time = self.length / self.sampling_frequency
            return ts.signals.Sinusoidal(
                amplitude=float(rng.uniform(0.5, 3.0)),
                frequency=float((self.length / p) / stop_time),
                ftype=rng.choice([np.sin, np.cos]),
            )
        if signal_type == "car":
            return ts.signals.CAR(
                ar_param=float(rng.uniform(0.1, 0.95)),
                sigma=float(rng.uniform(0.1, 0.8)),
                start_value=float(rng.uniform(-0.5, 0.5)),
            )
        if signal_type == "narma":
            order = int(rng.integers(5, 15))
            return ts.signals.NARMA(
                order=order,
                coefficients=[float(rng.uniform(0.6, 0.9)), float(rng.uniform(0.02, 0.08)),
                              float(rng.uniform(1.0, 2.0)), float(rng.uniform(0.05, 0.15))],
                initial_condition=rng.normal(0, 0.1, size=order),
            )
        if signal_type == "pseudoperiodic":
            p = int(rng.integers(6, max(12, self.length // 10)))
            return ts.signals.PseudoPeriodic(
                amplitude=float(rng.uniform(0.5, 2.5)),
                frequency=int(p),
                ampSD=float(rng.uniform(0.05, 0.25)),
                freqSD=float(rng.uniform(0.01, 0.15) * p),
                ftype=rng.choice([np.sin, np.cos]),
            )
        if signal_type == "autoregressive":
            p = int(rng.integers(1, 4))
            phi = rng.uniform(-0.6, 0.6, size=p)
            while np.sum(np.abs(phi)) > 0.9:
                phi = rng.uniform(-0.6, 0.6, size=p)
            return ts.signals.AutoRegressive(
                ar_param=phi.tolist(),
                sigma=float(rng.uniform(0.1, 0.7)),
                start_value=[np.array([v], dtype=float) for v in rng.normal(0, 0.2, size=p)],
            )
        raise ValueError(f"Unknown signal type: {signal_type}")

    def _gen_single_series(self, seed: int) -> np.ndarray:
        np.random.seed(seed)
        local_rng = np.random.default_rng(seed=seed)
        try:
            n_signals = local_rng.integers(1, self.max_signals + 1)
            selected = local_rng.choice(self.signal_types, size=n_signals, replace=True,
                                        p=[0.2, 0.25, 0.2, 0.15, 0.2])
            stop_time = self.length / self.sampling_frequency
            sampler = ts.TimeSampler(stop_time=stop_time)
            need_regular = any(s in {"autoregressive", "narma"} for s in selected)
            try:
                if need_regular or local_rng.random() >= self.irregular_prob:
                    time_samples = sampler.sample_regular_time(num_points=self.length)
                else:
                    time_samples = sampler.sample_irregular_time(
                        num_points=self.length, keep_percentage=float(local_rng.uniform(80, 95)),
                    )
            except (ValueError, RuntimeError):
                time_samples = sampler.sample_regular_time(num_points=self.length)

            combined = None
            for sig_type in selected:
                sig = self._create_signal(sig_type, local_rng)
                w = float(local_rng.uniform(0.4, 1.2))
                comp = np.asarray(ts.TimeSeries(signal_generator=sig).sample(time_samples)[0], dtype=np.float32)
                if combined is None:
                    combined = w * comp
                elif local_rng.random() < 0.8:
                    combined = combined + w * comp
                else:
                    combined = combined * (1 + 0.05 * w * comp)

            noise_cfg = local_rng.choice(self.noise_bank)
            noise_gen = ts.noise.GaussianNoise(std=noise_cfg["std"]) if noise_cfg["type"] == "gaussian" else ts.noise.RedNoise(std=noise_cfg["std"], tau=noise_cfg["tau"])
            noise_ts = ts.TimeSeries(ts.signals.Sinusoidal(amplitude=0.0, frequency=0.1), noise_generator=noise_gen)
            samples = combined + noise_ts.sample(time_samples)[0]
            if len(samples) != self.length:
                target_times = np.linspace(time_samples[0], time_samples[-1], self.length)
                samples = np.interp(target_times, time_samples, samples)
            return np.asarray(samples, dtype=np.float32)[None, :]
        except (ValueError, TypeError, RuntimeError):
            fb = ts.TimeSeries(ts.signals.Sinusoidal(amplitude=1.0, frequency=0.1),
                               noise_generator=ts.noise.GaussianNoise(std=0.1))
            t = ts.TimeSampler(stop_time=self.length / self.sampling_frequency).sample_regular_time(num_points=self.length)
            return np.asarray(fb.sample(t)[0], dtype=np.float32)[None, :]

    def generate_series(self, n: int) -> np.ndarray:
        seeds = [self.random_seed + i for i in range(n)]
        if n == 1:
            return self._gen_single_series(seeds[0])
        from tqdm import tqdm
        return np.vstack([self._gen_single_series(int(s)) for s in tqdm(seeds, desc="Generating TimeSynth")])