"""Chaotic dynamical system generators: Lorenz attractor and Mackey-Glass.

References:
- Lorenz (1963), "Deterministic Nonperiodic Flow", J. Atmos. Sci.
- Mackey & Glass (1977), "Oscillation and Chaos in Physiological Control Systems", Science.
"""

import logging
import numpy as np
from .base import BaseSynthesizer

logger = logging.getLogger(__name__)


class ChaoticSynthesizer(BaseSynthesizer):
    """Lorenz and Mackey-Glass chaotic time series."""

    def __init__(self, length: int = 1024, random_seed: int = 42) -> None:
        super().__init__(length=length, random_seed=random_seed)
        self.burn_in = 2000

    def _lorenz_rk4(self, sigma, rho, beta_1, dt, n_steps, x0, rng):
        """4th-order Runge-Kutta for the Lorenz system."""
        x, y, z = x0
        trajectory = np.empty((n_steps, 3))
        for i in range(n_steps):
            trajectory[i] = [x, y, z]
            dx1 = sigma * (y - x)
            dy1 = x * (rho - z) - y
            dz1 = x * y - beta_1 * z

            xm = x + 0.5 * dt * dx1
            ym = y + 0.5 * dt * dy1
            zm = z + 0.5 * dt * dz1

            dx2 = sigma * (ym - xm)
            dy2 = xm * (rho - zm) - ym
            dz2 = xm * ym - beta_1 * zm

            xm2 = x + 0.5 * dt * dx2
            ym2 = y + 0.5 * dt * dy2
            zm2 = z + 0.5 * dt * dz2

            dx3 = sigma * (ym2 - xm2)
            dy3 = xm2 * (rho - zm2) - ym2
            dz3 = xm2 * ym2 - beta_1 * zm2

            xe = x + dt * dx3
            ye = y + dt * dy3
            ze = z + dt * dz3

            dx4 = sigma * (ye - xe)
            dy4 = xe * (rho - ze) - ye
            dz4 = xe * ye - beta_1 * ze

            x += dt / 6 * (dx1 + 2*dx2 + 2*dx3 + dx4)
            y += dt / 6 * (dy1 + 2*dy2 + 2*dy3 + dy4)
            z += dt / 6 * (dz1 + 2*dz2 + 2*dz3 + dz4)
        return trajectory

    def _mackey_glass(self, tau, n_steps, beta_mg, gamma_mg, n_exp, dt, rng):
        """Euler integration of Mackey-Glass delay differential equation.

        dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)
        """
        history_len = int(tau / dt) + 1
        total = history_len + n_steps
        x = np.empty(total)
        x[:history_len] = 0.9 + rng.uniform(-0.1, 0.1, size=history_len)
        for i in range(history_len, total):
            x_tau = x[i - history_len]
            x[i] = x[i-1] + dt * (beta_mg * x_tau / (1 + x_tau**n_exp) - gamma_mg * x[i-1])
            if not np.isfinite(x[i]):
                x[i] = x[i-1]
        return x[history_len:]

    def _gen_single_series(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed=seed)
        total_steps = self.length + self.burn_in
        system = rng.choice(["lorenz", "mackey_glass"])

        if system == "lorenz":
            sigma = float(rng.uniform(8, 12))
            rho = float(rng.uniform(24, 30))
            beta_l = float(rng.uniform(2.0, 3.5))
            dt = float(rng.uniform(0.005, 0.02))
            x0 = rng.uniform(-1, 1, size=3)
            traj = self._lorenz_rk4(sigma, rho, beta_l, dt, total_steps, x0, rng)
            # Pick a random coordinate
            coord = int(rng.integers(0, 3))
            series = traj[self.burn_in:, coord]
        else:
            tau = float(rng.uniform(15, 30))
            n_mg = float(rng.choice([8, 9, 10, 11, 12]))
            beta_mg = float(rng.uniform(0.15, 0.25))
            gamma_mg = float(rng.uniform(0.05, 0.15))
            dt = float(rng.uniform(0.5, 2.0))
            raw = self._mackey_glass(tau, total_steps, beta_mg, gamma_mg, n_mg, dt, rng)            
            series = raw[self.burn_in:]

        if len(series) > self.length:
            series = series[:self.length]
        elif len(series) < self.length:
            series = np.pad(series, (0, self.length - len(series)), mode='edge')

        if not np.all(np.isfinite(series)):
            series = rng.standard_normal(self.length)

        return series.astype(np.float32, copy=False)[None, :]