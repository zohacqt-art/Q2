"""Pure 1-D wavefunction maths: packet, free evolution, measurement, collapse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

HBAR = 1.0
MASS = 1.0

X_MIN, X_MAX, N_POINTS = -20.0, 20.0, 512


def make_grid(n: int = N_POINTS, x_min: float = X_MIN, x_max: float = X_MAX) -> np.ndarray:
    return np.linspace(x_min, x_max, n, endpoint=False)


def gaussian_packet(x: np.ndarray, x0: float = -6.0, sigma: float = 1.5,
                    k0: float = 2.0) -> np.ndarray:
    """Normalised minimum-uncertainty packet centred at x0 with momentum hbar*k0."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    psi = np.exp(-((x - x0) ** 2) / (4.0 * sigma ** 2)) * np.exp(1j * k0 * x)
    return normalise(psi, x)


def two_slit_packet(x: np.ndarray, sep: float = 5.0, sigma: float = 1.2,
                    k0: float = 0.0) -> np.ndarray:
    """Superposition of two packets - the textbook 'cat' / double-slit state."""
    left = np.exp(-((x + sep / 2) ** 2) / (4.0 * sigma ** 2))
    right = np.exp(-((x - sep / 2) ** 2) / (4.0 * sigma ** 2))
    psi = (left + right) * np.exp(1j * k0 * x)
    return normalise(psi, x)


def normalise(psi: np.ndarray, x: np.ndarray) -> np.ndarray:
    dx = float(x[1] - x[0])
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    if norm == 0:
        raise ValueError("cannot normalise a zero wavefunction")
    return psi / norm


def density(psi: np.ndarray, x: np.ndarray) -> np.ndarray:
    """|psi|^2 as a probability density (integrates to 1 over the grid)."""
    dx = float(x[1] - x[0])
    p = np.abs(psi) ** 2
    total = p.sum() * dx
    return p / total if total > 0 else p


def evolve(psi: np.ndarray, x: np.ndarray, dt: float,
           potential: np.ndarray | None = None) -> np.ndarray:
    """One split-operator step. Free particle when potential is None."""
    dx = float(x[1] - x[0])
    k = 2.0 * np.pi * np.fft.fftfreq(x.size, d=dx)
    kinetic = np.exp(-1j * HBAR * (k ** 2) * dt / (2.0 * MASS))

    if potential is None:
        out = np.fft.ifft(kinetic * np.fft.fft(psi))
    else:
        half_v = np.exp(-0.5j * potential * dt / HBAR)
        out = half_v * psi
        out = np.fft.ifft(kinetic * np.fft.fft(out))
        out = half_v * out
    return normalise(out, x)


def barrier_potential(x: np.ndarray, height: float = 0.0, width: float = 1.0,
                      centre: float = 0.0) -> np.ndarray:
    v = np.zeros_like(x, dtype=float)
    if height != 0.0:
        v[np.abs(x - centre) <= width / 2.0] = height
    return v


def expectation_x(psi: np.ndarray, x: np.ndarray) -> float:
    dx = float(x[1] - x[0])
    return float(np.sum(x * density(psi, x)) * dx)


def uncertainty_x(psi: np.ndarray, x: np.ndarray) -> float:
    dx = float(x[1] - x[0])
    mean = expectation_x(psi, x)
    var = float(np.sum((x - mean) ** 2 * density(psi, x)) * dx)
    return float(np.sqrt(max(var, 0.0)))


def sample_positions(psi: np.ndarray, x: np.ndarray, n_samples: int = 50,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Born rule sampling: draw n outcomes from |psi(x)|^2 without collapsing."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    g = rng or np.random.default_rng()
    dx = float(x[1] - x[0])
    weights = density(psi, x) * dx
    weights = np.clip(weights, 0.0, None)
    weights = weights / weights.sum()
    idx = g.choice(x.size, size=n_samples, p=weights)
    # jitter inside the cell so histograms look continuous
    return x[idx] + g.uniform(-dx / 2, dx / 2, size=n_samples)


def collapse(psi: np.ndarray, x: np.ndarray, measured_x: float,
             width: float = 0.35) -> np.ndarray:
    """Projective position measurement: state jumps to a narrow packet at measured_x."""
    collapsed = np.exp(-((x - measured_x) ** 2) / (4.0 * width ** 2)).astype(complex)
    return normalise(collapsed, x)


def measure_once(psi: np.ndarray, x: np.ndarray,
                 rng: np.random.Generator | None = None,
                 width: float = 0.35) -> Tuple[float, np.ndarray]:
    """Single measurement: returns (outcome, post-measurement state)."""
    outcome = float(sample_positions(psi, x, 1, rng)[0])
    return outcome, collapse(psi, x, outcome, width)


@dataclass
class WaveSnapshot:
    t: float
    psi: np.ndarray


def evolve_series(psi: np.ndarray, x: np.ndarray, dt: float, steps: int,
                  potential: np.ndarray | None = None) -> List[WaveSnapshot]:
    """Animation frames of free/potential evolution."""
    out: List[WaveSnapshot] = [WaveSnapshot(0.0, psi)]
    cur = psi
    for i in range(1, steps + 1):
        cur = evolve(cur, x, dt, potential)
        out.append(WaveSnapshot(i * dt, cur))
    return out
