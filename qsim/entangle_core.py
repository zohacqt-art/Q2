"""Pure entanglement maths: SPDC photon pairs, polarisation correlations, CHSH."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

SQRT2 = np.sqrt(2.0)

# |Phi+> = (|HH> + |VV>) / sqrt(2), basis order: HH, HV, VH, VV
PHI_PLUS = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / SQRT2
PHI_MINUS = np.array([1.0, 0.0, 0.0, -1.0], dtype=complex) / SQRT2
PSI_PLUS = np.array([0.0, 1.0, 1.0, 0.0], dtype=complex) / SQRT2
PSI_MINUS = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / SQRT2

BELL_STATES: Dict[str, np.ndarray] = {
    "|Φ+⟩ = (|HH⟩ + |VV⟩)/√2": PHI_PLUS,
    "|Φ−⟩ = (|HH⟩ − |VV⟩)/√2": PHI_MINUS,
    "|Ψ+⟩ = (|HV⟩ + |VH⟩)/√2": PSI_PLUS,
    "|Ψ−⟩ = (|HV⟩ − |VH⟩)/√2": PSI_MINUS,
}


def analyzer_vectors(theta_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Transmitted (+) and reflected (-) polarisation vectors of a polariser."""
    t = np.deg2rad(theta_deg)
    plus = np.array([np.cos(t), np.sin(t)], dtype=complex)
    minus = np.array([-np.sin(t), np.cos(t)], dtype=complex)
    return plus, minus


def joint_probabilities(state: np.ndarray, a_deg: float,
                        b_deg: float) -> Dict[str, float]:
    """P(++), P(+-), P(-+), P(--) for analyzer angles a and b."""
    a_p, a_m = analyzer_vectors(a_deg)
    b_p, b_m = analyzer_vectors(b_deg)
    out: Dict[str, float] = {}
    for a_lbl, a_vec in (("+", a_p), ("-", a_m)):
        for b_lbl, b_vec in (("+", b_p), ("-", b_m)):
            proj = np.kron(a_vec, b_vec)
            amp = np.vdot(proj, state)
            out[a_lbl + b_lbl] = float(np.abs(amp) ** 2)
    total = sum(out.values())
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    return out


def correlation(state: np.ndarray, a_deg: float, b_deg: float) -> float:
    """E(a,b) = P(++) + P(--) - P(+-) - P(-+), in [-1, 1]."""
    p = joint_probabilities(state, a_deg, b_deg)
    return p["++"] + p["--"] - p["+-"] - p["-+"]


def quantum_correlation_curve(state: np.ndarray, a_deg: float,
                              b_grid: np.ndarray) -> np.ndarray:
    return np.array([correlation(state, a_deg, float(b)) for b in b_grid])


def classical_correlation(delta_deg: float) -> float:
    """Triangle-wave LHV correlation, period 180 degrees, E(0)=1, E(90)=-1."""
    d = abs(float(delta_deg)) % 180.0
    d = min(d, 180.0 - d)          # fold into 0..90
    return 1.0 - 2.0 * (d / 90.0)


def classical_correlation_curve(b_grid: np.ndarray, a_deg: float = 0.0) -> np.ndarray:
    """Best local hidden-variable model: a triangular correlation curve."""
    return np.array([classical_correlation(float(b) - a_deg) for b in np.asarray(b_grid)])


@dataclass
class TrialResult:
    n_pairs: int
    coincidences: Dict[str, int]
    measured_E: float
    theory_E: float

    @property
    def visibility(self) -> float:
        return abs(self.measured_E)


def simulate_pairs(state: np.ndarray, a_deg: float, b_deg: float,
                   n_pairs: int = 500,
                   rng: np.random.Generator | None = None) -> TrialResult:
    """Monte-Carlo detection of n entangled pairs at the two analyzers."""
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")
    g = rng or np.random.default_rng()
    probs = joint_probabilities(state, a_deg, b_deg)
    labels = ["++", "+-", "-+", "--"]
    weights = np.array([probs[k] for k in labels], dtype=float)
    weights = np.clip(weights, 0.0, None)
    weights /= weights.sum()
    draws = g.choice(4, size=n_pairs, p=weights)
    counts = {lbl: int(np.sum(draws == i)) for i, lbl in enumerate(labels)}
    e = (counts["++"] + counts["--"] - counts["+-"] - counts["-+"]) / n_pairs
    return TrialResult(n_pairs, counts, float(e), correlation(state, a_deg, b_deg))


def chsh_S(state: np.ndarray, a1: float = 0.0, a2: float = 45.0,
           b1: float = 22.5, b2: float = 67.5) -> float:
    """CHSH parameter. |S| <= 2 classically, up to 2*sqrt(2) quantum mechanically."""
    return abs(
        correlation(state, a1, b1)
        - correlation(state, a1, b2)
        + correlation(state, a2, b1)
        + correlation(state, a2, b2)
    )


def spdc_cone_points(n: int = 400, half_angle: float = 3.0,
                     rng: np.random.Generator | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Signal/idler emission angles on the type-II SPDC cones (for the visual)."""
    g = rng or np.random.default_rng()
    phi = g.uniform(0, 2 * np.pi, n)
    jitter = g.normal(0, 0.12, n)
    r = half_angle + jitter
    return r * np.cos(phi), r * np.sin(phi)


def pair_rate(pump_mw: float, crystal_mm: float, efficiency: float = 1e-6) -> float:
    """Toy SPDC brightness model: pairs per second."""
    return max(0.0, pump_mw) * max(0.0, crystal_mm) * efficiency * 1e9
