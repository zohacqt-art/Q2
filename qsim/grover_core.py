"""Pure Grover-search maths. No Streamlit imports here, so it is unit-testable."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List

import numpy as np

N_BOXES = 16
N_QUBITS = 4


def uniform_state(n: int = N_BOXES) -> np.ndarray:
    """Equal superposition produced by H^{otimes n} on |0...0>."""
    return np.full(n, 1.0 / math.sqrt(n), dtype=float)


def oracle_flip(amps: np.ndarray, target_index: int) -> np.ndarray:
    """Phase oracle: |x> -> -|x> for the marked item, identity otherwise."""
    out = np.array(amps, dtype=float, copy=True)
    out[target_index] *= -1.0
    return out


def diffusion(amps: np.ndarray) -> np.ndarray:
    """Grover diffuser: inversion about the mean amplitude."""
    arr = np.array(amps, dtype=float, copy=True)
    mean = arr.mean()
    return 2.0 * mean - arr


def probabilities(amps: np.ndarray) -> np.ndarray:
    p = np.abs(np.asarray(amps, dtype=float)) ** 2
    total = p.sum()
    return p / total if total > 0 else p


def optimal_iterations(n: int = N_BOXES) -> int:
    """floor(pi/4 * sqrt(N)) -- 3 for N = 16."""
    return max(1, int(math.floor((math.pi / 4.0) * math.sqrt(n))))


@dataclass
class GroverFrame:
    """One animation frame of the algorithm."""

    step: int
    stage: str  # "init" | "oracle" | "diffusion"
    label: str
    amps: np.ndarray
    target_prob: float


def grover_frames(target_box: int, iterations: int | None = None,
                  n: int = N_BOXES) -> List[GroverFrame]:
    """Full amplitude history: init, then (oracle, diffusion) per iteration."""
    if not 1 <= target_box <= n:
        raise ValueError(f"target_box must be in 1..{n}, got {target_box}")
    if iterations is None:
        iterations = optimal_iterations(n)

    idx = target_box - 1
    amps = uniform_state(n)
    frames = [GroverFrame(0, "init", "Hadamard - equal superposition",
                          amps, float(probabilities(amps)[idx]))]

    for k in range(1, iterations + 1):
        amps = oracle_flip(amps, idx)
        frames.append(GroverFrame(k, "oracle", f"Iteration {k}: oracle phase flip",
                                  amps, float(probabilities(amps)[idx])))
        amps = diffusion(amps)
        frames.append(GroverFrame(k, "diffusion", f"Iteration {k}: diffusion (amplify)",
                                  amps, float(probabilities(amps)[idx])))
    return frames


def measure(amps: np.ndarray, rng: random.Random | None = None) -> int:
    """Collapse the register; returns a 1-based box number."""
    probs = probabilities(amps)
    r = (rng or random).random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += float(p)
        if r <= acc:
            return i + 1
    return len(probs)


@dataclass
class ClassicalRun:
    order: List[int] = field(default_factory=list)  # boxes opened, in order
    checks: int = 0                                 # boxes opened until found
    found: int = 0                                  # box that held the object


def classical_search(target_box: int, n: int = N_BOXES,
                     rng: random.Random | None = None) -> ClassicalRun:
    """Brute force: open boxes one at a time in random order until the hit."""
    r = rng or random
    order = list(range(1, n + 1))
    r.shuffle(order)
    opened: List[int] = []
    for box in order:
        opened.append(box)
        if box == target_box:
            break
    return ClassicalRun(order=opened, checks=len(opened), found=target_box)


def classical_expected_checks(n: int = N_BOXES) -> float:
    """Average boxes opened by brute force = (N + 1) / 2."""
    return (n + 1) / 2.0


def success_probability(target_box: int, iterations: int | None = None,
                        n: int = N_BOXES) -> float:
    frames = grover_frames(target_box, iterations, n)
    return frames[-1].target_prob
