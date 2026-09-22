"""Run with:  python -m pytest tests  (or simply: python tests/test_cores.py)"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qsim import entangle_core as ec
from qsim import grover_core as gc
from qsim import wave_core as wc


# ----------------------------- Grover -----------------------------------
def test_uniform_state_normalised():
    amps = gc.uniform_state()
    assert math.isclose(float((amps ** 2).sum()), 1.0, abs_tol=1e-12)
    assert math.isclose(float(amps[0]), 0.25, abs_tol=1e-12)


def test_oracle_only_flips_target():
    amps = gc.uniform_state()
    out = gc.oracle_flip(amps, 5)
    assert out[5] < 0
    assert np.all(out[np.arange(16) != 5] > 0)
    assert math.isclose(float((out ** 2).sum()), 1.0, abs_tol=1e-12)


def test_diffusion_preserves_norm():
    amps = gc.oracle_flip(gc.uniform_state(), 3)
    out = gc.diffusion(amps)
    assert math.isclose(float((out ** 2).sum()), 1.0, abs_tol=1e-10)


def test_three_iterations_are_optimal_for_16():
    assert gc.optimal_iterations(16) == 3
    best = gc.success_probability(7, 3)
    assert best > 0.95, best
    assert gc.success_probability(7, 6) < best  # over-rotation hurts


def test_frames_shape_and_monotonic_growth():
    frames = gc.grover_frames(12)
    assert len(frames) == 1 + 2 * 3
    diff_probs = [f.target_prob for f in frames if f.stage == "diffusion"]
    assert diff_probs == sorted(diff_probs)


def test_every_target_reaches_high_probability():
    for box in range(1, 17):
        assert gc.success_probability(box) > 0.95


def test_bad_target_rejected():
    for bad in (0, 17, -3):
        try:
            gc.grover_frames(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_classical_search_finds_target():
    import random
    rng = random.Random(7)
    for box in (1, 9, 16):
        run = gc.classical_search(box, rng=rng)
        assert run.order[-1] == box
        assert 1 <= run.checks <= 16
    assert gc.classical_expected_checks(16) == 8.5


def test_measure_is_in_range_and_biased_to_target():
    import random
    amps = gc.grover_frames(4)[-1].amps
    rng = random.Random(0)
    hits = [gc.measure(amps, rng) for _ in range(300)]
    assert all(1 <= h <= 16 for h in hits)
    assert hits.count(4) / 300 > 0.9


# --------------------------- Wavefunction --------------------------------
def test_packet_is_normalised():
    x = wc.make_grid()
    psi = wc.gaussian_packet(x)
    dx = x[1] - x[0]
    assert math.isclose(float((np.abs(psi) ** 2).sum() * dx), 1.0, abs_tol=1e-9)


def test_free_evolution_moves_and_spreads():
    x = wc.make_grid()
    psi = wc.gaussian_packet(x, x0=-6.0, sigma=1.5, k0=2.0)
    x0, s0 = wc.expectation_x(psi, x), wc.uncertainty_x(psi, x)
    out = psi
    for _ in range(20):
        out = wc.evolve(out, x, 0.05)
    x1, s1 = wc.expectation_x(out, x), wc.uncertainty_x(out, x)
    assert x1 > x0          # momentum k0 > 0 pushes it right
    assert s1 > s0          # packets spread
    dx = x[1] - x[0]
    assert math.isclose(float((np.abs(out) ** 2).sum() * dx), 1.0, abs_tol=1e-9)


def test_barrier_evolution_stays_finite():
    x = wc.make_grid()
    v = wc.barrier_potential(x, height=6.0, width=1.0, centre=0.0)
    psi = wc.gaussian_packet(x, x0=-6.0, k0=2.0)
    for _ in range(40):
        psi = wc.evolve(psi, x, 0.02, v)
    assert np.all(np.isfinite(np.abs(psi)))


def test_two_slit_state_is_bimodal():
    x = wc.make_grid()
    psi = wc.two_slit_packet(x, sep=8.0, sigma=1.0)
    rho = wc.density(psi, x)
    left = rho[x < 0].max()
    right = rho[x > 0].max()
    middle = rho[np.abs(x) < 0.5].max()
    assert left > 5 * middle and right > 5 * middle


def test_sampling_follows_born_rule():
    x = wc.make_grid()
    psi = wc.gaussian_packet(x, x0=3.0, sigma=1.0, k0=0.0)
    s = wc.sample_positions(psi, x, 4000, np.random.default_rng(1))
    assert abs(float(s.mean()) - 3.0) < 0.2
    assert abs(float(s.std()) - 1.0) < 0.2


def test_collapse_narrows_the_state():
    x = wc.make_grid()
    psi = wc.gaussian_packet(x, sigma=2.0)
    before = wc.uncertainty_x(psi, x)
    outcome, after = wc.measure_once(psi, x, np.random.default_rng(3))
    assert wc.uncertainty_x(after, x) < before
    assert abs(wc.expectation_x(after, x) - outcome) < 0.5


def test_evolve_series_length():
    x = wc.make_grid()
    frames = wc.evolve_series(wc.gaussian_packet(x), x, 0.05, 10)
    assert len(frames) == 11 and frames[-1].t > frames[0].t


# --------------------------- Entanglement --------------------------------
def test_bell_states_normalised():
    for name, s in ec.BELL_STATES.items():
        assert math.isclose(float(np.vdot(s, s).real), 1.0, abs_tol=1e-12), name


def test_phi_plus_correlations():
    s = ec.PHI_PLUS
    assert math.isclose(ec.correlation(s, 0, 0), 1.0, abs_tol=1e-9)
    assert math.isclose(ec.correlation(s, 0, 90), -1.0, abs_tol=1e-9)
    assert math.isclose(ec.correlation(s, 0, 45), 0.0, abs_tol=1e-9)


def test_joint_probabilities_sum_to_one():
    for a in (0, 22.5, 45, 90):
        for b in (0, 30, 67.5):
            p = ec.joint_probabilities(ec.PHI_PLUS, a, b)
            assert math.isclose(sum(p.values()), 1.0, abs_tol=1e-9)
            assert all(v >= -1e-12 for v in p.values())


def test_chsh_violation():
    s = ec.chsh_S(ec.PHI_PLUS)
    assert math.isclose(s, 2 * math.sqrt(2), abs_tol=1e-6), s
    assert s > 2.0


def test_classical_correlation_bounded_by_2():
    delta = lambda a, b: ec.classical_correlation(a - b)
    s = abs(delta(0, 22.5) - delta(0, 67.5) + delta(45, 22.5) + delta(45, 67.5))
    assert s <= 2.0 + 1e-9, s


def test_monte_carlo_matches_theory():
    r = ec.simulate_pairs(ec.PHI_PLUS, 0, 30, 20000, np.random.default_rng(5))
    assert abs(r.measured_E - r.theory_E) < 0.05
    assert sum(r.coincidences.values()) == 20000


def test_spdc_cone_and_rate():
    xs, ys = ec.spdc_cone_points(200, 3.0, np.random.default_rng(2))
    assert xs.size == ys.size == 200
    assert 2.0 < float(np.mean(np.hypot(xs, ys))) < 4.0
    assert ec.pair_rate(10, 5) > 0 and ec.pair_rate(0, 5) == 0


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                passed += 1
                print(f"PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
