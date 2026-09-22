"""Tab 3 - entangled photon pairs from a nonlinear (SPDC) crystal."""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from . import entangle_core as ec

FRAME_DELAY = 0.06


def _fresh_state() -> None:
    st.session_state.e_state_name = list(ec.BELL_STATES)[0]
    st.session_state.e_a = 0.0
    st.session_state.e_b = 22.5
    st.session_state.e_result = None
    st.session_state.e_log = []


def _ensure_state() -> None:
    if "e_state_name" not in st.session_state:
        _fresh_state()


# ------------------------------- visuals ---------------------------------
def _draw_crystal(progress: float, a_deg: float, b_deg: float,
                  outcome: tuple[str, str] | None = None):
    """Schematic of pump -> BBO crystal -> two entangled photons -> analyzers."""
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-3, 3)
    ax.axis("off")

    # pump beam
    pump_end = min(4.0, -0.5 + 5.0 * min(progress * 2, 1.0))
    ax.plot([-0.5, pump_end], [0, 0], color="#a855f7", lw=3.5, solid_capstyle="round")
    ax.text(0.2, 0.45, "pump 405 nm", color="#c4b5fd", fontsize=9)

    # crystal
    ax.add_patch(plt.Rectangle((4.0, -1.0), 0.9, 2.0, facecolor="#0ea5e9",
                               alpha=0.35, edgecolor="#38bdf8", lw=2))
    ax.text(4.45, 1.25, "BBO\nnonlinear crystal", color="#7dd3fc", fontsize=9,
            ha="center", va="bottom")

    # down-converted pair
    if progress > 0.5:
        frac = (progress - 0.5) / 0.5
        end = 4.9 + frac * 4.2
        ax.plot([4.9, end], [0, frac * 1.35], color="#22d3ee", lw=2.6)
        ax.plot([4.9, end], [0, -frac * 1.35], color="#f59e0b", lw=2.6)
        ax.text(6.0, 1.25, "signal 810 nm", color="#22d3ee", fontsize=9)
        ax.text(6.0, -1.5, "idler 810 nm", color="#f59e0b", fontsize=9)

    # analyzers
    for y, ang, col, name in ((1.35, a_deg, "#22d3ee", "Alice"),
                              (-1.35, b_deg, "#f59e0b", "Bob")):
        ax.add_patch(plt.Rectangle((9.1, y - 0.55), 0.22, 1.1, facecolor=col, alpha=0.55))
        t = np.deg2rad(ang)
        ax.plot([9.21 - 0.45 * np.cos(t), 9.21 + 0.45 * np.cos(t)],
                [y - 0.45 * np.sin(t), y + 0.45 * np.sin(t)],
                color="#e2e8f0", lw=2.2)
        ax.text(9.6, y, f"{name}\n{ang:.1f}°", color=col, fontsize=9, va="center")

    if outcome is not None:
        ax.text(5.0, -2.5,
                f"detection:  Alice {outcome[0]}   ·   Bob {outcome[1]}",
                color="#e2e8f0", fontsize=11, ha="center", weight="bold")

    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    return fig


def _draw_correlation_curve(state, a_deg, measured=None):
    grid = np.linspace(0, 180, 181)
    quantum = ec.quantum_correlation_curve(state, a_deg, grid)
    classical = ec.classical_correlation_curve(grid, a_deg)

    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.plot(grid, quantum, color="#22d3ee", lw=2.4, label="quantum  E = cos 2(a−b)")
    ax.plot(grid, classical, color="#94a3b8", lw=1.8, ls="--",
            label="best local hidden-variable model")
    ax.axhline(0, color="#475569", lw=0.8)
    if measured is not None:
        ax.plot([measured[0]], [measured[1]], "o", color="#f43f5e", ms=10,
                label=f"measured E = {measured[1]:+.3f}")
    ax.set_xlabel("Bob's analyzer angle b (degrees)")
    ax.set_ylabel("correlation E(a, b)")
    ax.set_ylim(-1.15, 1.15)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig


def _draw_counts(counts: dict[str, int]):
    labels = ["++", "+-", "-+", "--"]
    values = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.bar(["A+B+", "A+B−", "A−B+", "A−B−"], values,
           color=["#22c55e", "#ef4444", "#ef4444", "#22c55e"], alpha=0.85)
    ax.set_ylabel("coincidence counts")
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    return fig


# -------------------------------- tab ------------------------------------
def render_entanglement_tab() -> None:
    _ensure_state()

    st.subheader("🔗 Entangled photon pairs from a nonlinear crystal (SPDC)")
    st.caption(
        "A pump photon entering a BBO crystal occasionally splits into two lower-energy "
        "photons. Neither has a definite polarisation, yet their polarisations are perfectly "
        "correlated — measuring one instantly fixes the other, however far apart they are."
    )

    c1, c2, c3 = st.columns([2, 1, 1])
    state_name = c1.selectbox("Bell state produced", list(ec.BELL_STATES),
                              index=list(ec.BELL_STATES).index(st.session_state.e_state_name),
                              key="e_state_select")
    st.session_state.e_state_name = state_name
    state = ec.BELL_STATES[state_name]

    a_deg = c2.slider("Alice angle a (°)", 0.0, 180.0, float(st.session_state.e_a), 0.5,
                      key="e_a_slider")
    b_deg = c3.slider("Bob angle b (°)", 0.0, 180.0, float(st.session_state.e_b), 0.5,
                      key="e_b_slider")
    st.session_state.e_a, st.session_state.e_b = a_deg, b_deg

    p1, p2, p3, p4 = st.columns([1.1, 1, 1.25, 0.9])
    n_pairs = p1.select_slider("Pairs to detect", [50, 100, 500, 1000, 5000, 20000],
                               value=500, key="e_pairs_slider")
    fire_one = p2.button("💥 Fire one pair", key="e_btn_one", use_container_width=True)
    fire_many = p3.button("🔬 Run full experiment", key="e_btn_many", use_container_width=True)
    do_reset = p4.button("🔄 Reset", key="e_btn_reset", use_container_width=True)

    if do_reset:
        _fresh_state()
        st.rerun()

    st.divider()
    crystal_area = st.empty()
    status = st.empty()

    # ---- animate a single pair ----
    if fire_one:
        probs = ec.joint_probabilities(state, a_deg, b_deg)
        labels = list(probs)
        weights = np.array([probs[k] for k in labels])
        weights = np.clip(weights, 0, None)
        weights /= weights.sum()
        pick = labels[int(np.random.default_rng().choice(len(labels), p=weights))]
        for frac in np.linspace(0.05, 1.0, 16):
            fig = _draw_crystal(float(frac), a_deg, b_deg)
            crystal_area.pyplot(fig, clear_figure=True)
            plt.close(fig)
            time.sleep(FRAME_DELAY)
        fig = _draw_crystal(1.0, a_deg, b_deg, outcome=(pick[0], pick[1]))
        crystal_area.pyplot(fig, clear_figure=True)
        plt.close(fig)
        agree = pick in ("++", "--")
        status.success(
            f"Alice got **{pick[0]}**, Bob got **{pick[1]}** — "
            + ("they agree." if agree else "they disagree.")
            + f"  Theory says they agree {100 * (probs['++'] + probs['--']):.1f} % "
              "of the time at these angles."
        )
    else:
        fig = _draw_crystal(1.0, a_deg, b_deg)
        crystal_area.pyplot(fig, clear_figure=True)
        plt.close(fig)

    # ---- full run ----
    if fire_many:
        bar = st.progress(0.0, text="collecting coincidences …")
        for i in range(1, 6):
            bar.progress(i / 5, text=f"collecting coincidences … {i * n_pairs // 5}")
            time.sleep(0.08)
        result = ec.simulate_pairs(state, a_deg, b_deg, int(n_pairs))
        bar.empty()
        st.session_state.e_result = result
        st.session_state.e_log.append({
            "a (°)": a_deg, "b (°)": b_deg, "pairs": result.n_pairs,
            "E measured": round(result.measured_E, 4),
            "E theory": round(result.theory_E, 4),
        })
        status.success(
            f"{result.n_pairs} pairs detected. Measured correlation "
            f"E = {result.measured_E:+.3f} (theory {result.theory_E:+.3f})."
        )

    # ---- metrics ----
    theory_E = ec.correlation(state, a_deg, b_deg)
    chsh = ec.chsh_S(state)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Δ = a − b", f"{a_deg - b_deg:+.1f}°")
    m2.metric("Predicted E(a,b)", f"{theory_E:+.3f}")
    res = st.session_state.e_result
    m3.metric("Measured E", f"{res.measured_E:+.3f}" if res else "—",
              delta=f"{res.measured_E - res.theory_E:+.3f}" if res else None)
    m4.metric("CHSH S", f"{chsh:.3f}",
              delta="violates the classical bound 2" if chsh > 2 else "no violation")

    st.pyplot(_draw_correlation_curve(state, a_deg,
                                      measured=(b_deg, res.measured_E) if res else None),
              clear_figure=True)
    plt.close("all")

    if res:
        left, right = st.columns([1, 1.3])
        with left:
            st.markdown("##### Coincidence counts")
            st.pyplot(_draw_counts(res.coincidences), clear_figure=True)
            plt.close("all")
        with right:
            st.markdown("##### Outcome table")
            total = sum(res.coincidences.values())
            rows = [{"Alice": k[0], "Bob": k[1], "counts": v,
                     "fraction": round(v / total, 4),
                     "theory": round(ec.joint_probabilities(state, a_deg, b_deg)[k], 4)}
                    for k, v in res.coincidences.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if st.session_state.e_log:
        st.markdown("##### Experiment log")
        st.dataframe(pd.DataFrame(st.session_state.e_log), use_container_width=True,
                     hide_index=True)

    with st.expander("What makes this 'entangled' rather than just correlated"):
        st.markdown(
            """
**Inside the crystal.** Spontaneous parametric down-conversion: roughly one pump photon in
a billion splits into a signal and an idler photon that conserve energy (ω_p = ω_s + ω_i) and
momentum (phase matching). In type-II SPDC the pair emerges on two cones, and where the cones
intersect the photons are polarisation-entangled: |Φ⁺⟩ = (|HH⟩ + |VV⟩)/√2.

**No local state.** Neither photon has a polarisation of its own — each alone looks completely
random, 50/50 at *any* analyzer angle. Only the *pair* has a definite property.

**Correlation stronger than any classical model.** Quantum mechanics predicts
E(a,b) = cos 2(a−b). The best local hidden-variable theory can only manage the dashed
triangular curve. Feeding the angles 0°, 45°, 22.5°, 67.5° into CHSH gives
S = 2√2 ≈ 2.828, beating the classical limit of 2 — experimentally confirmed and now
loophole-free.

**Still no faster-than-light signalling.** Alice's own results are random regardless of Bob's
setting; the correlation only appears when the two records are compared over a classical channel.
            """
        )
