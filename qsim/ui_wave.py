"""Tab 2 - superposition, evolution, measurement and wavefunction collapse."""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from . import wave_core as wc

FRAME_DELAY = 0.05


# ------------------------------- state -----------------------------------
def _build_initial(kind: str, sigma: float, k0: float) -> np.ndarray:
    x = st.session_state.w_x
    if kind == "Double slit (two packets)":
        return wc.two_slit_packet(x, sep=8.0, sigma=sigma, k0=k0)
    return wc.gaussian_packet(x, x0=-6.0, sigma=sigma, k0=k0)


def _fresh_state(kind: str = "Single Gaussian packet", sigma: float = 1.5,
                 k0: float = 2.0) -> None:
    st.session_state.w_x = wc.make_grid()
    st.session_state.w_kind = kind
    st.session_state.w_sigma = sigma
    st.session_state.w_k0 = k0
    psi = _build_initial(kind, sigma, k0)
    st.session_state.w_psi0 = psi.copy()
    st.session_state.w_psi = psi.copy()
    st.session_state.w_t = 0.0
    st.session_state.w_collapsed = False
    st.session_state.w_last_outcome = None
    st.session_state.w_samples = []


def _ensure_state() -> None:
    if "w_psi" not in st.session_state:
        _fresh_state()


# ------------------------------- plots -----------------------------------
def _plot_state(psi: np.ndarray, t: float, collapsed: bool,
                outcome: float | None, reference: np.ndarray | None = None):
    x = st.session_state.w_x
    rho = wc.density(psi, x)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1.2]})

    if reference is not None:
        ax1.plot(x, wc.density(reference, x), color="#64748b", lw=1.0, ls="--",
                 label="before measurement")
    ax1.plot(x, rho, color="#22d3ee", lw=2.0, label="|ψ(x)|²")
    ax1.fill_between(x, rho, color="#22d3ee", alpha=0.22)
    if outcome is not None and collapsed:
        ax1.axvline(outcome, color="#f43f5e", lw=1.6, ls=":",
                    label=f"outcome x = {outcome:.2f}")
    ax1.set_ylabel("probability density")
    ax1.set_title(("COLLAPSED state" if collapsed else "Superposition")
                  + f"   ·   t = {t:.2f}")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.15)

    ax2.plot(x, np.real(psi), color="#a78bfa", lw=1.2, label="Re ψ")
    ax2.plot(x, np.imag(psi), color="#fbbf24", lw=1.2, label="Im ψ")
    ax2.set_xlabel("position x")
    ax2.set_ylabel("amplitude")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.15)

    fig.tight_layout()
    return fig


def _plot_histogram(samples, psi):
    x = st.session_state.w_x
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.hist(samples, bins=40, density=True, color="#38bdf8", alpha=0.75,
            edgecolor="#0f172a", label=f"{len(samples)} measurements")
    ax.plot(x, wc.density(psi, x), color="#f43f5e", lw=2.0, label="|ψ(x)|² prediction")
    ax.set_xlabel("measured position")
    ax.set_ylabel("frequency density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig


# -------------------------------- tab ------------------------------------
def render_wave_tab() -> None:
    _ensure_state()
    x = st.session_state.w_x

    st.subheader("🌊 Superposition, evolution and collapse")
    st.caption(
        "Before measurement the particle has no position — only a wavefunction ψ(x). "
        "Measuring draws one outcome from |ψ(x)|² (the Born rule) and collapses ψ into a "
        "narrow spike around that outcome."
    )

    # ---- setup row ----
    s1, s2, s3 = st.columns([1.6, 1, 1])
    kind = s1.selectbox("Initial state",
                        ["Single Gaussian packet", "Double slit (two packets)"],
                        index=0 if st.session_state.w_kind.startswith("Single") else 1,
                        key="w_kind_select")
    sigma = s2.slider("Width σ", 0.6, 3.0, float(st.session_state.w_sigma), 0.1,
                      key="w_sigma_slider")
    k0 = s3.slider("Momentum k₀", -4.0, 4.0, float(st.session_state.w_k0), 0.5,
                   key="w_k0_slider")

    if (kind != st.session_state.w_kind or sigma != st.session_state.w_sigma
            or k0 != st.session_state.w_k0):
        _fresh_state(kind, sigma, k0)
        st.rerun()

    # ---- action row ----
    b1, b2, b3, b4 = st.columns(4)
    do_evolve = b1.button("▶️ Evolve wave", key="w_btn_evolve", use_container_width=True,
                          help="Animate free Schrödinger evolution.")
    do_measure = b2.button("📏 Measure position", key="w_btn_measure", use_container_width=True,
                           help="One projective measurement — the wavefunction collapses.")
    do_repeat = b3.button("🎲 50 repeated measurements", key="w_btn_repeat",
                          use_container_width=True,
                          help="Re-prepare the same state 50 times and record each outcome.")
    do_reset = b4.button("🔄 Reset", key="w_btn_reset", use_container_width=True)

    if do_reset:
        _fresh_state(kind, sigma, k0)
        st.rerun()

    st.divider()
    plot_area = st.empty()
    status = st.empty()

    # ---- evolve animation ----
    if do_evolve:
        if st.session_state.w_collapsed:
            status.info("Evolving the collapsed state — it spreads out again immediately.")
        psi = st.session_state.w_psi
        for step in range(40):
            psi = wc.evolve(psi, x, 0.05)
            st.session_state.w_t += 0.05
            if step % 2 == 0:
                fig = _plot_state(psi, st.session_state.w_t,
                                  st.session_state.w_collapsed,
                                  st.session_state.w_last_outcome)
                plot_area.pyplot(fig, clear_figure=True)
                plt.close(fig)
                time.sleep(FRAME_DELAY)
        st.session_state.w_psi = psi

    # ---- single measurement ----
    if do_measure:
        before = st.session_state.w_psi.copy()
        outcome, after = wc.measure_once(before, x)
        st.session_state.w_last_outcome = outcome
        st.session_state.w_collapsed = True
        # animate the collapse by blending the two densities
        for frac in np.linspace(0.0, 1.0, 12):
            blended = (1 - frac) * before + frac * after
            blended = wc.normalise(blended, x)
            fig = _plot_state(blended, st.session_state.w_t, frac > 0.5, outcome,
                              reference=before)
            plot_area.pyplot(fig, clear_figure=True)
            plt.close(fig)
            time.sleep(FRAME_DELAY)
        st.session_state.w_psi = after
        st.session_state.w_samples.append(float(outcome))
        status.success(
            f"Measured x = {outcome:.3f}. The spread collapsed from "
            f"σ = {wc.uncertainty_x(before, x):.2f} to "
            f"σ = {wc.uncertainty_x(after, x):.2f}."
        )

    # ---- 50 repeated measurements ----
    if do_repeat:
        psi_ref = st.session_state.w_psi if not st.session_state.w_collapsed \
            else st.session_state.w_psi0
        samples = wc.sample_positions(psi_ref, x, 50)
        st.session_state.w_samples = [float(s) for s in samples]
        status.success(
            "50 identically prepared copies measured. Each single outcome is random; "
            "together they rebuild |ψ(x)|²."
        )
        st.session_state.w_repeat_ref = psi_ref

    # ---- static draw when nothing animated ----
    if not (do_evolve or do_measure):
        fig = _plot_state(st.session_state.w_psi, st.session_state.w_t,
                          st.session_state.w_collapsed,
                          st.session_state.w_last_outcome)
        plot_area.pyplot(fig, clear_figure=True)
        plt.close(fig)

    # ---- metrics ----
    psi = st.session_state.w_psi
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("⟨x⟩", f"{wc.expectation_x(psi, x):.3f}")
    m2.metric("Δx (spread)", f"{wc.uncertainty_x(psi, x):.3f}")
    m3.metric("State", "collapsed" if st.session_state.w_collapsed else "superposition")
    m4.metric("Measurements recorded", len(st.session_state.w_samples))

    # ---- histogram of repeated measurements ----
    if len(st.session_state.w_samples) >= 2:
        st.markdown("##### Statistics of repeated measurements")
        ref = st.session_state.get("w_repeat_ref", st.session_state.w_psi0)
        fig = _plot_histogram(st.session_state.w_samples, ref)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        s = np.asarray(st.session_state.w_samples)
        st.dataframe(
            pd.DataFrame({"mean outcome": [s.mean()], "std of outcomes": [s.std()],
                          "min": [s.min()], "max": [s.max()], "count": [s.size]}).round(3),
            use_container_width=True, hide_index=True,
        )

    with st.expander("Why measurement is different from evolution"):
        st.markdown(
            """
**Between measurements** ψ obeys the Schrödinger equation: smooth, deterministic, reversible.
The packet slides at velocity ħk₀/m and spreads, because its momentum components move at
different speeds.

**At measurement** the rule changes. The outcome x is random with probability |ψ(x)|²dx, and
immediately afterwards the state is the eigenstate belonging to that outcome — a narrow spike.
That jump is not described by the Schrödinger equation, which is the measurement problem.

**One run tells you nothing; many runs tell you everything.** A single outcome looks arbitrary,
but 50 identically prepared copies reproduce |ψ(x)|² — the histogram converges to the red curve.

**Collapse then spreads again.** Evolve after collapsing: a very narrow packet has a very wide
momentum spread (Δx·Δp ≥ ħ/2), so it expands fast.
            """
        )
