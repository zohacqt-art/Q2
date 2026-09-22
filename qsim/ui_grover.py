"""Tab 1 - Grover search vs classical search over 16 boxes."""

from __future__ import annotations

import random
import time

import numpy as np
import pandas as pd
import streamlit as st

from . import grover_core as gc
from .styles import classical_grid_html, quantum_grid_html

N = gc.N_BOXES
STEP_DELAY = 0.16          # seconds per classical animation frame (per box opened)
QUANTUM_STEP_DELAY = 0.04  # seconds per quantum animation frame (kept short so the
                           # quantum run visibly finishes faster than the classical one,
                           # matching its real advantage of far fewer oracle queries)


# ------------------------------- state -----------------------------------
def _fresh_state() -> None:
    st.session_state.g_hidden = random.randint(1, N)
    st.session_state.g_amps = gc.uniform_state().tolist()
    st.session_state.g_opened = []
    st.session_state.g_found = False
    st.session_state.g_classical_checks = None
    st.session_state.g_classical_time = None
    st.session_state.g_quantum_steps = None
    st.session_state.g_quantum_time = None
    st.session_state.g_measured = None
    st.session_state.g_prob_history = []
    st.session_state.g_log = []


def _ensure_state() -> None:
    if "g_hidden" not in st.session_state:
        _fresh_state()
    st.session_state.setdefault("g_reveal", False)
    st.session_state.setdefault("g_iterations", gc.optimal_iterations(N))
    st.session_state.setdefault("g_run", None)      # pending animation


# ----------------------------- animations --------------------------------
def _animate_classical(placeholder, status) -> None:
    hidden = st.session_state.g_hidden
    run = gc.classical_search(hidden)
    t0 = time.perf_counter()
    for i, box in enumerate(run.order, start=1):
        is_last = box == hidden
        placeholder.markdown(
            classical_grid_html(run.order[: i - 1], hidden, current=box,
                                reveal=st.session_state.g_reveal, found=False),
            unsafe_allow_html=True,
        )
        status.info(f"Opening box {box} … ({i} of up to {N} checks)")
        time.sleep(STEP_DELAY)
        if is_last:
            break
    elapsed = time.perf_counter() - t0

    st.session_state.g_opened = run.order
    st.session_state.g_found = True
    st.session_state.g_classical_checks = run.checks
    st.session_state.g_classical_time = elapsed
    st.session_state.g_log.append(
        {"Method": "Classical", "Queries": run.checks, "Seconds": round(elapsed, 3),
         "Result": f"box {hidden}"}
    )
    placeholder.markdown(
        classical_grid_html(run.order, hidden, reveal=True, found=True),
        unsafe_allow_html=True,
    )
    status.success(f"Classical search opened {run.checks} box(es) to find box {hidden}.")


def _animate_quantum(placeholder, status) -> None:
    hidden = st.session_state.g_hidden
    iterations = int(st.session_state.g_iterations)
    frames = gc.grover_frames(hidden, iterations)
    t0 = time.perf_counter()

    history = []
    for frame in frames:
        flash = frame.stage == "oracle"
        placeholder.markdown(
            quantum_grid_html(frame.amps, hidden, reveal=st.session_state.g_reveal,
                              negative_flash=flash),
            unsafe_allow_html=True,
        )
        if frame.stage == "oracle":
            status.warning(f"🔄 {frame.label} — amplitude of box {hidden} becomes negative")
        elif frame.stage == "diffusion":
            status.info(f"📈 {frame.label} — P(target) = {frame.target_prob * 100:.1f}%")
        else:
            status.info(f"🌐 {frame.label} — every box at {100 / N:.1f}%")
        history.append({"Step": frame.label, "P(target)": frame.target_prob})
        time.sleep(QUANTUM_STEP_DELAY)

    final = frames[-1].amps
    measured = gc.measure(final)
    elapsed = time.perf_counter() - t0
    queries = iterations   # one oracle call per Grover iteration

    st.session_state.g_amps = np.asarray(final).tolist()
    st.session_state.g_measured = measured
    st.session_state.g_quantum_steps = queries
    st.session_state.g_quantum_time = elapsed
    st.session_state.g_prob_history = history
    st.session_state.g_log.append(
        {"Method": "Quantum (Grover)", "Queries": queries, "Seconds": round(elapsed, 3),
         "Result": f"box {measured}" + ("" if measured == hidden else " ✗")}
    )

    placeholder.markdown(
        quantum_grid_html(final, hidden, reveal=True, measured=measured),
        unsafe_allow_html=True,
    )
    if measured == hidden:
        status.success(
            f"Measured box {measured} — correct, after only {queries} oracle queries "
            f"(success probability {frames[-1].target_prob * 100:.1f}%)."
        )
    else:
        status.error(
            f"Measured box {measured}, the object was in {hidden}. Grover is probabilistic — "
            "run it again and it almost always lands on the target."
        )


# -------------------------------- tab ------------------------------------
def render_grover_tab() -> None:
    _ensure_state()
    hidden = st.session_state.g_hidden

    st.subheader("🎯 Finding one object hidden in 16 boxes")
    st.caption(
        "Classical search must open boxes one by one — about N/2 = 8 on average, 16 worst case. "
        "Grover's algorithm uses an oracle phase flip plus inversion about the mean to amplify "
        "the hidden box's amplitude, and finds it in ~√N ≈ 3 queries."
    )

    # ---- control row: toggle + 4 buttons ----
    c1, c2, c3, c4, c5 = st.columns([1.5, 1, 1, 1.15, 0.9])
    with c1:
        st.session_state.g_reveal = st.toggle(
            "👁️ Show hidden object", value=st.session_state.g_reveal, key="g_reveal_toggle",
            help="Peek at where the object is. The algorithms never use this.",
        )
    run_classical = c2.button("▶️ Run Classical", key="g_btn_classical", use_container_width=True)
    run_quantum = c3.button("⚛️ Run Quantum", key="g_btn_quantum", use_container_width=True)
    run_both = c4.button("⚖️ Compare Together", key="g_btn_compare", use_container_width=True)
    do_reset = c5.button("🔄 Reset", key="g_btn_reset", use_container_width=True)

    if do_reset:
        _fresh_state()
        st.rerun()

    if st.session_state.g_reveal:
        st.info(f"🎯 The object is hidden in **box {hidden}**.")

    st.divider()

    # ---- the two grids ----
    left, right = st.columns(2)
    with left:
        st.markdown("#### 🧮 Classical grid")
        classical_box = st.empty()
        classical_status = st.empty()
    with right:
        st.markdown("#### ⚛️ Quantum grid (amplitudes)")
        quantum_box = st.empty()
        quantum_status = st.empty()

    # draw current (static) state first
    classical_box.markdown(
        classical_grid_html(st.session_state.g_opened, hidden,
                            reveal=st.session_state.g_reveal,
                            found=st.session_state.g_found),
        unsafe_allow_html=True,
    )
    quantum_box.markdown(
        quantum_grid_html(st.session_state.g_amps, hidden,
                          reveal=st.session_state.g_reveal,
                          measured=st.session_state.g_measured),
        unsafe_allow_html=True,
    )

    with left:
        st.markdown(
            '<div class="legend">🔵 unopened &nbsp;•&nbsp; ⚫ opened &amp; empty '
            '&nbsp;•&nbsp; 🟠 opening now &nbsp;•&nbsp; 🟢 found</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<div class="legend">Brighter = larger |amplitude|² &nbsp;•&nbsp; '
            '🟣 negative amplitude after the oracle flip &nbsp;•&nbsp; 🟢 measured</div>',
            unsafe_allow_html=True,
        )
        st.slider("Grover iterations", 1, 8, key="g_iterations",
                  help="π/4·√16 ≈ 3 is optimal. More iterations over-rotate and the "
                       "probability falls again.")

    # ---- run the requested animations ----
    if run_classical or run_both:
        st.session_state.g_opened, st.session_state.g_found = [], False
        _animate_classical(classical_box, classical_status)
    if run_quantum or run_both:
        st.session_state.g_measured = None
        _animate_quantum(quantum_box, quantum_status)

    # ---- comparison panel ----
    st.divider()
    st.markdown("#### 📊 Comparison — queries and time")

    cc = st.session_state.g_classical_checks
    qq = st.session_state.g_quantum_steps
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Classical queries", cc if cc is not None else "—",
              help="Boxes physically opened.")
    m2.metric("Quantum queries", qq if qq is not None else "—",
              help="Oracle calls = Grover iterations.")
    if cc and qq:
        m3.metric("Speed-up", f"{cc / qq:.2f}×", delta=f"{cc - qq} fewer queries")
    else:
        m3.metric("Speed-up", "—")
    m4.metric(
        "Theory",
        f"{gc.classical_expected_checks(N):.1f} vs {gc.optimal_iterations(N)}",
        help="Average classical N/2 = 8.5 vs Grover ⌊π/4·√N⌋ = 3.",
    )

    t1, t2 = st.columns(2)
    t1.metric("Classical wall-clock",
              f"{st.session_state.g_classical_time:.3f} s"
              if st.session_state.g_classical_time else "—")
    t2.metric("Quantum wall-clock",
              f"{st.session_state.g_quantum_time:.3f} s"
              if st.session_state.g_quantum_time else "—")
    st.caption(
        "Wall-clock here is animation time, not hardware speed — the meaningful "
        "comparison is the *number of oracle queries*."
    )

    if st.session_state.g_prob_history:
        st.markdown("##### Amplitude of the target through the algorithm")
        hist = pd.DataFrame(st.session_state.g_prob_history)
        st.line_chart(hist.set_index("Step"), height=240)

    if st.session_state.g_log:
        st.markdown("##### Run log")
        st.dataframe(pd.DataFrame(st.session_state.g_log), use_container_width=True,
                     hide_index=True)

    with st.expander("How the oracle and the diffuser actually work"):
        st.markdown(
            """
**1. Superposition.** Four Hadamards put the 4-qubit register into all 16 states at once,
each with amplitude 1/4 (probability 6.25 %).

**2. Oracle phase flip.** The oracle recognises the marked box and multiplies *only* that
amplitude by −1. Probabilities are unchanged — |−1/4|² is still 6.25 % — so measuring here
gives nothing. The information is hidden in the *phase*.

**3. Diffusion (inversion about the mean).** Reflecting every amplitude about the average
turns the lone negative amplitude into a large positive one while shrinking the other 15.

**4. Repeat ≈ √N times.** Each round rotates the state by ~2·arcsin(1/√N) toward the target.
After 3 rounds on 16 items the target carries ≈ 96 % probability; a 4th round starts
rotating *past* it, which is why the slider can make results worse.
            """
        )
