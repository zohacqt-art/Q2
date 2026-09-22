"""
Quantum vs Classical — interactive Streamlit simulation suite.

Run with:   streamlit run app.py
"""

from __future__ import annotations

import matplotlib
import streamlit as st

matplotlib.use("Agg")                 # headless backend, must precede pyplot imports
import matplotlib.pyplot as plt       # noqa: E402

from qsim.styles import inject_css     # noqa: E402
from qsim.ui_entangle import render_entanglement_tab   # noqa: E402
from qsim.ui_grover import render_grover_tab           # noqa: E402
from qsim.ui_wave import render_wave_tab               # noqa: E402

st.set_page_config(
    page_title="Quantum vs Classical — Interactive Simulation",
    page_icon="⚛️",
    layout="wide",
)

plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "text.color": "#e2e8f0",
    "axes.labelcolor": "#cbd5e1",
    "axes.edgecolor": "#475569",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "axes.titlecolor": "#e2e8f0",
    "font.size": 9,
})

inject_css()

st.title("⚛️ Quantum vs Classical — Interactive Simulation Suite")
st.caption(
    "Three animated simulations: Grover's search over 16 boxes, wavefunction "
    "superposition and collapse, and polarisation-entangled photon pairs from a "
    "nonlinear crystal."
)

tab1, tab2, tab3 = st.tabs([
    "🎯 Grover Algorithm",
    "🌊 Superposition & Collapse",
    "🔗 Entanglement (SPDC)",
])

with tab1:
    render_grover_tab()

with tab2:
    render_wave_tab()

with tab3:
    render_entanglement_tab()

st.divider()
st.caption(
    "Educational simulations computed from the exact state vectors — "
    "not data from laboratory hardware."
)
