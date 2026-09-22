"""Shared CSS plus the HTML box-grid renderers used by the Grover tab."""

from __future__ import annotations

import html
from typing import Iterable, Sequence

import numpy as np
import streamlit as st

CSS = """
<style>
.box-grid { display:grid; grid-template-columns:repeat(4, 1fr); gap:8px; max-width:420px; }
.qbox { min-height:66px; border-radius:12px; display:flex; flex-direction:column;
        align-items:center; justify-content:center; color:#fff; font-weight:700;
        font-size:17px; border:2px solid rgba(255,255,255,.18);
        transition:background .25s ease, transform .18s ease; line-height:1.15; }
.qbox .pct { font-size:11px; font-weight:600; opacity:.9; margin-top:3px; }
.qbox.hit { transform:scale(1.06); border-color:#facc15; }
.legend { color:#94a3b8; font-size:.82rem; margin-top:8px; }
.panel { padding:14px 16px; border:1px solid #334155; border-radius:14px;
         background:rgba(15,23,42,.55); }
.badge { display:inline-block; padding:3px 10px; border-radius:999px;
         font-size:.75rem; font-weight:700; letter-spacing:.3px; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def classical_grid_html(opened: Iterable[int], hidden: int, current: int | None = None,
                        reveal: bool = False, found: bool = False, n: int = 16) -> str:
    opened_set = set(opened)
    parts = ['<div class="box-grid">']
    for box in range(1, n + 1):
        cls, label, sub = "qbox", str(box), ""
        if found and box == hidden:
            bg, label, sub = "#16a34a", f"{box}", "FOUND"
            cls += " hit"
        elif box == current:
            bg, sub = "#f59e0b", "opening"
            cls += " hit"
        elif box in opened_set:
            bg, label, sub = "#334155", f"{box}", "empty"
        elif reveal and box == hidden:
            bg, sub = "#dc2626", "hidden"
        else:
            bg = "#1d4ed8"
        parts.append(
            f'<div class="{cls}" style="background:{bg}">{html.escape(label)}'
            f'<span class="pct">{html.escape(sub)}</span></div>'
        )
    parts.append("</div>")
    return "".join(parts)


def quantum_grid_html(amps: Sequence[float], hidden: int, reveal: bool = False,
                      measured: int | None = None, negative_flash: bool = False) -> str:
    amps_arr = np.asarray(amps, dtype=float)
    probs = amps_arr ** 2
    total = probs.sum()
    if total > 0:
        probs = probs / total
    peak = max(float(probs.max()), 1e-12)

    parts = ['<div class="box-grid">']
    for i, (a, p) in enumerate(zip(amps_arr, probs), start=1):
        share = float(p / peak)
        if measured == i:
            bg, cls = "#16a34a", "qbox hit"
        elif negative_flash and a < 0:
            bg, cls = "#7c3aed", "qbox hit"          # oracle phase-flipped
        else:
            r = int(24 + 200 * share)
            g = int(48 + 80 * share)
            b = int(120 + 110 * (1 - share))
            bg, cls = f"rgb({r},{g},{b})", "qbox"
        mark = " 🎯" if (reveal and i == hidden) else ""
        sign = "−" if a < 0 else ""
        parts.append(
            f'<div class="{cls}" style="background:{bg}">{i}{mark}'
            f'<span class="pct">{sign}{p * 100:.1f}%</span></div>'
        )
    parts.append("</div>")
    return "".join(parts)
