# ⚛️ Quantum vs Classical — Interactive Streamlit Simulation Suite

Three animated, interactive simulations in one app.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 (Streamlit opens it automatically).

Shortcuts: `./run.sh` on macOS/Linux, `run_windows.bat` on Windows.

Verify the physics at any time:

```bash
python tests/test_cores.py        # 23 checks, no Streamlit needed
```

## What each tab does

### Tab 1 — 🎯 Grover Algorithm
Two 4×4 grids of boxes 1–16 side by side, one classical and one quantum, with a hidden
object placed at random.

* **👁️ Show hidden object** — toggle that reveals the target (the algorithms never use it)
* **▶️ Run Classical** — opens boxes one at a time, animated, until the object turns up
* **⚛️ Run Quantum** — animates each Grover step: oracle phase flip (marked box turns purple
  and its amplitude goes negative), then diffusion (inversion about the mean), which grows
  the target's probability from 6.25 % → 47 % → 91 % → 96 %
* **⚖️ Compare Together** — runs both and fills the comparison panel
* **🔄 Reset** — new hidden object, cleared metrics

Comparison panel shows queries used, the speed-up ratio, wall-clock time, theory
(8.5 classical vs 3 quantum), a line chart of the target's probability through the
algorithm, and a run log. The iteration slider goes to 8 so you can watch over-rotation
*reduce* the success probability past 3 iterations.

### Tab 2 — 🌊 Superposition & Collapse
The wavefunction before measurement and after collapse.

* Choose a single Gaussian packet or a double-slit (two-packet) superposition; set width σ
  and momentum k₀
* **▶️ Evolve wave** — animated split-operator Schrödinger evolution; the packet travels and
  spreads. Bottom panel shows Re ψ and Im ψ oscillating
* **📏 Measure position** — one Born-rule outcome, with the collapse animated from the spread
  state into a narrow spike; the pre-measurement density stays as a dashed ghost
* **🎲 50 repeated measurements** — 50 identically prepared copies, plotted as a histogram
  against the |ψ(x)|² prediction, plus outcome statistics
* **🔄 Reset** — back to the freshly prepared state

Live metrics: ⟨x⟩, spread Δx, whether the state is collapsed, number of measurements.

### Tab 3 — 🔗 Entanglement (SPDC)
Animated nonlinear-crystal experiment: a pump photon enters a BBO crystal and down-converts
into a signal/idler pair sent to Alice's and Bob's polarisation analyzers.

* Pick any of the four Bell states; set both analyzer angles
* **💥 Fire one pair** — animates one photon pair through the crystal to the detectors and
  shows the ± outcomes
* **🔬 Run full experiment** — 50 to 20 000 pairs, with coincidence counts, an outcome table
  comparing measured fractions to theory, and an experiment log
* **🔄 Reset**

The correlation plot draws quantum E = cos 2(a−b) against the best local hidden-variable
model, with your measured point marked. The CHSH metric reads S = 2.828 > 2 — the Bell
violation.

## File map

| File | Contains |
|---|---|
| `app.py` | page config, theme, tab layout |
| `qsim/grover_core.py` | oracle, diffuser, frames, measurement, classical search |
| `qsim/wave_core.py` | packets, split-operator evolution, Born sampling, collapse |
| `qsim/entangle_core.py` | Bell states, joint probabilities, correlations, CHSH, SPDC |
| `qsim/ui_grover.py` | Tab 1 |
| `qsim/ui_wave.py` | Tab 2 |
| `qsim/ui_entangle.py` | Tab 3 |
| `qsim/styles.py` | CSS and the HTML box-grid renderers |
| `tests/test_cores.py` | 23 physics tests |

The `*_core.py` modules import no Streamlit, so they can be tested, reused, or imported into
a notebook on their own.

## What was fixed from the previous version

* Physics and UI were tangled in the same modules — they are now separated, and the physics
  is covered by tests
* Grover amplitudes were renormalised ad hoc; the oracle and diffuser now preserve the norm
  exactly (verified to 1e-10)
* The classical and quantum grids did not animate step by step — both now stream frames
* Missing `Compare Together` path and missing query/time metrics
* Wavefunction tab had no collapse animation, no repeated-measurement statistics, and no
  reset; evolution now uses the split-operator method with FFT instead of a naive step that
  leaked norm
* Entanglement tab had no crystal animation, no Bell-state choice, and no CHSH calculation
* `matplotlib` figures were never closed, leaking memory across reruns — every figure is now
  closed after rendering, and the `Agg` backend is set before `pyplot` is imported

## Note

Everything is simulated from exact state vectors for teaching purposes; no laboratory
hardware or quantum backend is involved.

## Deploying to Streamlit Community Cloud

Your repo must contain **only** these files — delete the old flat `grover.py`,
`wavefunction.py`, and `entanglement.py` from the repo root, or Streamlit will keep
importing them:

```
app.py
requirements.txt
qsim/__init__.py
qsim/grover_core.py
qsim/wave_core.py
qsim/entangle_core.py
qsim/ui_grover.py
qsim/ui_wave.py
qsim/ui_entangle.py
qsim/styles.py
```

Set the main file to `app.py` and reboot the app from *Manage app* so the container
reinstalls dependencies.

### About `StreamlitDuplicateElementId`

Streamlit derives each widget's ID from its type, label, and parameters — not from which
tab it sits in. Three buttons all labelled `🔄 Reset` therefore hash to the same ID and the
app crashes on the second one. Every widget here now passes an explicit unique `key=`
(`g_btn_reset`, `w_btn_reset`, `e_btn_reset`, and so on), which is what the error message
is asking for. If you add widgets later, give each one a key.
