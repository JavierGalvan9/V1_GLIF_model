# GLIF3 membrane integration

Why the V1 model's membrane step was replaced by an exact propagator
(`v1_model_utils/glif_propagators.py`), and what it bought.

**Start with `report.pdf`** --- the derivation, the equations and the results,
in the notation of the thesis. `integration_accuracy.ipynb` is its companion:
same numbers, same figures, with the code that produces them.

## Layout

| file | what it is |
|---|---|
| `report.pdf` / `report.tex` | the report; `classicthesis`, `biblatex`, compiled with `latexmk` |
| `integration_accuracy.ipynb` | the study with outputs saved --- readable without running |
| `build_notebook.py` | regenerates the notebook; edit this, not the `.ipynb` |
| `_study.py` | the analysis: one function per result, no plotting, no printing |
| `_figures.py` | one definition per figure, shared by the notebook and the report |
| `_simulation.py` | drivers, cell-type loading, and the float64 ground truths |
| `export_figures.py` | renders `figures/*.pdf`, `numbers.tex` and `benchtable.tex` for LaTeX |
| `verify_gradients.py` | finite-difference check of the backward pass, both backends |
| `benchmark_kernels.py` | GPU timings of the previous CUDA kernels against the current ones |

The report quotes no number by hand: `numbers.tex` and `benchtable.tex` are
generated from `_study`, which the notebook also calls. Neither document can
drift from the other.

## Notation

The report follows the thesis (Galván Fraile 2025, §1.2--1.3); the code keeps
its own names.

| symbol | code | meaning |
|---|---|---|
| $V_j[t]$ | `v` | membrane potential, in units of $\Delta V = V_{th}-E_L$ |
| $C^{\rm syn}_{j,r}[t]$ | `psc_rise` | rise variable of receptor basis $r$ |
| $I^{\rm syn}_{j,r}[t]$ | `psc` | postsynaptic current of receptor basis $r$ |
| $I^{s}_j[t]$ | `asc` | after-spike current, $s\in\{1,2\}$ |
| $S_j[t]$ | `prev_z` | spike indicator |
| $\alpha,\ \alpha_r,\ \beta_s$ | `decay`, `syn_decay`, `asc_decay` | decay factors |
| $\mathcal{A}_{j,r},\ \mathcal{B}_{j,r},\ \mathcal{D}^{s}_j$ | `psc_factor`, `psc_rise_factor`, `asc_factor` | how each current drives $V_j$ across the step |
| $\mathcal{R}_j,\ \mathcal{S}_j$ | `reset_coeff`, `asc_spike_factor` | a spike's reset, and the ASC it injects |

$\mathcal{R}_j$ and $\mathcal{S}_j$ both multiply the same spike and could be
folded into one constant. They are not, because `detach_reset` and
`detach_asc_reset` gate them separately in the backward pass --- see §3.2 and
§3.6 of the report.

## Reproducing

The analysis is pure NumPy/SciPy --- no TensorFlow, no GPU:

```bash
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace integration_accuracy.ipynb

python export_figures.py          # figures/*.pdf, numbers.tex, benchtable.tex
latexmk -pdf report.tex
```

Run both with the same interpreter. They agree to machine precision either
way, but a matrix exponential's last digits depend on the LAPACK build, so
mixing environments makes the two documents differ in digits that carry no
meaning.

The kernel benchmark needs a GPU. It compiles the current operator sources and
those of `GLIF_LEGACY_REF` (default `HEAD~1`) side by side, renaming the legacy
ops so both libraries can share one TensorFlow registry, then times each state
transition with 64 steps chained inside a `tf.function`:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_kernels.py
```

Pin `CUDA_VISIBLE_DEVICES` to GPUs of one compute capability --- the operator
cache builds one artifact per architecture. Set `GLIF_LEGACY_REF=HEAD` while
the propagator change is still uncommitted, and `GLIF_BENCH_ARCH` to override
the detected architecture.

The gradient check also needs TensorFlow (and a GPU for the `cuda` backend). It
writes `gradient_check.json`, which the notebook and the report read:

```bash
CUDA_VISIBLE_DEVICES=0 python verify_gradients.py
```

Run the whole suite on GPUs of one compute capability, and prefer a single GPU:
the test suite takes about a minute on one and stalls on three.
