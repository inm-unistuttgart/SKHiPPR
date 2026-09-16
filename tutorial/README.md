# SKHiPPR tutorial

A one-hour, hands-on introduction to SKHiPPR for researchers who already know nonlinear
dynamics.

| file | what it is |
|---|---|
| `skhippr_tutorial.ipynb` | the tutorial notebook — 35 code cells, runs in ~20 s |
| `STORYLINE.md` | the presenter's plan: timings, the narrative spine, what to emphasise |

## Running it

### On Google Colab (nothing to install)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/inm-unistuttgart/SKHiPPR/blob/main/tutorial/skhippr_tutorial.ipynb)

Click the badge (also at the top of the notebook itself), or open it manually at:

```
https://colab.research.google.com/github/inm-unistuttgart/SKHiPPR/blob/main/tutorial/skhippr_tutorial.ipynb
```

**This only works once `tutorial/` is merged into `main` and pushed to the public repository**
(`github.com/inm-unistuttgart/SKHiPPR`) — the notebook uses APIs (`PendulumDAE`,
`BlockOnBelt`, the current `plot_continuation`/`plot_hill_matrix_blocks` signatures, ...) that
are not yet on `main` as of this writing. Until you push, either run the notebook locally
(below), or replace `main` in the URL with whatever branch you are testing on and use that link
instead.

The first code cell (§ 0) detects Colab automatically and installs `skhippr` straight from
GitHub with `%pip install`. There is no separate setup step. It also asserts that the runtime
has **Python 3.12 or newer** (SKHiPPR uses `typing.override` internally); if Colab's default
runtime is older, the cell's error message gives a one-line fix using
[`condacolab`](https://github.com/conda-incubator/condacolab) — re-running the cell afterwards
proceeds normally.

### Locally

From the repository root:

```bash
pip install jupyterlab      # not part of the SKHiPPR dependencies
jupyter lab tutorial/skhippr_tutorial.ipynb
```

The same first cell detects that it is *not* on Colab and instead puts the repository root on
`sys.path`, so the notebook works whether you start Jupyter from the repository root or from
inside `tutorial/`, and it takes precedence over any editable install of `skhippr` that points
at a different checkout. The cell prints which `skhippr` it ended up using — check that line
before reporting an import problem.

Requirements beyond SKHiPPR's own (`numpy`, `scipy`, `matplotlib`): none. The notebook ships
without stored outputs on purpose; every figure is meant to appear as the participant runs it.

## What it covers

1. `AbstractEquation` and `EquationSystem`, and why the distinction matters
2. Pseudo-arclength continuation; a `BranchPoint` is an `EquationSystem` with one appended
   anchor equation, so `branch_point.equations[0]` is still your equation
3. `AbstractODE` / `AbstractDAE` — a root of the residual is an equilibrium, with stability
   attached automatically
4. `Fourier` and `HBMEquation` for periodic solutions
5. Stability methods as interchangeable plug-in objects (Koopman-Hill, classical Hill,
   single-pass Runge-Kutta)
6. Continuation of periodic solutions, using literally the § 2 loop
7. Autonomous systems (`HBMSystem`), DAEs (`HBMEquationDAE` + `KoopmanHillDAE`), shooting
8. Recap table and five exercises

## Systems used

Deliberately small, so nothing waits on a solver:

- a circle and an angle condition (algebraic, §§ 1–2)
- the saddle-node normal form $\dot x = \mu - x^2$, written from scratch (§ 3)
- a block on a moving belt with smoothed friction — Hopf bifurcation (§ 3)
- the constrained planar pendulum as a DAE (§§ 3, 7)
- a Duffing oscillator — frequency response with stability (§§ 4–6)
- a van der Pol oscillator — autonomous limit cycles (§ 7)

## Related material in this repository

`examples/` holds shorter, single-purpose scripts: `circle.py`, `duffing_minimal.py`,
`vanderpol_minimal.py`, `duffing_visualization.py`, `pendulum_dae.py`. The tutorial is the
guided path through the ideas those scripts each demonstrate once.

Full API documentation: <https://inm-unistuttgart.github.io/SKHiPPR/>
