# SKHiPPR tutorial

A one-hour, hands-on introduction to SKHiPPR for researchers who already know nonlinear
dynamics.

| file | what it is |
|---|---|
| `skhippr_tutorial.ipynb` | the tutorial notebook — 35 code cells, runs in ~20 s |
| `STORYLINE.md` | the presenter's plan: timings, the narrative spine, what to emphasise |

## Running it

From the repository root:

```bash
pip install jupyterlab      # not part of the SKHiPPR dependencies
jupyter lab tutorial/skhippr_tutorial.ipynb
```

The first cell puts the repository root on `sys.path`, so the notebook works whether you start
Jupyter from the repository root or from inside `tutorial/`, and it takes precedence over any
editable install of `skhippr` that points at a different checkout. The cell prints which
`skhippr` it ended up using — check that line before reporting an import problem.

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
