# SKHiPPR in one hour — didactic storyline

A plan for a 60-minute hands-on tutorial built around `skhippr_tutorial.ipynb`.
Read this before presenting; it explains *why* the notebook is ordered the way it is
and which sentence carries each segment.

---

## Audience

Researchers in nonlinear dynamics who already know equilibria, Floquet theory, harmonic
balance and pseudo-arclength continuation. **Nothing about the theory needs to be taught.**
What they do not know is how SKHiPPR expresses those ideas in code, and whether the toolbox
will bend to the problem they actually care about.

Consequently the tutorial never explains *what* a Floquet multiplier is. It explains where
it lives (`equation.eigenvalues`), who put it there (a `StabilityMethod` object), and what it
costs to swap the method that computed it (one line).

## The thesis

> SKHiPPR is built from a single idea and a single structural move.
>
> **The idea:** an equation is an object whose *unknowns are its attributes*. Solving means
> letting a solver mutate those attributes until the residual vanishes.
>
> **The move:** to gain a degree of freedom, append one equation and one unknown. Continuation,
> phase anchoring for autonomous systems, and every extension the user will ever write are the
> same move.

Everything in the hour is an instance of one of those two sentences. If the audience leaves
able to state them, the tutorial worked.

## The spine

The notebook escalates through five levels of problem while the *calling code stays fixed*:

| level | equation | unknown | solved by |
|---|---|---|---|
| algebraic | a circle | a point in the plane | `NewtonSolver` |
| equilibrium | an ODE | a state | the same `NewtonSolver` |
| equilibrium | a DAE | state + multiplier | the same `NewtonSolver` |
| periodic | an HBM residual | Fourier coefficients | the same `NewtonSolver` |
| branch | any of the above | + a parameter | the same continuator |

The payoff line, delivered in §6, is that the frequency-response continuation loop is
*character for character* the loop from §2 that traced a circle. Set this up early: when the
circle branch is computed in §2, say out loud "remember these eight lines".

---

## Timed plan

| min | § | On screen | The sentence that carries it |
|----:|---|---|---|
| 0–3 | 0 | Roadmap, the object map, imports | "Two sentences explain the whole toolbox." |
| 3–13 | 1 | `CircleEquation`, `NewtonSolver`, `EquationSystem` | "The unknowns are attributes, addressed by name — so solving *writes through* to your physics object." |
| 13–23 | 2 | Circle branch; anatomy of a `BranchPoint` | "A branch point **is** an equation system, with one anchor equation appended. Your original equation is still in there, at `bp.equations[0]`." |
| 23–31 | 3 | Saddle–node ODE, block-on-belt Hopf, pendulum DAE | "An ODE's residual *is* its right-hand side, so an ODE object is already an equilibrium problem." |
| 31–41 | 4 | `Fourier`, `HBMEquation`, one solved cycle | "`Fourier` is a configuration object, not a solver; `HBMEquation` is an ODE seen through it." |
| 41–48 | 5 | Four stability methods on one solution | "Stability is a constructor argument. You never call it." |
| 48–55 | 6 | Duffing FRC + stability-coloured plots | "This is the §2 loop. Nothing was added." |
| 55–60 | 7–8 | Autonomous, DAE, shooting; recap table | "Same move, every time: append an equation, append an unknown." |

Timings assume the presenter runs cells live and talks over them. Total notebook runtime is
about 10 seconds, so nothing is lost to waiting; the budget is entirely speech.

---

## Segment notes

### §1 — Equation vs. EquationSystem (the load-bearing distinction)

This is the segment most likely to be rushed, and the one that must not be. The distinction
is not bureaucratic:

- An **`AbstractEquation`** knows how to produce a residual vector from its own attributes.
  It has no opinion about what is unknown. `CircleEquation` is simultaneously a problem in
  `y` and a problem in `radius`.
- An **`EquationSystem`** is the pairing of equations with a *named list of unknowns*. It is
  the object that can be square, and therefore the object a Newton solver can consume.

Demonstrate the pairing by solving the *same* circle equation for `radius` (works: 1 residual,
1 unknown) and then for `y` (fails: 1 residual, 2 unknowns). The failure is the motivation for
`EquationSystem`, so let the `ValueError` print.

Then the write-through demo: after `solver.solve(system)`, `circle.y` has changed. Say
explicitly: *the solver did not return a solution vector, it edited your objects.* This is
what makes `branch_point.equations[0].x_time()` meaningful later.

**Do not skip the trap cell.** Changing a parameter on an equation (`system.equations[0].radius
= 3.0`) does *not* clear `system.solved`, because the assignment never passed through the
system. A second `solve()` then returns immediately with the stale answer. The fix —
`system.solved = False` — takes one line, but an audience that meets this at home instead of
here will lose an afternoon.

### §2 — Continuation, and the anatomy of a BranchPoint

Two modes, and the audience should be able to choose between them afterwards:

- **implicit**: the system is *underdetermined* by exactly one equation (circle alone: one
  residual, two unknowns). No continuation parameter is named; arclength closes the system.
- **explicit**: the system is already square, and a named `continuation_parameter` is promoted
  to an unknown, which re-opens it by one.

Then dissect a branch point live. The three facts to elicit:
`isinstance(bp, EquationSystem)` is `True`; `bp.equations` is the original list plus a
`ContinuationAnchor`; `bp.unknowns` has gained the continuation parameter. The anchor
contributes an identically-zero residual whose *derivative* is the previous tangent — which is
exactly how "orthogonal to the tangent" becomes an equation.

Emphasise `bp.equations[0]`: because every branch point carries its own duplicated equations,
a list of branch points is a list of fully-formed solutions. Nothing needs to be re-solved to
post-process. (Footnote for the curious: the *first* branch point shares the original objects;
later ones are copies.)

The spiral cell — incrementing `radius` from inside the loop — is 30 seconds of screen time
and is the moment people realise the loop body is theirs to control.

### §3 — ODEs and DAEs

`AbstractODE.residual_function()` returns `dynamics()`. That one line is the segment: a root of
the residual is an equilibrium, so an ODE object is already a solvable equation, and the
`StabilityEquilibrium` method is attached automatically in the constructor.

Order: (1) the saddle-node normal form written from scratch — twelve lines, the whole
extension contract visible at once; (2) its fold, traversed by arclength, which is the honest
argument for pseudo-arclength over naive parameter stepping; (3) the block-on-belt, where a
Hopf bifurcation at `vdr ≈ 0.52` is detected purely by watching `bp.stable` flip. The Hopf is
the bridge: beyond it the answer is a limit cycle, which §4 will compute.

The pendulum DAE closes the segment. Its hanging equilibrium comes out exact
(`λ = −mg/2ℓ = −4.905`), and the constraint is just another row of the residual.
**Caveat to state aloud:** the default equilibrium stability method eigen-decomposes `df/dx`
and ignores the singular mass matrix, so `dae.stable` is not meaningful here. Promise the
DAE-aware machinery in §7 and deliver it.

### §4 — Fourier and HBMEquation

Keep `Fourier` and `HBMEquation` conceptually apart:

- `Fourier` is a *configuration* object — `N_HBM`, `L_DFT`, `n_dof`, real or complex — that
  owns the transforms and the spectral differentiation operator. Show a round trip and a
  spectral derivative against the analytic one (errors at 1e-15) so it is credible, then move on.
- `HBMEquation` is an ODE *seen through* a `Fourier`: residual by alternating frequency/time,
  unknown `X`, closed-form Jacobian (the Hill matrix).

Mention the attribute delegation: `hbm.alpha` reaches into the ODE, `hbm.omega` is the ODE's
`omega`. It mirrors the system→equation write-through from §1, so it should feel familiar
rather than magical.

### §5 — Stability as a plug-in

The whole point is that the audience never calls a stability routine. They pass an object to a
constructor; after `solve()`, `equation.stable` and `equation.eigenvalues` are populated.

Drive it home by swapping four methods on one already-solved cycle — direct Koopman-Hill,
subharmonic Koopman-Hill, the classical sorted Hill eigenvalue problem, and a single-pass RK4
— and printing the multipliers side by side. They agree to five digits at a few milliseconds
each. This is also the natural 30-second window for the Koopman-Hill projection itself:
`Φ(t) = C exp(Ht) W`, no sorting, with a convergence guarantee (Bayer & Leine 2023, 2025).

The Hill-matrix block plot belongs here: the visible off-diagonal decay is the decay of the
Fourier coefficients of `df/dx`, which is the quantity the error bound is stated in.

### §6 — Continuation of periodic solutions

Say nothing new. Scroll back to §2, then show that the loop is identical. Let the plots do the
talking: 1-D, 2-D and 3-D `plot_continuation` from the same branch via different `plot_fun`s,
Floquet multipliers along the branch, and one animation.

If time is short, this is the segment to compress — but never to cut, because it is the
delivery of the promise made in §2.

### §7 — Versatility

Three short beats, each one sentence long:

- **autonomous** — `HBMSystem` appends a phase anchor and promotes `omega` to an unknown.
  Print `equations` and `unknowns` before and after continuation so the audience sees a 2×2
  system become 3×3. *The same move as the continuation anchor.*
- **DAE** — `HBMEquationDAE` + `KoopmanHillDAE`. The punchline is quantitative: the DAE
  formulation of the pendulum returns the ODE formulation's two Floquet multipliers to 15
  digits, plus zeros for the algebraically constrained directions.
- **shooting** — `ShootingSystem` is a different discretisation behind the same interface.
  Mention only; do not run long.

### §8 — Recap

One table: *to change X, swap Y*. Then the exercises, which are deliberately open-ended and
map onto real research tasks (a new ODE, a new stability method, a two-parameter sweep, a
subharmonic branch via `period_k`).

---

## Things deliberately left out

State these as "not covered" rather than letting them look like gaps:

- bifurcation *detection and localisation* (test functions, branch switching) — the loop body
  is the user's, and SKHiPPR does not localise for you;
- the error-bound machinery (`exponential_decay_parameters`, `error_bound_fundamental_matrix`)
  beyond a mention in §5;
- `PseudoSpectrumEquation`, the spatial pendulum, the Manlab benchmarks;
- performance tuning (`ScipyFsolveSolver`, `ScipyRootSolver`, `real_formulation=False`).

## If you are running short

Cut in this order: §7c (shooting), the 3-D continuation plot in §6, the block-on-belt in §3.
Never cut §1's trap cell or §2's branch-point dissection — they are what the rest stands on.
