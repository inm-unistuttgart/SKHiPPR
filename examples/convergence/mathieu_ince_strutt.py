"""This example file creates some mathieu-equation-related figures for the Bayer2025 paper:

* Ince-Strutt diagram
* Special case on the stability boundary with coinciding eigenvalues

"""

from collections.abc import Callable, Iterable
from typing import override
import numpy as np
from copy import copy

from scipy.integrate import solve_ivp

from skhippr.Fourier import Fourier
from skhippr.equations.AbstractEquation import AbstractEquation
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.cycles.hbm import HBMEquation
from skhippr.odes.ltp import MathieuODE, SmoothedMeissner
from skhippr.solvers.newton import NewtonSolver
from skhippr.solvers.continuation import pseudo_arclength_continuator
from skhippr.stability.KoopmanHillProjection import (
    KoopmanHillProjection,
    KoopmanHillSubharmonic,
)
import matplotlib.pyplot as plt

from skhippr.visualization.data_export import save_pdf, save_png


class FixedHarmonic(AbstractEquation):
    """This equation fixes the cosine component of the ``harmo``-th harmonic of the ``dof``-th degree of  freedom to ``value``.
    ``fourier`` is used during initialization to know the correct number of degrees of freedom and the formulation.
    """

    def __init__(self, X, fourier, harmo=1, dof=0, value=1.0):
        super().__init__(None)
        self.X = X
        self.value = value
        self.anchor = np.zeros(fourier.n_dof * (2 * fourier.N_HBM + 1))
        if fourier.real_formulation:
            self.anchor[harmo * fourier.n_dof + dof] = 1
        else:
            self.anchor[
                [
                    (self.fourier.N_HBM + harmo) * self.fourier.n_dof + dof,
                    (self.fourier.N_HBM - harmo) * self.fourier.n_dof + dof,
                ]
            ] = 0.5

    def residual_function(self):
        return np.atleast_1d(np.inner(self.anchor, self.X) - self.value)

    def closed_form_derivative(self, variable):
        match variable:
            case "X":
                return np.atleast_2d(self.anchor)
            case "value":
                return np.atleast_2d(-1)
            case _:
                return np.atleast_2d(0)


class InceStruttSystem(EquationSystem):
    """
    This is a subclass of :py:class:`~skhippr.problems.HBM.HBMProblem` for finding periodic solutions as stability boundaries of an Ince-Strutt-Type chart.

    An additional equation is appended to the HBMEquation used to fix a selected harmonic and degree of freedom to 1,
    identifying one (nontrivial) periodic solution out of the dense set of periodic solutions at a stability boundary.

    The following attributes are added/modified compared to the parent :py:class:`~skhippr.problems.HBM.HBMProblem`:

    """

    def __init__(
        self,
        hbm: HBMEquation,
        harmo_anchor=1,
        dof_anchor=0,
        varying_parameter="b",
    ):
        anchor = FixedHarmonic(hbm.X, hbm.fourier, harmo_anchor, dof_anchor)

        super().__init__(
            equations=[hbm, anchor],
            unknowns=["X", varying_parameter],
            equation_determining_stability=hbm,
        )


class InceStruttShootingEquation(AbstractEquation):
    def __init__(self, ode: MathieuODE, period_doubling=False):
        self.ode = ode
        if period_doubling:
            self.traceval = -2
        else:
            self.traceval = 2
        self.T = 2 * np.pi / self.ode.omega
        super().__init__(stability_method=None)

    def __setattr__(self, name, value):
        if name in ("a", "b"):
            setattr(self.ode, name, value)
        return super().__setattr__(name, value)

    def determine_monodromy_matrix(self):
        Phi = np.eye(2)
        for k in range(2):
            sol = solve_ivp(
                self.ode.dynamics,
                (0, self.T),
                Phi[:, k],
                t_eval=[self.T],
                atol=1e-9,
                rtol=1e-9,
            )
            Phi[:, k] = sol.y[:, -1]
        return Phi

    def residual_function(self):
        # Integrate over one period and compute the trace of the monodromy matrix
        Phi = self.determine_monodromy_matrix()
        trace = np.trace(Phi)
        return np.atleast_1d(trace - self.traceval)

    def closed_form_derivative(self, variable):
        raise NotImplementedError(
            "Derivative of shooting residual w.r.t. parameters not implemented. Consider using finite differences or automatic differentiation."
        )


def plot_ince_strutt(N_HBM=10, subh=True, a_max=3, b_max=5, pixelsize=0.2):
    a_min = -0.5
    num_a = int((a_max - a_min) / pixelsize) + 1
    num_b = int(b_max / pixelsize) + 1
    a_grid = np.linspace(-0.5, a_max, num=num_a)
    b_grid = np.linspace(0, b_max, num=num_b)

    fourier = Fourier(N_HBM=N_HBM, L_DFT=60, n_dof=2, real_formulation=True)
    X = np.zeros((2 * fourier.N_HBM + 1) * fourier.n_dof)
    ode = SmoothedMeissner(
        t=0, x=np.array([0.0, 0.0]), a=1, b=1, omega=1, damping=0, smoothing=1
    )

    if subh:
        stabmethod = KoopmanHillSubharmonic(fourier=fourier)
    else:
        stabmethod = KoopmanHillProjection(fourier=fourier)
    hbm = HBMEquation(
        ode=ode,
        omega=ode.omega,
        fourier=fourier,
        initial_guess=X,
        stability_method=stabmethod,
        period_k=1,
    )
    _, magnitudes_fm = ince_strutt_rastered(a_grid, b_grid, hbm)
    ax = plot_magnitude(magnitudes_fm, logscale=True, a_grid=a_grid, b_grid=b_grid)

    return ax, (hbm, ode, a_grid, b_grid)


def shoot_stab_bdry(ode, a_0, b_0=0.05, period_doubling=False, b_max=3, a_min=-0.5):
    shoot_eq = InceStruttShootingEquation(ode, period_doubling=period_doubling)
    solver = NewtonSolver(verbose=True)

    bdry = []
    for direction in (1, -1):
        print(f"initial guess: a = {a_0}, b = {b_0}, direction = {direction}")
        shoot_eq.a = a_0
        shoot_eq.ode.a = a_0
        shoot_eq.b = b_0
        shoot_eq.ode.b = b_0

        sys = EquationSystem(equations=[shoot_eq], unknowns=["a"])
        solver.solve(sys)
        solver.verbose = False

        for branch_point in pseudo_arclength_continuator(
            sys,
            solver,
            continuation_parameter="b",
            num_steps=100,
            stepsize=0.05,
            initial_direction=direction,
        ):
            print(f"next branch point: a = {branch_point.a}, b = {branch_point.b}")

            if direction > 0:
                bdry.append([branch_point.a, branch_point.b])
            else:
                bdry.insert(0, [branch_point.a, branch_point.b])

            if branch_point.b > b_max or branch_point.b < 0 or branch_point.a < a_min:
                break

    return np.array(bdry)


def compute_all_stab_bdries(ode, a_max, a_min, b_max, ax=None, **kwargs_plot):
    b_0s = [0.03, 0.03, 0.03, 0.5, 0.5, 1.75, 1.75]
    a_0s = [0, 0.24, 0.26, 0.95, 1.1, 2.35, 2.5]
    pds = [False, True, True, False, False, True, True]
    bdries = []

    for a_base, b_base, pd in zip(a_0s, b_0s, pds):
        if a_base > a_max:
            continue

        a_0 = a_base
        b_0 = b_base
        bdry = shoot_stab_bdry(
            ode, a_0=a_0, b_0=b_0, period_doubling=pd, b_max=b_max, a_min=a_min
        )
        bdries.append(bdry)
        if ax is not None:
            ax.plot(bdry[:, 0], bdry[:, 1], **kwargs_plot)

    return bdries


def plot_stab_bdry_hbm(hbm, ode, a_grid, b_grid, ax):
    # Continuation along stability boundary

    as_tongue = [0]
    solver = NewtonSolver(verbose=True)

    for a in as_tongue:
        solver.verbose = True
        hbm.ode.a = a
        hbm.ode.b = 0.1
        hbm.X[2] = 1  # nonzero initial guess
        hbm.X[3] = -ode.omega
        sys_IS = InceStruttSystem(hbm, varying_parameter="a")
        print(
            f"Initial guess for period-{sys_IS.equations[0].period_k} IS solution: a = {sys_IS.equations[0].ode.a}, b = {sys_IS.equations[0].ode.b}"
        )
        solver.solve(sys_IS)
        assert sys_IS.solved
        print(
            f"Solved IS system at initial guess for period-{sys_IS.equations[0].period_k} solution: a = {sys_IS.equations[0].ode.a}, b = {sys_IS.equations[0].ode.b}"
        )

        _as = []
        bs = []
        solver.verbose = False

        for branch_point in pseudo_arclength_continuator(
            sys_IS,
            solver,
            continuation_parameter="b",
            num_steps=100,
            stepsize=0.05,
            initial_direction=1,
        ):
            print(f"next branch point: a = {branch_point.a}, b = {branch_point.b}")
            _as.append(branch_point.a)
            bs.append(branch_point.b)
            if (not min(a_grid) < branch_point.a < max(a_grid)) or (
                not min(b_grid) < branch_point.b < max(b_grid)
            ):
                break
        ax.plot(_as, bs)


def ince_strutt_rastered(a_grid: Iterable, b_grid: Iterable, hbm: HBMEquation):
    """Return a len(a_grid)*len(b_grid)*n_dof array of solved HBM equations corresponding to the (a, b) grid"""

    solver = NewtonSolver()
    ince_strutt = []
    magnitude_fm = np.zeros((len(b_grid), len(a_grid)))

    for k, b in enumerate(b_grid):
        list_b = []
        print(f"b = {b:.3f} ({k:2d}/{len(b_grid):2d}) ", end="\n")

        for l, a in enumerate(a_grid):
            hbm = copy(hbm)
            hbm.a = a
            hbm.b = b
            solver.solve_equation(equation=hbm, unknown="X")  # solved at initial guess
            # Update the derivative as it was not used when solved at initial guess
            hbm.derivative("X", update=True)
            stable = hbm.determine_stability(update=True)
            list_b.append(hbm)
            magnitude_fm[k, l] = np.max(np.abs(hbm.eigenvalues))

        ince_strutt.append(list_b)

    return ince_strutt, magnitude_fm


def plot_magnitude(magnitude_fm, logscale=True, a_grid=None, b_grid=None):

    # if logscale:
    #     magnitude_fm = np.log10(magnitude_fm)
    # else:
    #     # ensure that stability cutoff is at magnitude = 0
    #     magnitude_fm = magnitude_fm - 1

    shade = np.maximum(1, np.minimum(magnitude_fm, 10))  # np.maximum(0, magnitude_fm)

    if logscale:
        norm = "log"
    else:
        norm = "linear"

    # Show image
    fig, ax = plt.subplots()
    image = ax.pcolormesh(a_grid, b_grid, shade, cmap="gray_r", norm=norm)
    cbar = fig.colorbar(image, label="largest FM (magnitude)")
    ax.set_xlabel("$a$")
    ax.set_ylabel("$b$")
    cbar.set_ticks((1, 10), labels=("$\\leq 10^0$ (stable)", "$\\geq 10^1$"))

    return ax


def test_plotting_magnitudes():
    magnitudes = 10 ** (3 * np.random.rand(150, 150))
    a_grid = -3 + np.arange(magnitudes.shape[1])
    b_grid = 5 + np.arange(magnitudes.shape[0])

    for logscale in (True, False):
        plt.figure()
        plot_magnitude(magnitudes, logscale=logscale, a_grid=a_grid, b_grid=b_grid)

    plt.show()


def plot_ince_strutt_bdries(ode, axs, a_grid, b_grid):
    bdries = compute_all_stab_bdries(
        ode,
        a_max=max(a_grid),
        a_min=min(a_grid),
        b_max=max(b_grid),
    )

    for ax in axs:
        for bdry in bdries:
            ax.plot(bdry[:, 0], bdry[:, 1], "r-")
        ax.set_xlim(min(a_grid), max(a_grid))
        ax.set_ylim(min(b_grid), max(b_grid))

    return bdries


if __name__ == "__main__":

    N_HBM = 3
    a_max = 3.5
    b_max = 3
    pixelsize = 0.01

    subhs = [True, False]
    labels = ["subharmonic", "direct"]

    ode = MathieuODE(t=0, x=np.array([0.0, 0.0]), a=1, b=1, omega=1, damping=0)
    # test_plotting_magnitudes()
    axs = []
    for subh, label in zip(subhs, labels):
        ax, (_, ode, a_grid, b_grid) = plot_ince_strutt(
            N_HBM=N_HBM, subh=subh, a_max=a_max, b_max=b_max, pixelsize=0.01
        )
        axs.append(ax)
        ax.set_title(f"N = {N_HBM}, {label}")

    plot_ince_strutt_bdries(ode, axs, a_grid, b_grid)
    for ax, label in zip(axs, labels):
        save_png(ax, f"ince_strutt_N_{N_HBM}_{label}.png", dpi=900)

    # plot_stab_bdry(*args, ax)
    # shoot_stab_bdry(ode, a_0=0.0, period_doubling=False, b_max=3)

    plt.show()
