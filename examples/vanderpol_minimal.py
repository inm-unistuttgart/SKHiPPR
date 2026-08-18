"""Minimal example  / template for creating a simple bifurcation diagram of an autonomous dynamical system with HBM.
This example uses a Van der Pol oscillator, but ``ode`` can be instantiated as any autonomous :py:class:`~skhippr.odes.AbstractODE.AbstractODE`.
"""

from typing import override
import numpy as np
import matplotlib.pyplot as plt

# --- Fourier configuration ---
from skhippr.Fourier import Fourier

# --- Differential equation ---
from skhippr.odes.AbstractODE import AbstractODE

# --- HBM equation system ---
from skhippr.cycles.hbm import HBMSystem

# --- Stability method ---
from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic

# --- Continuation ---
from skhippr.solvers.continuation import pseudo_arclength_continuator, BranchPoint

# --- Newton solver ---
from skhippr.solvers.newton import NewtonSolver

# quick visualization
from skhippr.visualization.continuation import plot_continuation


class Vanderpol(AbstractODE):
    """
    Autonomous van der Pol oscillator as a subclass of :py:class:`~skhippr.odes.AbstractODE.AbstractODE`. ::

        dx[0]/dt = x[1]
        dx[1]/dt = nu * (1 - x[0]**2) * x[1] - x[0]

    """

    def __init__(self, x: np.ndarray, nu: float, t=0):
        super().__init__(autonomous=True, n_dof=2)
        self.nu = nu
        self.x = x
        self.t = t

    @override
    def dynamics(self, t=None, x=None):

        if x is None:
            x = self.x

        self.check_dimensions(x=x)

        f = np.zeros_like(x)
        f[0, ...] = x[1, ...]
        f[1, ...] = self.nu * (1 - x[0, ...] ** 2) * x[1, ...] - x[0, ...]
        return f

    @override
    def closed_form_derivative(self, variable, t=None, x=None):
        if x is None:
            x = self.x

        self.check_dimensions(t=t, x=x)

        match variable:
            case "x":
                df_dx = np.zeros((2, *x.shape), dtype=x.dtype)
                df_dx[0, 1, ...] = 1
                df_dx[1, 0, ...] = -1 - 2 * self.nu * x[0, ...] * x[1, ...]
                df_dx[1, 1, ...] = self.nu * (1 - x[0, ...] ** 2)
                return df_dx
            case "nu":
                df_dnu = np.zeros_like(x)
                df_dnu[1, ...] = (1 - x[0, ...] ** 2) * x[1, ...]
                return df_dnu[:, np.newaxis, ...]
            case _:
                raise NotImplementedError(
                    f"Derivative w.r.t {variable} not implemented in closed form."
                )


def compute_frc(nu_range=(0, 5), max_stepsize=0.1):
    """
    Demonstration for creating a simple frequency response curve of a non-autonomous dynamical system with HBM.

    This function performs the following steps:

    #. Creation of :py:class:`~skhippr.Fourier`, :py:class:`~skhippr.solvers.newton.NewtonSolver`, and :py:class:`~skhippr.stability.KoopmmanHillProjection.KoopmanHillSubharmonic` objects to collect method parameters
    #. Instantiation of a :py:class:`~skhippr.odes.AbstractODE.AbstractODE` (here: :py:class:`~skhippr.odes.nonautonomous.Duffing`) object which contains the ODE
    #. Setup of an initial guess
    #. Setup and solution of the :py:class:`~skhippr.cycles.hbm.HBMEquation`, which formalizes the Harmonic Balance equations
    #. Creation of an :py:class:`~skhippr.equations.EquationSystem.EquationSystem` containing only the HBM equations as input to the continuation method
    #. Continuation of the frequency response curve using :py:func:`~skhippr.cycles.continuation.pseudo_arclength_continuator` and collecting the branch points
    #. Plotting the continuation curve from the collected :py:class:`~skhippr.solvers.continuation.BranchPoint` objects via SKHiPPR visualization tools.

    Returns
    -------

    None
    """

    # --- Parameters and creation of ODE (Van der Pol oscillator) ---

    ode = Vanderpol(t=0, x=[2.0, 0.0], nu=nu_range[0])

    # --- FFT, stability method and Newton solver configuration ---
    N_HBM = 45
    L_DFT = 1000

    fourier = Fourier(N_HBM=N_HBM, L_DFT=L_DFT, n_dof=2, real_formulation=True)
    stability_method = KoopmanHillSubharmonic(fourier, tol=1e-4, autonomous=True)
    solver = NewtonSolver(verbose=False)

    # --- Initial guess in time and frequency domain ---
    ts = fourier.time_samples(1)
    x0_samples = np.array([2 * np.cos(ts), -2 * np.sin(ts)])
    X0 = fourier.DFT(x0_samples)

    # --- Set up the Harmonic Balance equation, immediately as a system to add a phase anchor condition
    initial_system = HBMSystem(
        ode=ode,
        omega=1,
        fourier=fourier,
        initial_guess=X0,
        stability_method=stability_method,
    )

    # BranchPoints (a subclass of EquationSystem) extend the initial EquationSystem with one added equation for the tangency condition.
    frc: list[BranchPoint] = []

    # --- Iterate through the branch. Almost all arguments are optional. ---
    for branch_point in pseudo_arclength_continuator(
        initial_system=initial_system,
        solver=solver,
        stepsize=max_stepsize,
        stepsize_range=(0.001, max_stepsize),
        continuation_parameter="nu",
        initial_direction=nu_range[1] - nu_range[0],
        verbose=True,
        num_steps=4000,
    ):
        frc.append(branch_point)

        # break if nu exceeds maximum
        if branch_point.nu > nu_range[1]:
            break

    return initial_system, frc


if __name__ == "__main__":
    _, frc = compute_frc()
    plot_continuation(
        frc,
        plot_fun=lambda point: np.max(point.equations[0].x_time()[1, :]),
        ylabel="max_t|x_2(t)|",
        title="Van der Pol continuation",
    )
    plt.show()
