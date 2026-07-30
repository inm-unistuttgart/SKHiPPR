"""Van der Pol oscillator: continuation w.r.t. nu and animation of the resulting phase portrait."""

import matplotlib.pyplot as plt
import numpy as np

# ODE
from skhippr.odes.autonomous import Vanderpol

# FFT configuration
from skhippr.Fourier import Fourier

# HBM and stability
from skhippr.cycles.hbm import HBMSystem
from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic

# Solution procedure
from skhippr.solvers.newton import NewtonSolver
from skhippr.solvers.continuation import pseudo_arclength_continuator

# only for type hinting
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.odes.AbstractODE import AbstractODE
from skhippr.solvers.continuation import BranchPoint

# Visualization
from skhippr.visualization.cycles import (
    animate_period,
    animate_floquet_multipliers,
    animate_phase,
    animate_floquet_exponents,
)
from skhippr.visualization.continuation import plot_continuation
from skhippr.visualization.data_export import save_animation


def main():
    """Demonstration of the continuation of the Van der Pol oscillator w.r.t. nu and animation of the resulting phase portrait.
    This function performs the following steps:

    #. Setup of the :py:class:`~skhippr.odes.autonomous.Vanderpol` ODE
    #. Setup of a :py:class:`~skhippr.cycles.hbm.HBMSystem` object with the :py:class:`~skhippr.odes.autonomous.Vanderpol` ode and a :py:class:`~skhippr.Fourier.Fourier` object, encoding both the HBM equations and the phase anchor.
    #. Continuation of the HBM system w.r.t. nu using the :py:func:`~skhippr.solvers.continuation.pseudo_arclength_continuator` and a :py:class:`~skhippr.solvers.newton.NewtonSolver`
    #. Analysis of the resulting branch of solutions, extracting time series, amplitudes, Floquet multipliers, and stability
    #. Visualization of the results, including an animation of the phase portrait and Floquet multipliers, as well as plots of amplitude and frequency w.r.t. nu
    #. Saving the animation by passing a relative path as a string.
    """

    print("Van der Pol oscillator: continuation w.r.t. nu")

    # --- Setup ---
    newton_solver = NewtonSolver(verbose=True)
    ode = Vanderpol(x=[2.0, 0.0], nu=0.1)
    hbm_system: EquationSystem = setup_hbm_system(ode, newton_solver)

    # --- Continuation ---
    branch: list[BranchPoint] = []
    nu_range = (ode.nu, 5)
    newton_solver.verbose = False

    for branch_point in pseudo_arclength_continuator(
        initial_system=hbm_system,
        solver=newton_solver,
        stepsize=0.1,
        stepsize_range=(0.001, 0.15),
        initial_direction=1,
        num_steps=1000,
        continuation_parameter="nu",
        verbose=True,
    ):
        branch.append(branch_point)
        if not nu_range[0] <= branch_point.nu <= nu_range[1]:
            break

    # --- Create animations from the HBMEquations in the continuation branch ---
    # ax, animation0 = animate_phase(branch, scaling="dynamic")
    # _, animation1 = animate_period(branch, scaling="dynamic")
    # _, animation2 = animate_floquet_multipliers(branch, scaling="unit_circle")
    # _, animation3 = animate_floquet_exponents(branch, scaling="static")
    plot_continuation(
        branch=branch,
        plot_fun=lambda point: np.max(point.equations[0].x_time()[0, :]),
    )
    plot_continuation(branch, plot_fun=lambda point: point.omega)

    # --- Export animations ---
    # Animations can be saved as a .gif and as video files such as .mp4.
    # Video formats require the user to have FFmpeg installed.
    # save_animation(animation0, "plots/vanderpol_animations/phase_animation.gif")

    return  # animation0 #, animation1, animation2, animation3


def setup_hbm_system(ode: AbstractODE, solver: NewtonSolver = None):

    omega_0 = 1
    fourier = Fourier(N_HBM=45, L_DFT=1000, n_dof=ode.n_dof, real_formulation=True)
    stability_method = KoopmanHillSubharmonic(
        fourier=fourier, tol=1e-4, autonomous=True
    )
    X0 = generate_initial_condition(fourier, omega_0)

    hbm_system = HBMSystem(ode, omega_0, fourier, X0, stability_method=stability_method)
    if solver:
        solver.solve(hbm_system)
        print(
            f"Initial problem convergence: {hbm_system.solved}. omega = {hbm_system.omega}"
        )
    return hbm_system


def generate_initial_condition(fourier, omega_0):
    ts = fourier.time_samples(omega_0)
    x0_samples = np.vstack(
        (2 * np.cos(omega_0 * ts), -2 * omega_0 * np.sin(omega_0 * ts))
    )
    X0 = fourier.DFT(x0_samples)
    return X0


if __name__ == "__main__":
    animations = main()
    plt.show()
