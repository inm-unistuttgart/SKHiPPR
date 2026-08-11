"""
Demonstrates the visualization capabilities of SKHiPPR for periodic solutions using a Duffing oscillator.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Fourier configuration ---
from skhippr.Fourier import Fourier

# --- System function ---
from skhippr.odes.nonautonomous import Duffing

# --- HBM ---
from skhippr.cycles.hbm import HBMSystem

# --- Stability method ---
from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic

# --- Solver ---
from skhippr.solvers.newton import NewtonSolver

# --- Continuation ---
from skhippr.solvers.continuation import pseudo_arclength_continuator

# --- Visualization ---
from skhippr.visualization.cycles import (
    plot_period,
    plot_phase,
    plot_floquet_multipliers,
    animate_floquet_multipliers,
    plot_floquet_exponents,
    animate_floquet_exponents,
    plot_hill_matrix_blocks,
)
from skhippr.visualization.continuation import plot_continuation

from skhippr.visualization.data_export import (
    save_png,
    save_pdf,
)


def visualize_single_solution(hbm_sys: HBMSystem, save=False):
    """
    Demonstrates solving an ordinary differential equation for a periodic solution using HBM and visualizes properties of the solved :py:class:`~skhippr.cycles.hbm.HBMSystem` using SKHiPPR.

    This function does the following:

    #. instantiate a :py:class:`~skhippr.cycles.hbm.HBMSystem` from a :py:class:`~skhippr.odes.nonautonomous.Duffing` object
    #. solve it using :py:func:`skhippr.solvers.newton.NewtonSolver.solve`
    #. visualize solution properties.

    It generates multiple figures:

    #. A phase plot of the periodic solution
    #. A phase plot of the periodic solution with additional keyword arguments
    #. A plot of the Floquet multipliers in the complex plane, along with the unit circle for reference
    #. A plot of the Floquet exponents in the complex plane
    #. A plot of the Hill matrix entries colored by their 2-norm
    #. A plot of the time series over a non-integer amount of periods
    #. A figure containing three of the plots as subplots.

    Then demonstrates saving one of the plots using SKHiPPR visualization export functions.
    """

    omega = hbm_sys.equations[0].omega

    # --- Plotting a HBMSystem directly will use the first valid HBMEquation contained in the system ---
    ax_phase = plot_phase(
        hbm=hbm_sys,
        title=f"Phase plot (Duffing, omega = {omega}), HBMSystem passed",
        xlabel="x_0",
        ylabel="x_1",
    )

    # --- Alternatively, a specific HBMEquation can be extracted from the system and passed to the plotting function(s) ---
    hbm_equation = hbm_sys.equations[0]
    ax_phase_equ = plot_phase(
        hbm=hbm_equation,
        title=f"Phase plot (Duffing, omega = {omega}), HBMEquation passed",
        xlabel="x_0",
        ylabel="x_1",
        linestyle="--",
        color="r",
    )

    # --- Same for the period in time
    ax_period = plot_period(
        hbm=hbm_sys,
        n_periods=1.22,
        title=f"Time series (Duffing, omega = {omega})",
    )

    # --- Stability measures ---
    ax_FM = plot_floquet_multipliers(
        hbm=hbm_sys,
        title=f"Floquet multipliers (Duffing, omega = {omega}), HBMSystem passed",
    )
    ax_FE = plot_floquet_exponents(
        hbm=hbm_sys,
        title=f"Floquet exponents (Duffing, omega = {omega}), HBMSystem passed",
    )

    # --- Visualize the Hill matrix ---
    ax_hill_matrix = plot_hill_matrix_blocks(
        hbm=hbm_sys, real_formulation=None, logscale=True, cmap="plasma"
    )

    # All the visualization functions return the axis object in which the plot lives,
    # and they can also be passed an axis object to plot into. This allows for the use of subplots.

    # --- Realisation with subplots ---
    _, axs = plt.subplots(nrows=2, ncols=2)
    plot_phase(hbm=hbm_sys, ax=axs[0][0])
    plot_floquet_multipliers(hbm=hbm_sys, ax=axs[0][1])
    axs[0][1].axis("equal")
    plot_floquet_exponents(hbm=hbm_sys, ax=axs[1][0])

    if save:
        # --- Save the hill matrix visualization plot ---
        # The relative path for saving a file can be given if the filepath string starts without a "/".
        # Forward slashes "/" can be used regardless of operating system.
        save_pdf(axes=ax_hill_matrix, filepath="plots/duffing_plots/hill_matrix.pdf")
        save_png(axes=ax_hill_matrix, filepath="plots/duffing_plots/hill_matrix.png")


def visualize_continuation_result(frc):
    # --- Plot the continuation curve using SKHiPPR visualization functions---

    # for the continuation plot, define the 1-3 scalar quantities plotted against each other
    def plot_fun_1d(branch_point):
        # return the maximum of the first unknown in time domain. If only one parameter is returned, the continuation param appears on the x axis.
        return np.max(branch_point.equations[0].x_time()[0, :])

    def plot_fun_2d(branch_point):
        # return the maximum of the first unknown in time domain and the continuation parameter. If two parameters are returned, they appear on the x and y axis in their respective order. This allows, for instance, to produce a rotated plot.
        return (
            np.max(branch_point.equations[0].x_time()[0, :]),
            branch_point.omega,
        )

    def plot_fun_3d(branch_point):
        # return the continuation parameter, the maximum of the first unknown in the time domain, and the amplitude of the first harmonic.If three parameters are returned, they appear on the x, y and z axis in their respective order.
        return (
            branch_point.omega,
            np.max(branch_point.equations[0].x_time()[0, :]),
            branch_point.X[branch_point.equations[0].fourier.n_dof],
        )

    ax_1d = plot_continuation(
        frc,
        plot_fun=plot_fun_1d,
        title="Continuation curve (Duffing, 1D plot)",
        xlabel="omega",
        ylabel="max(x_0(t))",
    )

    ax_2d = plot_continuation(
        frc,
        plot_fun=plot_fun_2d,
        title="Flipped Continuation curve (Duffing, 2D plot)",
        xlabel="max(x_0(t))",
        ylabel="omega",
    )

    ax_3d = plot_continuation(
        frc,
        plot_fun=plot_fun_3d,
        title="Continuation curve (Duffing, 3D plot)",
        xlabel="omega",
        ylabel="max(x_0(t))",
        zlabel="X_1",
    )


def animate_continuation_result(frc):
    # --- Animate the Floquet multipliers and exponents along the continuation branch ---
    _, animation1 = animate_floquet_multipliers(hbm_set=frc)
    _, animation2 = animate_floquet_exponents(hbm_set=frc)
    return (
        animation1,
        animation2,
    )


def main():

    # --- FFT, stability method and Newton solver configuration ---
    fourier = Fourier(N_HBM=30, L_DFT=300, n_dof=2, real_formulation=True)
    stability_method = KoopmanHillSubharmonic(fourier, tol=1e-4)
    solver = NewtonSolver(verbose=True)

    # --- Setup parameters ---
    omega = 0.3
    F = 0.5

    # --- Instantiation of the ODE at initial point ---
    ode = Duffing(t=0, x=[1.0, 0.0], alpha=1, beta=2, delta=0.16, F=F, omega=omega)

    # --- Initial guess in time and frequency domain ---
    ts = fourier.time_samples(omega)
    x0_samples = np.array([np.cos(ts * omega), -omega * np.sin(ts * omega)])
    X0 = fourier.DFT(x0_samples)

    # --- HBM equation system setup ---
    hbm_sys = HBMSystem(
        ode=ode,
        omega=ode.omega,
        fourier=fourier,
        initial_guess=X0,
        stability_method=stability_method,
    )

    # --- Solve initial point and visualize---
    solver.solve(hbm_sys)
    assert hbm_sys.solved

    visualize_single_solution(hbm_sys, save=False)

    # --- Continuation and visualize ---

    # --- Preallocate the result of the FRC continuation ---
    # BranchPoints (a subclass of EquationSystem) extend the initial EquationSystem with one added equation for the tangency condition.
    frc = []

    # it is a good idea to turn off the solver verbosity before starting a continuation
    solver.verbose = False

    # --- Iterate through the branch. ---
    for branch_point in pseudo_arclength_continuator(
        initial_system=hbm_sys,
        solver=solver,
        stepsize=0.1,
        stepsize_range=(0.001, 0.1),
        continuation_parameter="omega",
        initial_direction=1,
        verbose=True,
        num_steps=4000,
    ):
        frc.append(branch_point)

        # break if omega exceeds maximum
        if branch_point.omega > 2.5:
            break

    visualize_continuation_result(frc)
    animations = animate_continuation_result(frc)

    return animations


if __name__ == "__main__":
    animations = main()
    plt.show()
