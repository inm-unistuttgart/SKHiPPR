"""
Find and visualize a periodic solution of the Duffing oscillator using SKHiPPR.

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

# --- Visualization ---
from skhippr.visualization.cycles import (
    plot_period,
    plot_floquet_multipliers,
    plot_floquet_exponents,
    plot_phase,
    plot_hill_matrix_blocks,
)
from skhippr.visualization.data_export import (
    save_png,
    save_pdf,
    save_tikz
)

def main():
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

    # --- FFT, stability method and Newton solver configuration ---
    fourier = Fourier(N_HBM=30, L_DFT=300, n_dof=2, real_formulation=True)
    stability_method = KoopmanHillSubharmonic(fourier, tol=1e-4)
    solver = NewtonSolver(verbose=True)

    # --- Setup parameters ---
    omega = 0.8
    F = 0.05

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

    # --- Plotting a HBMSystem directly will use the first valid HBMEquation contained in the system ---
    plot_phase(hbm=hbm_sys)

    # --- Optional: keyword arguments can be given for plotting and exctracting a specific HBMEquation from a system---
    kwargs = {"linestyle": "--"}
    hbm_equation = hbm_sys.equations[0]
    plot_phase(hbm=hbm_equation, **kwargs)

    # --- New figure for each plot using standard visualization method calls ---
    plot_floquet_multipliers(hbm=hbm_sys)
    plot_floquet_exponents(hbm=hbm_sys)
    plot_period(hbm=hbm_sys, n_periods=1.22)
    ax = plot_hill_matrix_blocks(hbm=hbm_sys, real_formulation=None, logscale=True)

    # --- Realisation with subplots ---
    _, axs = plt.subplots(nrows=1, ncols=3)
    axs[0] = plot_phase(hbm=hbm_sys, ax=axs[0])
    axs[0].set_title("Phase plot of solution")
    axs[0].set_ylabel("x_1")
    axs[0].set_xlabel("x_0")
    fourier = hbm_sys.equations[0].fourier
    axs[1] = plot_floquet_multipliers(hbm=hbm_sys, ax=axs[1])
    axs[1].set_title("Floquet multipliers")
    axs[1].plot(
        np.cos(fourier.time_samples_normalized),
        np.sin(fourier.time_samples_normalized),
        "k",
    )
    axs[1].axis("equal")
    axs[2] = plot_floquet_exponents(hbm=hbm_sys, ax=axs[2])
    axs[2].set_title("Floquet exponents")

    # --- Save the hill matrix visualization plot ---
    # The relative path for saving a file can be given if the filepath string starts without a "/".
    # Forward slashes "/" can be used regardless of operating system. 
    save_pdf(axes=ax, filepath="plots/duffing_plots/hill_matrix.pdf")
    save_png(axes=ax, filepath="plots/duffing_plots/hill_matrix.png")
    
    # --- Saving using save_tikz requires tikzplotlib to be installed which is imported locally ---
    # save_tikz(axes=ax, filepath="duffing_plots/hill_matrix.tex")


if __name__ == "__main__":
    main()
    plt.show()
