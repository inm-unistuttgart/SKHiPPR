import numpy as np
import matplotlib.pyplot as plt

from skhippr.odes.ltp import MathieuODE, MathieuWithMass, MathieuWithMassInverted
from skhippr.cycles.hbm import HBMEquation, HBMEquationDAE
from skhippr.Fourier import Fourier
from skhippr.solvers.newton import NewtonSolver
from skhippr.odes.AbstractODE import AbstractDAE

from skhippr.visualization.cycles import plot_phase, plot_period

import warnings
from numpy.exceptions import ComplexWarning  # Corrected import

# Raise an error on ComplexWarning
warnings.filterwarnings("error", category=ComplexWarning)


def main():

    # Init

    a = 1
    b = 0.9
    omega = 1.0
    damping = 0.05
    forcing = 1

    N_HBM = 3
    N_ref = 30
    L_DFT = 1024

    solver = NewtonSolver(verbose=True)
    fourier_ref = Fourier(N_HBM=N_ref, L_DFT=L_DFT, n_dof=2, real_formulation=True)
    fourier = fourier_ref.__replace__(N_HBM=N_HBM)

    ode = MathieuODE(
        t=0,
        x=np.zeros((2, fourier_ref.L_DFT)),
        a=a,
        b=b,
        omega=omega,
        damping=damping,
        forcing=forcing,
    )

    # Reference solution
    hbm_ref, axes = hbm_and_plot(
        equ=ode, fourier=fourier_ref, solver=solver, label="ref"
    )

    # solution without mass matrix at small N_HBM
    hbm, axes = hbm_and_plot(
        equ=ode,
        fourier=fourier,
        solver=solver,
        axes=axes,
        hbm_ref=hbm_ref,
        linestyle="--",
        label="no mass",
    )

    # solution with inverted mass matrix
    ode_inv = MathieuWithMassInverted(
        t=0,
        x=np.zeros((2, fourier.L_DFT)),
        a=a,
        b=b,
        omega=omega,
        damping=damping,
        forcing=forcing,
    )
    hbm_inv, axes = hbm_and_plot(
        equ=ode_inv,
        fourier=fourier,
        solver=solver,
        axes=axes,
        hbm_ref=hbm_ref,
        initial_guess=hbm.X,
        label="inverted mass",
        linestyle=":",
    )

    ode_mass = MathieuWithMass(
        t=0,
        x=np.zeros((2, fourier.L_DFT)),
        a=a,
        b=b,
        omega=omega,
        damping=damping,
        forcing=forcing,
    )
    hbm_mass, axes = hbm_and_plot(
        equ=ode_mass,
        fourier=fourier,
        solver=solver,
        axes=axes,
        dae=True,
        hbm_ref=hbm_ref,
        label="with mass",
        linestyle="-.",
    )

    for ax in axes:
        ax.legend()

    g, g_inv = plot_g_functions(ode, ode_mass, fourier)
    plot_fourier_coeffs(fourier_ref.N_HBM, g, g_inv)


def plot_g_functions(ode, ode_mass, fourier):
    """Plot g and 1/g functions in time and frequency domain."""
    t_samples = fourier.time_samples(ode.omega)

    # Time domain plot
    _, ax_cos = plt.subplots()
    g = ode.a + ode.b * ode.g_fcn(t_samples)
    ax_cos.plot(t_samples, g, "-", label="g")

    g_inv = ode_mass.g(t_samples)
    ax_cos.plot(t_samples, g_inv, "--", label="1/g")
    ax_cos.plot(t_samples, g * g_inv, "-.", label="g*1/g")
    ax_cos.set_xlabel("t")
    ax_cos.legend()

    return g, g_inv


def plot_fourier_coeffs(N_HBM=30, *funs):
    """Frequency domain plot"""
    _, ax = plt.subplots()

    bar_width = (1 - 0.1) / len(funs)
    indices = np.arange(-N_HBM, N_HBM + 1)

    for i, fun in enumerate(funs):
        if len(fun.shape) == 1:
            fun = fun[np.newaxis, :]
        fourier = Fourier(
            N_HBM=N_HBM, L_DFT=fun.shape[1], n_dof=fun.shape[0], real_formulation=False
        )
        coeffs = fourier.DFT(fun)
        coeffs = np.reshape(coeffs, (fun.shape[0], -1), order="F")
        x = indices + (i - (len(funs) - 1) / 2) * bar_width
        ax.bar(
            x,
            np.linalg.norm(coeffs, axis=0),
            width=bar_width,
        )

    ax.set_xlabel("Harmonic index")
    ax.set_yscale("log")
    return ax


def hbm_and_plot(
    equ,
    fourier,
    solver,
    axes=(None, None, None),
    dae=False,
    hbm_ref: HBMEquation = None,
    initial_guess=None,
    **kwargs_plot,
):
    if dae:
        hbm = HBMEquationDAE(
            dae=equ, omega=equ.omega, fourier=fourier, initial_guess=initial_guess
        )
    else:
        hbm = HBMEquation(
            ode=equ, omega=equ.omega, fourier=fourier, initial_guess=initial_guess
        )

    solver.solve_equation(hbm, "X")

    axes = list(axes)
    axes[0] = plot_phase(hbm, ax=axes[0], **kwargs_plot)
    axes[1] = plot_period(hbm, ax=axes[1], **kwargs_plot)

    if hbm_ref is not None:
        if axes[2] is None:
            _, axes[2] = plt.subplots()
            axes[2].set_yscale("log")
            axes[2].set_xlabel("t")
            axes[2].set_ylabel("Error norm")

        t_samples = fourier.time_samples(equ.omega)
        error_dir = np.linalg.norm(hbm.x_time() - hbm_ref.x_time(), axis=0)
        axes[2].plot(t_samples, error_dir, **kwargs_plot)

    return hbm, axes


if __name__ == "__main__":
    main()
    plt.show()
