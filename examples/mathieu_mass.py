import numpy as np
import matplotlib.pyplot as plt

from skhippr.odes.ltp import MathieuODE, MathieuWithMass, MathieuWithMassInverted
from skhippr.cycles.hbm import HBMEquation, HBMEquationDAE
from skhippr.Fourier import Fourier
from skhippr.solvers.newton import NewtonSolver

from skhippr.visualization.cycles import plot_phase, plot_period

import warnings
from numpy.exceptions import ComplexWarning  # Corrected import

# Raise an error on ComplexWarning
warnings.filterwarnings("error", category=ComplexWarning)


def main():

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
    t_samples = fourier_ref.time_samples(omega)

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

    hbm_ref = HBMEquation(ode, ode.omega, fourier_ref)
    solver.solve_equation(hbm_ref, "X")
    ax_phase = plot_phase(hbm_ref, label="ref")
    ax_time = plot_period(hbm_ref, label="ref")

    _, ax_error = plt.subplots()

    # solution without mass matrix at small N_HBM

    fourier = Fourier(N_HBM=N_HBM, L_DFT=L_DFT, n_dof=2, real_formulation=True)
    hbm = HBMEquation(ode, ode.omega, fourier)
    solver.solve_equation(hbm, "X")
    plot_phase(hbm, ax=ax_phase, label="no mass", linestyle=":")
    plot_period(hbm, ax=ax_time, label="no mass", linestyle=":")
    error_dir = np.linalg.norm(hbm.x_time() - hbm_ref.x_time(), axis=0)
    ax_error.plot(t_samples, error_dir, "-", label=f"error dir N = {N_HBM}")

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
    hbm_inv = HBMEquation(ode_inv, ode_inv.omega, fourier, initial_guess=hbm.X)
    solver.solve_equation(hbm_inv, "X")
    plot_phase(hbm_inv, ax=ax_phase, label="inverted mass", linestyle=":")
    plot_period(hbm_inv, ax=ax_time, label="inverted mass", linestyle=":")
    error_inv = np.linalg.norm(hbm_inv.x_time() - hbm_ref.x_time(), axis=0)
    ax_error.plot(t_samples, error_inv, "--", label=f"error inv N = {N_HBM}")

    ode_mass = MathieuWithMass(
        t=0,
        x=np.zeros((2, fourier.L_DFT)),
        a=a,
        b=b,
        omega=omega,
        damping=damping,
        forcing=forcing,
    )
    hbm_mass = HBMEquationDAE(ode_mass, ode_mass.omega, fourier, initial_guess=None)

    solver.solve_equation(hbm_mass, "X")
    plot_phase(hbm_mass, ax=ax_phase, label="with mass", linestyle="--")
    plot_period(hbm_mass, ax=ax_time, label="with mass", linestyle="--")
    error_mass = np.linalg.norm(hbm_mass.x_time() - hbm_ref.x_time(), axis=0)
    ax_error.plot(t_samples, error_mass, "-.", label=f"error mass N = {N_HBM}")

    ax_phase.legend()
    ax_time.legend()
    ax_error.set_yscale("log")
    ax_error.set_xlabel("t")
    ax_error.set_ylabel("Error norm")
    ax_error.legend()

    fig, ax_cos = plt.subplots()
    g = ode.a + ode.b * ode.g_fcn(t_samples)
    ax_cos.plot(t_samples, g, "-", label="g")

    g_inv = ode_mass.g(t_samples)
    ax_cos.plot(t_samples, g_inv, "--", label="1/g")
    ax_cos.plot(t_samples, g * g_inv, "-.", label="g*1/g")
    ax_cos.set_xlabel("t")
    ax_cos.legend()

    fig, ax_FC = plt.subplots()
    fourier_scalar = Fourier(
        N_HBM=fourier_ref.N_HBM, L_DFT=fourier.L_DFT, n_dof=1, real_formulation=False
    )
    g_FC = fourier_scalar.DFT(g[np.newaxis, :])
    g_inv_FC = fourier_scalar.DFT(g_inv[np.newaxis, :])

    bar_width = 0.4
    indices = np.arange(-fourier_scalar.N_HBM, fourier_scalar.N_HBM + 1)

    ax_FC.bar(
        indices - bar_width / 2,
        np.squeeze(np.abs(g_FC)),
        width=bar_width,
        label="g",
    )
    ax_FC.bar(
        indices + bar_width / 2,
        np.squeeze(np.abs(g_inv_FC)),
        width=bar_width,
        label="g inv",
    )
    ax_FC.set_xlabel("Harmonic index")
    ax_FC.set_yscale("log")
    ax_FC.legend()


if __name__ == "__main__":
    main()
    plt.show()
