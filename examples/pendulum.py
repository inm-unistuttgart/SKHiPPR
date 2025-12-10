"""Illustrate HBM applied to DAEs and ODEs, with the example of the simple pendulum."""

import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("error", category=np.exceptions.ComplexWarning)


from skhippr.odes.daes import PendulumDAE, PendulumODE
from skhippr.Fourier import Fourier
from skhippr.solvers.newton import NewtonSolver
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.cycles.hbm import HBMEquation, HBMEquationDAE

from skhippr.stability.KoopmanHillProjection import (
    KoopmanHillSubharmonic,
    KoopmanHillDAE,
)


def main():

    solver = NewtonSolver(tolerance=1e-8, max_iterations=50, verbose=True)

    m = 1
    g = 9.81
    l = 1.0
    d = 0.05
    F = 0.5
    omega = 1.15
    phi = 0.0

    N_HBM = 15

    # ODE formulation
    ode = PendulumODE(m, d, g, l, F, omega, phi)
    dae_good = PendulumDAE(m, d, g, l, F, omega, phi)
    dae_bad = PendulumDAE(m, d, g, l, F, omega, phi)

    # HBM systems
    fourier_ode = Fourier(
        N_HBM=N_HBM, L_DFT=2000, n_dof=ode.n_dof, real_formulation=True
    )
    hbm_ode = HBMEquation(
        ode,
        omega,
        fourier=fourier_ode,
        initial_guess=np.zeros(ode.n_dof * (2 * fourier_ode.N_HBM + 1)),
        stability_method=KoopmanHillSubharmonic(
            fourier_ode, tol=1e-4, autonomous=False
        ),
    )
    sys_ode = EquationSystem(
        equations=[hbm_ode], unknowns="X", equation_determining_stability=hbm_ode
    )
    solver.solve(sys_ode)
    print("Solved ODE. \n")

    x_ode = hbm_ode.x_time()
    phi = x_ode[0, :]
    phi_dot = x_ode[1, :]
    x_dae_init = np.vstack(
        [
            l * np.sin(phi),
            -l * np.cos(phi),
            l * np.cos(phi) * phi_dot,
            l * np.sin(phi) * phi_dot,
            np.zeros_like(phi),
        ]
    )

    fourier_dae = Fourier(
        N_HBM=3, L_DFT=2000, n_dof=dae_good.n_dof, real_formulation=False
    )
    hbm_dae_good = HBMEquationDAE(
        dae_good,
        omega,
        fourier=fourier_dae,
        initial_guess=fourier_dae.DFT(x_dae_init),
        stability_method=KoopmanHillDAE(
            fourier_dae, tol=0, autonomous=False, tol_drazin=1e-5
        ),
    )
    sys_dae = EquationSystem(
        equations=[hbm_dae_good],
        unknowns="X",
        equation_determining_stability=hbm_dae_good,
    )

    solver.solve(sys_dae)
    print("Solved DAE. \n")

    fourier_dae = Fourier(
        N_HBM=4, L_DFT=2000, n_dof=dae_good.n_dof, real_formulation=False
    )
    hbm_dae_bad = HBMEquationDAE(
        dae_bad,
        omega,
        fourier=fourier_dae,
        initial_guess=fourier_dae.DFT(x_dae_init),
        stability_method=KoopmanHillDAE(
            fourier_dae, tol=0, autonomous=False, tol_drazin=1e-5
        ),
    )
    sys_dae = EquationSystem(
        equations=[hbm_dae_bad],
        unknowns="X",
        equation_determining_stability=hbm_dae_bad,
    )

    solver.solve(sys_dae)
    print("Solved DAE. \n")

    x_dae_solved = hbm_dae_good.x_time()
    t = hbm_dae_good.fourier.time_samples(omega=hbm_dae_good.omega)

    fig, axs = plt.subplots(4, 2)

    for i in range(4):
        axs[i][0].plot(t, x_dae_init[i, :])
        axs[i][0].plot(t, x_dae_solved[i, :], "--")
        axs[i][0].plot(t, hbm_dae_bad.x_time()[i, :], ":")

        axs[i][1].semilogy(t, np.abs(x_dae_init[i, :] - x_dae_init[i, :]))
        axs[i][1].semilogy(t, np.abs(x_dae_init[i, :] - x_dae_solved[i, :]), "--")
        axs[i][1].semilogy(
            t, np.abs(x_dae_init[i, :] - hbm_dae_bad.x_time()[i, :]), ":"
        )

    axs[0][0].set_title(
        f"Pendulum ODE (solid) vs. DAE N = {hbm_dae_good.fourier.N_HBM} (dashed) and N = {hbm_dae_bad.fourier.N_HBM} (dotted)"
    )

    axs[0][0].set_title(f"Error between DAE solution and ODE solution")

    fig_FM, ax = plt.subplots(nrows=1, ncols=1)
    phis = np.linspace(0, 2 * np.pi, 250)
    ax.plot(np.cos(phis), np.sin(phis), "gray", label="Unit circle")
    ax.plot(np.real(hbm_ode.eigenvalues), np.imag(hbm_ode.eigenvalues), "x")
    ax.plot(np.real(hbm_dae_good.eigenvalues), np.imag(hbm_dae_good.eigenvalues), "+")
    ax.plot(np.real(hbm_dae_bad.eigenvalues), np.imag(hbm_dae_bad.eigenvalues), "*")
    ax.set_aspect("equal", "box")
    ax.set_title(
        f"Floquet multipliers pendulum N_HBM = {hbm_dae_good.fourier.N_HBM} and {hbm_dae_bad.fourier.N_HBM}"
    )

    if hbm_dae_good.fourier.real_formulation:
        xvals_freqs_good = np.hstack(
            (
                np.arange(0, hbm_dae_good.fourier.N_HBM + 1),
                np.arange(1, hbm_dae_good.fourier.N_HBM + 1),
            )
        )
        xvals_freqs_bad = np.hstack(
            (
                np.arange(0, hbm_dae_bad.fourier.N_HBM + 1),
                np.arange(1, hbm_dae_bad.fourier.N_HBM + 1),
            )
        )
    else:
        xvals_freqs_good = np.arange(
            -hbm_dae_good.fourier.N_HBM, hbm_dae_good.fourier.N_HBM + 1
        )
        xvals_freqs_bad = np.arange(
            -hbm_dae_bad.fourier.N_HBM, hbm_dae_bad.fourier.N_HBM + 1
        )

    X_good = np.reshape(hbm_dae_good.X, shape=(dae_good.n_dof, -1), order="F")
    X_bad = np.reshape(hbm_dae_bad.X, shape=(dae_bad.n_dof, -1), order="F")
    hill_matrix_good = hbm_dae_good.hill_matrix()
    hill_matrix_bad = hbm_dae_bad.hill_matrix()

    fig, axs = plt.subplots(3, 1)
    axs[0].bar(
        xvals_freqs_good,
        np.linalg.norm(X_good, ord=2, axis=0),
    )
    axs[0].set_yscale("log")
    axs[0].set_title("DAE good case")
    axs[1].bar(
        xvals_freqs_bad,
        np.linalg.norm(X_bad, ord=2, axis=0),
    )
    axs[1].set_yscale("log")
    axs[1].set_title("DAE bad case")

    # Remove all additional harmonics
    if hbm_dae_bad.fourier.real_formulation:

        idx_cos_end = hbm_dae_good.fourier.N_HBM + 1
        idx_sin_start = hbm_dae_bad.fourier.N_HBM + 1
        idx_sin_end = hbm_dae_bad.fourier.N_HBM + hbm_dae_good.fourier.N_HBM + 1

        idx_low_harmos = np.r_[0:idx_cos_end, idx_sin_start:idx_sin_end]
        X_bad = X_bad[:, idx_low_harmos]

        idx_low_harmos_n = np.r_[
            0 : dae_good.n_dof * idx_cos_end,
            dae_good.n_dof * idx_sin_start : dae_good.n_dof * idx_sin_end,
        ]
        hill_matrix_bad = hill_matrix_bad[np.ix_(idx_low_harmos_n, idx_low_harmos_n)]

    else:
        idx_start = hbm_dae_bad.fourier.N_HBM - hbm_dae_good.fourier.N_HBM
        idx_end = hbm_dae_bad.fourier.N_HBM + hbm_dae_good.fourier.N_HBM + 1
        X_bad = X_bad[:, idx_start:idx_end]

        hill_matrix_bad = hill_matrix_bad[
            dae_good.n_dof * idx_start : dae_good.n_dof * idx_end,
            dae_good.n_dof * idx_start : dae_good.n_dof * idx_end,
        ]

    axs[2].bar(
        xvals_freqs_good,
        np.linalg.norm(X_good - X_bad, ord=2, axis=0),
    )
    axs[2].set_yscale("log")
    axs[2].set_title("Difference")

    plt.figure()
    plt.title("Hill's matrices difference norm")
    plt.imshow(np.abs(hill_matrix_good - hill_matrix_bad), cmap="viridis")
    plt.colorbar(label="Absolute difference")


if __name__ == "__main__":
    main()
    plt.show()
