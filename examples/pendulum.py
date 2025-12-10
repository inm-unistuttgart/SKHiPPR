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

    m = 0.8
    g = 9.81
    l = 1.0
    d = 0.5
    F = 1
    omega = 1.15
    phi = 0.0

    N_HBM = 15

    # ODE formulation
    ode = PendulumODE(m, d, g, l, F, omega, phi)
    dae_good = PendulumDAE(m, d, g, l, F, omega, phi)
    dae_bad = PendulumDAE(m, d, g, l, F, omega, phi)

    # HBM systems
    fourier_ode = Fourier(N_HBM=N_HBM, L_DFT=1000, n_dof=ode.n_dof)
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
        N_HBM=5, L_DFT=1000, n_dof=dae_good.n_dof, real_formulation=True
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
        N_HBM=6, L_DFT=1000, n_dof=dae_good.n_dof, real_formulation=True
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
    ax.set_aspect("equal", "box")
    ax.set_title(f"Floquet multipliers pendulum N_HBM = {hbm_dae_good.fourier.N_HBM}")

    print(hbm_dae_good.eigenvalues)
    print(hbm_dae_bad.eigenvalues)


if __name__ == "__main__":
    main()
    plt.show()
