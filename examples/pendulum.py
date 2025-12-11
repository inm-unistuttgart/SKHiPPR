"""Illustrate HBM applied to DAEs and ODEs, with the example of the simple pendulum."""

import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("error", category=np.exceptions.ComplexWarning)


from skhippr.odes.daes import PendulumDAE, PendulumODE
from skhippr.Fourier import Fourier
from skhippr.solvers.newton import NewtonSolver, ScipyFsolveSolver
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.cycles.hbm import HBMEquation, HBMEquationDAE
from skhippr.solvers.continuation import pseudo_arclength_continuator

from skhippr.stability.KoopmanHillProjection import (
    KoopmanHillSubharmonic,
    KoopmanHillDAE,
)


def plot_single_solution():

    solver_newton = NewtonSolver(tolerance=1e-8, max_iterations=50, verbose=True)
    solver = ScipyFsolveSolver(
        tolerance=1e-8, max_iterations=1000, verbose=True, use_fprime=True
    )

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
    dae = PendulumDAE(m, d, g, l, F, omega, phi)

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

    fourier_dae = Fourier(N_HBM=15, L_DFT=2000, n_dof=dae.n_dof, real_formulation=True)
    hbm_dae = HBMEquationDAE(
        dae,
        omega,
        fourier=fourier_dae,
        initial_guess=fourier_dae.DFT(x_dae_init),
        stability_method=KoopmanHillDAE(
            fourier_dae, tol=0, autonomous=False, tol_drazin=1e-5
        ),
    )
    print(f"Residual before solve: {np.max(np.abs(hbm_dae.residual(update=True)))}")
    sys_dae = EquationSystem(
        equations=[hbm_dae],
        unknowns="X",
        equation_determining_stability=hbm_dae,
    )

    solver.solve(sys_dae)
    print(f"Residual after solve: {np.max(np.abs(hbm_dae.residual(update=False)))}")
    print("Solved DAE. \n")

    x_dae_solved = hbm_dae.x_time()
    t = hbm_dae.fourier.time_samples(omega=hbm_dae.omega)

    fig, axs = plt.subplots(4, 2)

    for i in range(4):
        axs[i][0].plot(t, x_dae_init[i, :])
        axs[i][0].plot(t, x_dae_solved[i, :], "--")

        axs[i][1].semilogy(t, np.abs(x_dae_init[i, :] - x_dae_init[i, :]))
        axs[i][1].semilogy(t, np.abs(x_dae_init[i, :] - x_dae_solved[i, :]), "--")

    axs[0][0].set_title(
        f"Pendulum ODE (solid) vs. DAE N = {hbm_dae.fourier.N_HBM} (dashed)"
    )

    axs[0][0].set_title(f"Error between DAE solution and ODE solution")

    fig_FM, ax = plt.subplots(nrows=1, ncols=1)
    phis = np.linspace(0, 2 * np.pi, 250)
    ax.plot(np.cos(phis), np.sin(phis), "gray", label="Unit circle")
    ax.plot(np.real(hbm_ode.eigenvalues), np.imag(hbm_ode.eigenvalues), "x")
    ax.plot(np.real(hbm_dae.eigenvalues), np.imag(hbm_dae.eigenvalues), "+")
    ax.set_aspect("equal", "box")
    ax.set_title(f"Floquet multipliers pendulum N_HBM = {hbm_dae.fourier.N_HBM}")


def plot_frc():

    solver = ScipyFsolveSolver(
        tolerance=1e-8, max_iterations=1000, verbose=False, use_fprime=True
    )

    m = 1
    g = 9.81
    l = 1.0
    d = 0.05
    F = 0.5
    omega = 1.15
    phi = 0.0

    N_HBM = 15

    # Systems
    ode = PendulumODE(m, d, g, l, F, omega, phi)
    dae = PendulumDAE(m, d, g, l, F, omega, phi)

    fourier_ode = Fourier(
        N_HBM=N_HBM, L_DFT=1000, n_dof=ode.n_dof, real_formulation=True
    )
    fourier_dae = Fourier(
        N_HBM=N_HBM, L_DFT=1000, n_dof=dae.n_dof, real_formulation=True
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

    solver.solve_equation(hbm_ode, unknown="X")
    phi = hbm_ode.x_time()[0, :]
    phi_dot = hbm_ode.x_time()[1, :]

    x_dae_init = np.vstack(
        [
            l * np.sin(phi),
            -l * np.cos(phi),
            l * np.cos(phi) * phi_dot,
            l * np.sin(phi) * phi_dot,
            np.zeros_like(phi),
        ]
    )

    dae_initial_guess = np.zeros(dae.n_dof * (2 * fourier_ode.N_HBM + 1))
    dae_initial_guess[4] = 1.0
    dae_initial_guess += 1e-4 * np.random.rand(dae.n_dof * (2 * fourier_ode.N_HBM + 1))
    hbm_dae = HBMEquationDAE(
        dae,
        omega,
        fourier=fourier_dae,
        initial_guess=fourier_dae.DFT(x_dae_init),
        stability_method=KoopmanHillDAE(
            fourier_dae, tol=1e-4, autonomous=False, tol_drazin=1e-6
        ),
    )

    fig, axs = plt.subplots(1, 2)
    for idx, hbm in enumerate([hbm_ode, hbm_dae]):

        sys = EquationSystem(
            equations=[hbm], unknowns=["X"], equation_determining_stability=hbm
        )

        for branch_point in pseudo_arclength_continuator(
            initial_system=sys,
            solver=solver,
            continuation_parameter="omega",
            stepsize=0.01,
            stepsize_range=[0.001, 0.1],
            num_steps=2000,
            verbose=True,
        ):
            if branch_point.stable:
                color = "r"
            else:
                color = "b"
            axs[idx].plot(
                branch_point.omega,
                np.max(np.abs(branch_point.equations[0].x_time()[0, :])),
                f"{color}.",
            )

            if branch_point.omega > 4:
                break

    axs[0].set_title("Pendulum ODE FRC")
    axs[0].set_xlabel("omega")
    axs[0].set_ylabel("phi max")
    axs[1].set_title("Pendulum DAE FRC")
    axs[1].set_xlabel("omega")
    axs[1].set_ylabel("y max")


if __name__ == "__main__":
    plot_single_solution()
    plot_frc()
    plt.show()
