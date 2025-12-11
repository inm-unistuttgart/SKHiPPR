"""FRC of frictional oscillator. See Schütz (2025), Bachelor's thesis, and Legrand2023."""

import numpy as np
import matplotlib.pyplot as plt
import warnings

from examples.duffing_3d import visualize_solution


from skhippr.odes.daes import FrictionOscillator
from skhippr.Fourier import Fourier
from skhippr.solvers.newton import ScipyFsolveSolver
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.cycles.hbm import HBMEquation, HBMEquationDAE
from skhippr.solvers.continuation import pseudo_arclength_continuator

from skhippr.stability.KoopmanHillProjection import (
    KoopmanHillSubharmonic,
    KoopmanHillDAE,
)


def plot_frc():

    solver = ScipyFsolveSolver(
        tolerance=1e-8, max_iterations=1000, verbose=True, use_fprime=True
    )

    masses = [1, 1]
    g = 9.81
    stiffnesses = [1, 1]
    dampings = [0.5, 0.5]
    forcings = [0.5, 0]
    omega = 1.15
    phases = [0, 0]
    mu = 100000  # always stick

    prox_parameter = 1

    N_HBM = 35

    # Systems
    dae = FrictionOscillator(
        stiffnesses=stiffnesses,
        dampings=dampings,
        masses=masses,
        g=g,
        mu=mu,
        forcing_amplitudes=forcings,
        forcing_phases=phases,
        prox_parameter=prox_parameter,
    )

    print(dae.lam_crit)

    fourier = Fourier(N_HBM=N_HBM, L_DFT=1000, n_dof=dae.n_dof, real_formulation=True)

    hbm = HBMEquationDAE(
        dae,
        omega,
        fourier=fourier,
        initial_guess=np.random.rand(dae.n_dof * (2 * fourier.N_HBM + 1)),
        stability_method=KoopmanHillDAE(
            fourier, tol=1e-4, autonomous=False, tol_drazin=1e-6
        ),
    )

    solver.solve_equation(hbm, unknown="X")

    _, axs = plt.subplots(2, 2)
    x_time = hbm.x_time()
    fourier = hbm.fourier

    for i in range(2):
        axs[i][0].plot(x_time[i, :], x_time[i + 2, :], "-")
        axs[i][0].set_title(f"Phase plot of x_[{i}]")
        axs[i][0].set_ylabel(f"dx_[{i}]")
        axs[i][0].set_xlabel(f"x_[{i}]")

    floquet_multipliers = hbm.eigenvalues
    axs[0][1].plot(np.real(floquet_multipliers), np.imag(floquet_multipliers), "x")
    axs[0][1].set_title("Floquet multipliers")
    axs[0][1].plot(
        np.cos(fourier.time_samples_normalized),
        np.sin(fourier.time_samples_normalized),
        "k",
    )
    axs[0][1].axis("equal")

    _, axs = plt.subplots(hbm.n_dof, 1)
    x_time = hbm.x_time()
    for i in range(hbm.n_dof):
        axs[i].plot(hbm.fourier.time_samples(hbm.omega), x_time[i, :])

    # fig, ax = plt.subplots(1, 1)
    # sys = EquationSystem(
    #     equations=[hbm], unknowns=["X"], equation_determining_stability=hbm
    # )

    # for branch_point in pseudo_arclength_continuator(
    #     initial_system=sys,
    #     solver=solver,
    #     continuation_parameter="omega",
    #     stepsize=0.01,
    #     stepsize_range=[0.001, 0.1],
    #     num_steps=2000,
    #     verbose=True,
    # ):
    #     if branch_point.stable:
    #         color = "r"
    #     else:
    #         color = "b"
    #     axs[idx].plot(
    #         branch_point.omega,
    #         np.max(np.abs(branch_point.equations[0].x_time()[0, :])),
    #         f"{color}.",
    #     )

    #     if branch_point.omega > 4:
    #         break

    # axs[0].set_title("Pendulum ODE FRC")
    # axs[0].set_xlabel("omega")
    # axs[0].set_ylabel("phi max")
    # axs[1].set_title("Pendulum DAE FRC")
    # axs[1].set_xlabel("omega")
    # axs[1].set_ylabel("y max")


if __name__ == "__main__":
    plot_frc()
    plt.show()
