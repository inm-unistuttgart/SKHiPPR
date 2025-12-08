"""Illustrate HBM applied to DAEs and ODEs, with the example of the simple pendulum."""

import numpy as np
import matplotlib.pyplot as plt

from skhippr.odes.daes import PendulumDAE, PendulumODE
from skhippr.Fourier import Fourier
from skhippr.solvers.newton import NewtonSolver
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.cycles.hbm import HBMEquation, HBMEquationDAE


def main():

    solver = NewtonSolver(tolerance=1e-8, max_iterations=50, verbose=True)

    m = 0.8
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
    hbm_ode = HBMEquation(
        ode,
        omega,
        fourier=Fourier(N_HBM=N_HBM, L_DFT=1000, n_dof=ode.n_dof),
        initial_guess=np.zeros(ode.n_dof * (2 * N_HBM + 1)),
    )
    sys_ode = EquationSystem(equations=[hbm_ode], unknowns="X")
    solver.solve(sys_ode)
    print("Solved ODE. \n")

    x_ode = sys_ode.equations[0].x_time()
    phi = x_ode[0, :]
    phi_dot = x_ode[1, :]
    x_dae = np.vstack(
        [
            -l * np.cos(phi),
            l * np.sin(phi),
            l * np.sin(phi) * phi_dot,
            l * np.cos(phi) * phi_dot,
            np.zeros_like(phi),
        ]
    )

    fourier = Fourier(N_HBM=N_HBM, L_DFT=1000, n_dof=dae.n_dof)

    hbm_dae = HBMEquationDAE(
        dae,
        omega,
        fourier=fourier,
        initial_guess=fourier.DFT(x_dae),
    )
    sys_dae = EquationSystem(equations=[hbm_dae], unknowns="X")

    solver.solve(sys_dae)


if __name__ == "__main__":
    main()
    # plt.show()
