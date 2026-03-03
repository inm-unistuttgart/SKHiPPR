"""Create, export and import a reference solution for the MANLAB comparison examples"""

import numpy as np
import matplotlib.pyplot as plt

from skhippr.solvers.newton import NewtonSolver
from skhippr.Fourier import Fourier

from skhippr.odes.nonautonomous import Duffing
from skhippr.cycles.hbm import HBMEquation
from skhippr.cycles.shooting import ShootingSystem


def export_branch_with_accuracy(branch, filename="export.csv"):
    """Export all points on a branch with accuracy measure"""


def determine_ode_accuracy(hbm: HBMEquation, solver: NewtonSolver, **kwargs_odesolver):
    """Determine how accurate a HBM solution is by comparing it to a close-by shooting solution."""

    x_time = hbm.x_time()
    hbm.ode.x = x_time[:, 0]

    shoot = ShootingSystem(hbm.ode, T=2 * np.pi / hbm.omega, **kwargs_odesolver)

    x_ode_pre_solve = shoot.x_time(t_eval=hbm.fourier.time_samples(omega=hbm.omega))
    solver.solve(shoot)
    x_ode_post_solve = shoot.x_time(t_eval=hbm.fourier.time_samples(omega=hbm.omega))


def main():
    solver = NewtonSolver(tolerance=1e-8, verbose=True)
    ode = Duffing(t=0, x=0, omega=1, alpha=1, beta=0.1, F=0.5, delta=0.02)
    fourier = Fourier(N_HBM=20, L_DFT=1024, n_dof=ode.n_dof)

    initial_guess = np.zeros((2 * fourier.N_HBM + 1) * ode.n_dof)
    hbm = HBMEquation(
        ode=ode,
        omega=ode.omega,
        fourier=fourier,
        initial_guess=initial_guess,
        period_k=1,
        stability_method=None,
    )
    solver.solve_equation(hbm, "X")


if __name__ == "__main__":
    main()
    plt.show()
