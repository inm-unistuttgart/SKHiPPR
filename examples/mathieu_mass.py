import numpy as np
import matplotlib.pyplot as plt

from skhippr.odes.ltp import MathieuODE, MathieuWithMass, MathieuWithMassInverted
from skhippr.cycles.hbm import HBMEquation, HBMEquationDAE
from skhippr.Fourier import Fourier
from skhippr.solvers.newton import NewtonSolver

from skhippr.visualization.cycles import plot_phase


def main():

    a = 1.0
    b = 0.5
    omega = 1.0
    damping = 0.1
    forcing = 0.2

    solver = NewtonSolver(verbose=True)
    fourier = Fourier(N_HBM=30, L_DFT=1024, n_dof=2, real_formulation=True)
    ode = MathieuODE(
        t=0,
        x=np.zeros((2, fourier.L_DFT)),
        a=a,
        b=b,
        omega=omega,
        damping=damping,
        forcing=forcing,
    )
    hbm = HBMEquation(ode, ode.omega, fourier)

    solver.solve_equation(hbm, "X")
    ax = plot_phase(hbm, label="without mass")

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
    plot_phase(hbm_inv, ax=ax, label="inverted mass", linestyle=":")

    ode_mass = MathieuWithMass(
        t=0,
        x=np.zeros((2, fourier.L_DFT)),
        a=a,
        b=b,
        omega=omega,
        damping=damping,
        forcing=forcing,
    )
    hbm_mass = HBMEquationDAE(ode_mass, ode_mass.omega, fourier, initial_guess=hbm.X)

    solver.solve_equation(hbm_mass, "X")
    plot_phase(hbm_mass, ax=ax, label="with mass", linestyle="--")
    ax.legend()


if __name__ == "__main__":
    main()
    plt.show()
