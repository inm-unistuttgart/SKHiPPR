import numpy as np
import matplotlib.pyplot as plt

from skhippr.odes.nonautonomous import Duffing
from skhippr.solvers.newton import NewtonSolver
from skhippr.Fourier import Fourier
from skhippr.cycles.hbm import HBMSystem
from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic

from skhippr.visualization.cycles import (
    animate_floquet_exponents,
    animate_floquet_multipliers,
)
from skhippr.visualization.continuation import plot_continuation
from skhippr.visualization.data_export import save_animation, save_png
from generate_stability_data import generate_stability_data

""" Parameters """
# PARAMS = {
#     "alpha": 0.5,
#     "beta": 1,
#     "F": 5,
#     "delta": 0.1,
# }

PARAMS = {
    "alpha": 1.13,
    "beta": -0.2,
    "F": 0.15,
    "delta": 0.1,
}
N_HBM = 5
L_DFT = 2**8
SOLVER_TOL = 1e-10
NUM_STEPS = 400
STEPSIZE = 0.03
OMEGA_START = 0.01


def main():
    """Create reference data for the duffing oscillator, which can be parsed by the csv parser."""
    ode = Duffing(t=0, x=0, omega=OMEGA_START, **PARAMS)
    filename = f"examples/HillML/Duffing_alpha_{ode.alpha}_beta_{ode.beta}_F_{ode.F}_delta_{ode.delta}_N_{N_HBM}_L_{L_DFT}_solvertol_{SOLVER_TOL}.csv"
    fourier = Fourier(N_HBM=N_HBM, L_DFT=L_DFT, n_dof=ode.n_dof)
    initial_guess = np.zeros((2 * fourier.N_HBM + 1) * ode.n_dof)
    hbm = HBMSystem(
        ode=ode,
        omega=ode.omega,
        fourier=fourier,
        initial_guess=initial_guess,
        period_k=1,
        stability_method=KoopmanHillSubharmonic(fourier),
    )

    frc = list(
        generate_stability_data(
            filename=filename,
            initial_system=hbm,
            solver=NewtonSolver(tolerance=1e-8, verbose=False),
            stepsize=STEPSIZE,
            stepsize_range=(STEPSIZE, STEPSIZE),
            initial_direction=1,
            continuation_parameter="omega",
            verbose=True,
            num_steps=NUM_STEPS,
        )
    )

    ax = plot_continuation(
        frc,
        plot_fun,
        xlabel="omega",
        ylabel="max|x(t)|",
        title="Duffing reference solution",
    )
    anims = []
    anims.append(animate_floquet_exponents(frc)[1])
    anims.append(animate_floquet_multipliers(frc)[1])

    return anims, ax, filename


def plot_fun(bp):
    return bp.omega, np.max(np.abs(bp.equations[0].x_time()[0, :]))


if __name__ == "__main__":
    anims, ax, filename = main()
    for anim, label in zip(anims, ["FE", "FM"]):
        save_animation(anim, filename.replace(".csv", f"_{label}.gif"))
    save_png(ax, filename.replace(".csv", "_FRC.png"))
    plt.show()
