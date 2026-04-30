import numpy as np
import matplotlib.pyplot as plt

from skhippr.solvers.newton import ScipyRootSolver
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.solvers.continuation import pseudo_arclength_continuator
from skhippr.Fourier import Fourier

from friction_hbm import solve_hbm


def plot_frcs(frcs):
    _, ax = plt.subplots(1, 1)
    for frc in enumerate(frcs):
        try:
            smoothing = frc[0].smoothing
        except AttributeError:
            smoothing = "nonsmooth"
        ax.plot(
            [point.omega for point in frc],
            [np.max(point.equations[0].x_time()[0, :]) for point in frc],
            label=f"smoothing = {smoothing}",
        )
        ax.set_title(f"Friction oscillator FRC smoothing = {smoothing}")
        ax.set_xlabel("omega")
        ax.set_ylabel("|x[0]| max")


def compute_frc(
    name_case="Schuetz2", smoothing=50, solver=None, N_HBM=60, L_DFT=2000, num_steps=10
):

    if solver is None:
        solver = ScipyRootSolver(
            tolerance=1e-7,
            max_iterations=100,
            verbose=True,
            use_fprime=True,
            method="lm",
        )

    fourier = Fourier(N_HBM=N_HBM, L_DFT=L_DFT, n_dof=5, real_formulation=True)

    hbm = None

    frcs = []

    for k in range(2):

        solver.verbose = True

        hbm = solve_hbm(
            name_case=name_case,
            smoothing=smoothing,
            fourier=fourier,
            hbm_ref=hbm,
            solver=solver,
        )

        sys = EquationSystem(
            equations=[hbm], unknowns=["X"], equation_determining_stability=hbm
        )

        solver.verbose = False

        frcs[k] = []

        for branch_point in pseudo_arclength_continuator(
            initial_system=sys,
            solver=solver,
            continuation_parameter="omega",
            stepsize=0.1,
            stepsize_range=[0.001, 2],
            num_steps=num_steps,
            verbose=True,
        ):
            frcs[k].append(branch_point)

            if branch_point.omega > 1:
                break

    return frcs


if __name__ == "__main__":
    frcs = compute_frc()
    plot_frcs(frcs)
    plt.show()
