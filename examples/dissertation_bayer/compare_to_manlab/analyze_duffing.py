import numpy as np
import matplotlib.pyplot as plt
import csv

from skhippr.solvers.newton import NewtonSolver
from skhippr.Fourier import Fourier
from skhippr.cycles.hbm import HBMSystem

from skhippr.odes.nonautonomous import Duffing


from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic

from examples.dissertation_bayer.compare_to_manlab.create_reference import (
    iterate_reference_solution,
)
from examples.dissertation_bayer.compare_to_manlab.import_reference import (
    import_reference,
    plot_reference_data,
)


def main():
    ode, label = init(alpha=1, beta=0.1, F=0.5, delta=0.02, omega_init=0.01)

    N_HBM = 20
    atol = 1e-13
    rtol = 1e-13

    # create_Duffing_reference(
    #     ode, label, num_steps=10, N_HBM=N_HBM, atol=atol, rtol=rtol
    # )

    data = import_reference(
        filename=get_filename(label, N_HBM=N_HBM, atol=atol, rtol=rtol), ode=ode
    )

    plot_reference_data(data)


def init(alpha=1, beta=0.1, F=0.5, delta=0.02, omega_init=0.1):
    ode = Duffing(t=0, x=0, omega=omega_init, alpha=alpha, beta=beta, F=F, delta=delta)
    label = f"Duffing_alpha_{ode.alpha}_beta_{ode.beta}_F_{ode.F}_delta_{ode.delta}"
    del ode.eigenvalues
    del ode.stability_method
    return ode, label


def get_filename(label, **kwargs):
    N_HBM = kwargs.get("N_HBM", None)
    atol = kwargs.get("atol", None)
    rtol = kwargs.get("rtol", None)

    return f"examples/dissertation_bayer/compare_to_manlab/{label}_N_{N_HBM}_atol_{atol}_rtol_{rtol}.csv"


def create_Duffing_reference(
    ode, label, omega_max=10, num_steps=5, N_HBM=20, **kwargs_odesolver
):
    solver = NewtonSolver(tolerance=1e-13, verbose=False)
    fourier = Fourier(N_HBM=N_HBM, L_DFT=1024, n_dof=ode.n_dof)

    initial_guess = np.zeros((2 * fourier.N_HBM + 1) * ode.n_dof)
    hbm = HBMSystem(
        ode=ode,
        omega=ode.omega,
        fourier=fourier,
        initial_guess=initial_guess,
        period_k=1,
        stability_method=KoopmanHillSubharmonic(fourier),
    )

    for bp in iterate_reference_solution(
        filename=get_filename(label, N_HBM=N_HBM, **kwargs_odesolver),
        initial_system=hbm,
        solver=solver,
        stepsize=0.0001,
        stepsize_range=(0.0001, 3),
        initial_direction=1,
        continuation_parameter="omega",
        verbose=True,
        num_steps=num_steps,
        **kwargs_odesolver,
    ):
        if bp.omega > omega_max:
            break


if __name__ == "__main__":
    main()
    plt.show()
