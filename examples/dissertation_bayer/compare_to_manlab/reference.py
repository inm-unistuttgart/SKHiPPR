"""Create, export and import a reference solution for the MANLAB comparison examples"""

import numpy as np
import matplotlib.pyplot as plt
import csv

from skhippr.solvers.newton import NewtonSolver, EquationSystem
from skhippr.solvers.continuation import pseudo_arclength_continuator
from skhippr.Fourier import Fourier

from skhippr.odes.nonautonomous import Duffing
from skhippr.cycles.hbm import HBMEquation, HBMSystem
from skhippr.cycles.shooting import ShootingBVP

from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic

from skhippr.visualization.cycles import *


def create_Duffing_reference():
    solver = NewtonSolver(tolerance=1e-13, verbose=False)
    ode = Duffing(t=0, x=0, omega=0.1, alpha=1, beta=0.1, F=0.5, delta=0.02)
    fourier = Fourier(N_HBM=20, L_DFT=1024, n_dof=ode.n_dof)

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
        filename=f"examples/dissertation_bayer/compare_to_manlab/Duffing_alpha_{ode.alpha}_beta_{ode.beta}_F_{ode.F}_delta_{ode.delta}.csv",
        initial_system=hbm,
        solver=solver,
        stepsize=0.1,
        stepsize_range=(0.001, 3),
        initial_direction=1,
        continuation_parameter="omega",
        verbose=True,
        num_steps=5,
        atol=1e-14,
        rtol=1e-14,
    ):
        if bp.omega > 0.2:
            break


def iterate_reference_solution(
    filename,
    initial_system,
    solver,
    stepsize,
    stepsize_range,
    initial_direction,
    continuation_parameter,
    verbose,
    num_steps,
    FM_error_measure=None,
    **kwargs_odesolver,
):

    with open(filename, "w", newline="") as file:
        writer = csv.writer(file, delimiter=";")

        init_csv(initial_system.equations[0].fourier, writer, continuation_parameter)
        arclength = 0
        X_prev = initial_system.X

        for bp in pseudo_arclength_continuator(
            initial_system,
            solver,
            stepsize,
            stepsize_range,
            initial_direction,
            continuation_parameter,
            verbose,
            num_steps,
        ):
            arclength = arclength + np.linalg.norm(X_prev - bp.X)
            X_prev = bp.X

            to_csv(
                writer,
                bp.equations[0],
                continuation_parameter,
                arclength,
                solver,
                FM_error_measure=FM_error_measure,
                **kwargs_odesolver,
            )

            yield bp


def init_csv(fourier: Fourier, writer, name_param: str):

    errors = [
        name_param,
        "arclength",
        "shoot error init",
        "shoot error max",
        "shoot error FMs",
    ]
    X_labels = [f"X 0, {l}" for l in range(fourier.n_dof)]

    if fourier.real_formulation:
        for sc in ["c", "s"]:
            for k in range(1, fourier.N_HBM + 1):
                X_labels = X_labels + [f"X{sc} {k}, {l}" for l in range(fourier.n_dof)]

    else:
        for k in range(1, fourier.N_HBM + 1):
            X_labels = X_labels + [f"X {k}, {l}" for l in range(fourier.n_dof)]
            X_labels = [f"X {-k}, {l}" for l in range(fourier.n_dof)] + X_labels

    FM_labels = [f"FM {k}" for k in range(fourier.n_dof)]

    writer.writerow(errors + FM_labels + X_labels)


def to_csv(
    writer,
    hbm: HBMEquation,
    name_param,
    arclength,
    solver,
    FM_error_measure=None,
    **kwargs_odesolver,
):
    _, FMs = hbm.determine_stability(update=True)
    param = getattr(hbm, name_param)
    X = hbm.X
    errors, _ = determine_ode_accuracy(
        hbm,
        solver,
        visualize=False,
        FM_error_measure=FM_error_measure,
        **kwargs_odesolver,
    )
    row = np.hstack(
        (np.atleast_1d(param), np.atleast_1d(arclength), np.atleast_1d(errors), FMs, X),
        dtype=complex,
    )

    writer.writerow([str(val)[1:-1] for val in row])


def test_csv():
    filename = "examples/dissertation_bayer/compare_to_manlab/test_csv.csv"
    fourier = Fourier(N_HBM=3, n_dof=2, L_DFT=20)
    init_csv(fourier, filename)


def determine_ode_accuracy(
    hbm: HBMEquation,
    solver: NewtonSolver,
    visualize=False,
    FM_error_measure=None,
    **kwargs_odesolver,
):
    """Determine how accurate a HBM solution is by comparing it to a close-by shooting solution. Returns a 3-tuple of the errors and a 3-tuple of strings describing the type of error."""
    if FM_error_measure is None:
        FM_error_measure = lambda FMs, FMs_ref: np.min(np.abs(FMs_ref[0] - FMs))
    x_time = hbm.x_time()
    ts = hbm.fourier.time_samples(omega=hbm.omega)
    T = 2 * np.pi / hbm.omega
    ts_incl = np.append(ts, T)

    if visualize:
        ax_FM = plot_floquet_multipliers(
            hbm, marker="*", color="r", label="hbm"
        )  # they are later overwritten

    FMs_ref = hbm.eigenvalues

    hbm.ode.x = x_time[:, 0]
    shoot = ShootingBVP(hbm.ode, T=2 * np.pi / hbm.omega, **kwargs_odesolver)

    x_ode_pre_solve = shoot.x_time(t_eval=ts_incl)
    error_init = np.linalg.norm(x_ode_pre_solve[:, 0] - x_ode_pre_solve[:, -1])
    x_ode_pre_solve = x_ode_pre_solve[:, :-1]
    error_pre = np.max(np.linalg.norm(x_time - x_ode_pre_solve, ord=2, axis=0))

    _, eigenvalues = shoot.determine_stability(update=True)
    FMs_pre = eigenvalues + 1
    error_eig_pre = FM_error_measure(FMs_pre, FMs_ref)

    solver.solve_equation(shoot, "x")
    x_ode_post_solve = shoot.x_time(t_eval=ts)
    error_post = np.max(np.linalg.norm(x_time - x_ode_post_solve, ord=2, axis=0))

    _, eigenvalues = shoot.determine_stability(update=False)
    FMs_post = eigenvalues + 1
    error_eig_post = FM_error_measure(FMs_post, FMs_ref)

    hbm.eigenvalues = FMs_ref

    if visualize:
        ax_period = plot_period(hbm, label="hbm")
        ax_period.set_title("solutions")

        _, ax_error = plt.subplots(1, 1)

        labels = ["pre solve", "post solve"]
        markers = ["x", "+"]

        for k, (x_ode, FMs) in enumerate(
            zip([x_ode_pre_solve, x_ode_post_solve], [FMs_pre, FMs_post])
        ):
            ax_period.plot(ts, x_ode[0, :], label=labels[k])
            ax_error.semilogy(ts, np.abs(x_ode[0, :] - x_time[0, :]), label=labels[k])
            ax_FM.plot(np.real(FMs), np.imag(FMs), markers[k], label=labels[k])

        ax_period.legend()
        ax_error.legend()
        ax_FM.legend()

    error_labels = ("init", "pre", "FMs")
    return (error_init, error_pre, error_eig_pre), error_labels


def main():
    solver = NewtonSolver(tolerance=1e-13, verbose=True)
    ode = Duffing(t=0, x=0, omega=1, alpha=1, beta=0.1, F=0.5, delta=0.02)
    fourier = Fourier(N_HBM=20, L_DFT=1024, n_dof=ode.n_dof)

    initial_guess = np.zeros((2 * fourier.N_HBM + 1) * ode.n_dof)
    hbm = HBMEquation(
        ode=ode,
        omega=ode.omega,
        fourier=fourier,
        initial_guess=initial_guess,
        period_k=1,
        stability_method=KoopmanHillSubharmonic(fourier),
    )
    solver.solve_equation(hbm, "X")

    # errors = determine_ode_accuracy(hbm, solver, visualize=True, atol=1e-14, rtol=1e-14)
    # for err, cat in zip(errors, ["init", "pre", "post", "eig pre", "eig post"]):
    #     print(f"Error {cat}: {err}")

    filename = "examples/dissertation_bayer/compare_to_manlab/export.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        init_csv(hbm.fourier, writer, name_param="omega")
        to_csv(writer, hbm, "omega", 3, solver, atol=1e-14, rtol=1e-14)
        to_csv(writer, hbm, "omega", 3, solver, atol=1e-14, rtol=1e-14)

    return hbm, solver


if __name__ == "__main__":
    create_Duffing_reference()
    plt.show()
