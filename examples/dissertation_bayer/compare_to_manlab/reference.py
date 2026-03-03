"""Create, export and import a reference solution for the MANLAB comparison examples"""

import numpy as np
import matplotlib.pyplot as plt
import csv

from skhippr.solvers.newton import NewtonSolver
from skhippr.Fourier import Fourier

from skhippr.odes.nonautonomous import Duffing
from skhippr.cycles.hbm import HBMEquation
from skhippr.cycles.shooting import ShootingBVP

from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic

from skhippr.visualization.cycles import *


def init_csv(fourier: Fourier, filename: str):

    errors = ["param", "arclength", "init", "pre", "FMs"]
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

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(errors + FM_labels + X_labels)


def to_csv(filename, hbm: HBMEquation, param, arclength, solver, **kwargs_odesolver):
    _, FMs = hbm.determine_stability(update=True)
    X = hbm.X
    errors, _ = determine_ode_accuracy(hbm, solver, visualize=False, **kwargs_odesolver)
    row = np.hstack(
        (np.atleast_1d(param), np.atleast_1d(arclength), np.atleast_1d(errors), FMs, X),
        dtype=complex,
    )
    with open(filename, "a", newline="") as f:
        writer = csv.writer(
            f,
            delimiter=";",
        )
        writer.writerow([str(val)[1:-1] for val in row])


def test_csv():
    filename = "examples/dissertation_bayer/compare_to_manlab/test_csv.csv"
    fourier = Fourier(N_HBM=3, n_dof=2, L_DFT=20)
    init_csv(fourier, filename)


def determine_ode_accuracy(
    hbm: HBMEquation, solver: NewtonSolver, visualize=False, **kwargs_odesolver
):
    """Determine how accurate a HBM solution is by comparing it to a close-by shooting solution. Returns a 3-tuple of the errors and a 3-tuple of strings describing the type of error."""

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
    error_eig_pre = np.min(np.abs(FMs_ref[0] - FMs_pre))

    solver.solve_equation(shoot, "x")
    x_ode_post_solve = shoot.x_time(t_eval=ts)
    error_post = np.max(np.linalg.norm(x_time - x_ode_post_solve, ord=2, axis=0))

    _, eigenvalues = shoot.determine_stability(update=False)
    FMs_post = eigenvalues + 1
    error_eig_post = np.min(np.abs(FMs_ref[0] - FMs_post))

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
    fourier = Fourier(N_HBM=4, L_DFT=1024, n_dof=ode.n_dof)

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
    init_csv(hbm.fourier, filename)
    to_csv(filename, hbm, hbm.omega, 3, solver, atol=1e-14, rtol=1e-14)
    to_csv(filename, hbm, hbm.omega, 3, solver, atol=1e-14, rtol=1e-14)


if __name__ == "__main__":
    # test_csv()
    main()
    plt.show()
