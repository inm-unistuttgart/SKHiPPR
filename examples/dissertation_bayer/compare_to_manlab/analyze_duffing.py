import numpy as np
import matplotlib.pyplot as plt
import csv
import tikzplotlib
from tqdm import tqdm

from skhippr.solvers.newton import NewtonSolver
from skhippr.Fourier import Fourier
from skhippr.cycles.hbm import HBMSystem

from skhippr.odes.nonautonomous import Duffing

from skhippr.visualization.continuation import plot_continuation

from analyze_and_plot import *


from skhippr.stability.KoopmanHillProjection import (
    KoopmanHillSubharmonic,
    KoopmanHillProjection,
)
from skhippr.stability.ClassicalHill import ClassicalHill
from skhippr.stability.SinglePass import SinglePassRK4

from create_reference import (
    iterate_reference_solution,
)
from import_reference import (
    import_reference,
    plot_reference_data,
    iterate_from_reference,
    change_N_HBM,
)

from comptime_measurements import measure_time_to_hill, measure_stability_method


def main():
    ode, label = init_duffing(
        exponent=5, alpha=1, beta=1, F=3, delta=0.25, omega_init=5
    )

    # N_HBM = 120
    # atol = 1e-14
    # rtol = 1e-14

    N_HBM = 20
    atol = 1e-07
    rtol = 1e-07

    # create_Duffing_reference(
    #     ode, label, num_steps=10, N_HBM=N_HBM, atol=atol, rtol=rtol, omega_max=0
    # )

    data = import_reference(
        filename=get_filename(label, N_HBM=N_HBM, atol=atol, rtol=rtol), ode=ode
    )

    plot_reference_data(data, get_filename(label, N_HBM=N_HBM, atol=atol, rtol=rtol))

    solver = NewtonSolver(verbose=False)

    N_examine = 10
    L_examine = 1024
    fourier = Fourier(N_examine, L_examine, ode.n_dof, real_formulation=True)

    stability_methods = {
        "dir": KoopmanHillProjection(fourier),
        "subh": KoopmanHillSubharmonic(fourier),
        "imag": ClassicalHill(fourier, "imaginary"),
        "RK4": SinglePassRK4(fourier),
    }

    hbms = []
    comptimes = dict()
    FM_errors = {method: dict() for method in stability_methods.keys()}
    FM_times = {method: dict() for method in stability_methods.keys()}

    for k, hbm in tqdm(
        enumerate(
            iterate_from_reference(
                ode,
                data,
                N_examine,
                L_examine,
                True,
                stability_method=KoopmanHillSubharmonic,
            )
        ),
        total=len(data["arclength"]),
    ):

        solver.solve(hbm)

        hbms.append(hbm)

        if k >= 31:
            pass

        if k >= 1 and hbm.solved:
            X_ext_prev = np.hstack((data["X"][k - 1, :], data["param"][k - 1]))
            X_ext_ref = np.hstack((data["X"][k, :], data["param"][k]))

            hill_matrices, times, other = measure_time_to_hill(
                hbm, X_ext_ref, X_ext_prev, solver
            )
            for label, time in times.items():
                try:
                    comptimes[label][k] = time
                except KeyError:
                    comptimes[label] = np.zeros(len(data["param"]))
                    comptimes[label][k] = time

            for name_method, method in stability_methods.items():
                for label, mat in hill_matrices.items():
                    time, error, _ = measure_stability_method(
                        mat, method, data["FMs"], FM_error_measure
                    )
                    try:
                        FM_errors[name_method][label][k] = error
                    except KeyError:
                        FM_errors[name_method][label] = np.nan(len(data["param"]))
                        FM_times[name_method][label] = np.nan(len(data["param"]))
                        FM_errors[name_method][label][k] = error

                    FM_times[name_method][label][k] = time
        else:
            # Newton solver failed
            for label in comptimes.keys():
                comptimes[label][k] = np.nan

    # Plot computation times
    _, ax = plt.subplots(1, 1)
    for label, times in comptimes.items():
        ax.semilogy(data["arclength"], times, label=label)

    for name_method in FM_times.keys():
        for label, times in FM_times[name_method].items():
            ax.semilogy(data["arclength"], times, label=f"FM {name_method}, {label}")

    ax.set_title("times over arclength")
    ax.set_xlabel("arclength")
    ax.set_ylabel("comp. time")
    ax.legend()

    # Plot error
    _, ax = plt.subplots(1, 1)

    for name_method in FM_errors.keys():
        for label, errors in FM_errors[name_method].items():
            ax.semilogy(data["arclength"], errors, label=f"FM {name_method}, {label}")

    ax.set_title("errors over arclength")
    ax.set_xlabel("error")
    ax.set_ylabel("comp. time")
    ax.legend()

    plot_continuation(hbms, plot_fun)


def init_duffing(exponent=3, alpha=1, beta=0.1, F=0.5, delta=0.02, omega_init=0.1):
    ode = Duffing(
        t=0,
        x=0,
        omega=omega_init,
        alpha=alpha,
        beta=beta,
        F=F,
        delta=delta,
        exponent=exponent,
    )
    label = f"Duffing_{ode.exponent}_alpha_{ode.alpha}_beta_{ode.beta}_F_{ode.F}_delta_{ode.delta}"
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

    if ode.omega < 1:
        initial_direction = 1
    else:
        initial_direction = -1

    branch = []
    for bp in iterate_reference_solution(
        filename=get_filename(label, N_HBM=N_HBM, **kwargs_odesolver),
        initial_system=hbm,
        solver=solver,
        stepsize=0.1,
        stepsize_range=(0.0001, 0.1),
        initial_direction=initial_direction,
        continuation_parameter="omega",
        verbose=True,
        num_steps=num_steps,
        **kwargs_odesolver,
    ):
        branch.append(bp)
        if initial_direction * bp.omega > initial_direction * omega_max:
            break

    ax = plot_continuation(branch, plot_fun)
    ax.set_ylabel("max_t x_0(t)")
    tikzplotlib.save(f"{get_filename(label, N_HBM=N_HBM, **kwargs_odesolver)}.tikz")


def plot_fun(bp):
    if np.linalg.norm(bp.residual_function(update=True)) > 1e-4:
        return np.nan
    return np.max(np.abs(bp.equations[0].x_time()[0, :]))


def FM_error_measure(FMs, FMs_ref):
    return np.min(np.abs(FMs_ref[0] - FMs))


def step_1(
    exponent=5, alpha=1, beta=1, F=0.5, delta=0.02, Nmax=120, atol=1e-13, rtol=1e-13
):
    ode, label = init_duffing(exponent, alpha, beta, F, delta)
    filename = get_filename(label, N_HBM=Nmax, atol=atol, rtol=rtol)

    error_stats = compute_step_1(
        ode,
        filename=filename,
        Ns_HBM=[5, 10, 15],
        L_DFT=1024,
        stability_method_generator=KoopmanHillSubharmonic,
        solver=NewtonSolver(tolerance=1e-13, verbose=False),
    )

    ax = plot_step_1(error_stats)
    return ax, error_stats, filename


def iterate_step_1(
    exponent=5,
    alpha=1,
    beta=1,
    F=0.5,
    delta=0.02,
    Nmax=120,
    atol=1e-13,
    rtol=1e-13,
    labels_stabmethod=(KoopmanHillSubharmonic,),
):
    for label_stab in labels_stabmethod:
        match label_stab:
            case "subh":
                stability_method_generator = KoopmanHillSubharmonic
            case "dir":
                stability_method_generator = KoopmanHillProjection
            case "imag":
                stability_method_generator = lambda fourier: ClassicalHill(
                    fourier, "imaginary"
                )
            case "RK4":
                stability_method_generator = SinglePassRK4
            case _:
                raise ValueError(f"Unknown stability method {label_stab}")

        ax, error_stats, filename = step_1(
            exponent=exponent,
            alpha=alpha,
            beta=beta,
            F=F,
            delta=delta,
            Nmax=Nmax,
            atol=atol,
            rtol=rtol,
        )
        ax.set_title(f"step 1 FM errors for {label_stab}")
        tikzplotlib.save(f"{filename}_stab_{label_stab}.tikz")


if __name__ == "__main__":
    iterate_step_1(
        exponent=5,
        alpha=1,
        beta=1,
        F=3,
        delta=0.25,
        Nmax=40,
        atol=1e-12,
        rtol=1e-12,
        labels_stabmethod=["subh", "dir", "imag", "RK4"],
    )
    plt.show()
