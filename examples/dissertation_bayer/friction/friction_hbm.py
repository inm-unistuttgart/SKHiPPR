"""FRC of frictional oscillator. See Schütz (2025), Bachelor's thesis, and Legrand2023."""

from copy import copy
from typing import Any
from collections.abc import Generator

import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib


from skhippr.odes.daes import FrictionOscillator, SmoothedFrictionOscillator
from skhippr.Fourier import Fourier, round_to_significant_digits
from skhippr.solvers.newton import ScipyFsolveSolver, NewtonSolver, ScipyRootSolver
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.cycles.hbm import HBMEquationDAE
from skhippr.solvers.continuation import pseudo_arclength_continuator

from skhippr.stability.KoopmanHillProjection import (
    KoopmanHillDAE,
)

from friction_init import init_oscillator, get_description
from friction_direct import solve_friction
from drazin import plot_drazin_and_ratio
from plot_friction import (
    plot_hbm_result,
    plot_FM_convergence,
    plot_FM_error,
    plot_hbm_convergence,
)
from export_friction import to_csv, save_result_with_hill_matrix
from floquet import KH_other_N


def main(
    cases=("B",),
    smoothings=(np.inf,),
    Ns_HBM=(30, 40, 50),
    Ns_plot=(),
    L_DFT=2**14,
    max_residual=1e-9,
):
    print(Ns_HBM)
    for name_case in cases:
        for smoothing in smoothings:
            description = get_description(name_case, smoothing, max(Ns_HBM), L_DFT)

            try:
                hbms = []
                for hbm in iterate_over_N(
                    name_case=name_case,
                    smoothing=smoothing,
                    Ns_HBM=Ns_HBM,
                    L_DFT=L_DFT,
                    max_residual=max_residual,
                ):
                    hbms.append(hbm)

                    if hbm.fourier.N_HBM in Ns_plot:
                        description_plot = get_description(
                            name_case, smoothing, hbm.fourier.N_HBM, L_DFT
                        )
                        plot_hbm_result(hbm, description_plot, path=f"plots/")
                    print(
                        "---------------------------------------------------------------------"
                    )
                    print(f"smoothing = {smoothing}:")
                    print(hbm.eigenvalues)
                    print(
                        "---------------------------------------------------------------------"
                    )
                    save_result_with_hill_matrix(hbms[-1], description_plot)

                    KH_other_N(hbm, N_other=10, description=description)
            except MemoryError as ME:
                plt.close("all")
                print(f"Memory overflow: {ME}")
                continue

            if len(Ns_HBM) > 1:
                plot_and_save(hbms, description_plot, tol_drazin=1e-7, path="plots/")
            else:
                to_csv(hbms, f"plots/HBM_results_{description}.csv")

                plot_FM_convergence(
                    hbms, description, path=f"plots/FMs_{description}.tikz"
                )

                _, ax = plt.subplots(1, 1)

                plot_drazin_and_ratio(
                    hbms=hbms,
                    tol_drazin=1e-7,
                    description=description,
                    ratio_limits=[3 / 5, 4 / 5],
                    ax_drazin=ax,
                    ax_ratio=None,
                    path_drazin=None,
                    path_ratio=None,
                )


def solve_hbm(
    name_case,
    smoothing=np.inf,
    fourier: Fourier = None,
    hbm_ref: HBMEquationDAE = None,
    solver=None,
):
    if fourier is None:
        if hbm_ref is None:
            raise ValueError("fourier and hbm_ref may not be both None!")
        else:
            fourier = hbm_ref.fourier

    if not fourier.real_formulation:
        raise ValueError("Only real formulation supported!")

    if solver is None:
        solver = ScipyRootSolver(
            tolerance=1e-8,
            max_iterations=1000,
            verbose=True,
            use_fprime=True,
            method="lm",
        )

    dae = init_oscillator(name_case, smoothing=smoothing)

    # Initial guess for lambda
    if hbm_ref is None:
        initial_guess_lambda = None
    else:
        X = fourier.resize_coefficients(hbm_ref.X)
        X = np.reshape(X, (fourier.n_dof, -1), order="F")
        initial_guess_lambda = X[4, :]

    # Solve with substituted formulation
    equ_lambda = solve_friction(
        oscillator=dae,
        fourier=fourier,
        omega=dae.omega,
        smoothing=smoothing,
        solver=solver,
        initial_guess=initial_guess_lambda,
    )

    X, dX = equ_lambda.FC_dX()
    initial_guess = np.real(np.vstack(((X, dX, equ_lambda.Lambda))).flatten(order="F"))

    # Solve actual problem with substituted warm-start
    hbm = HBMEquationDAE(
        dae,
        dae.omega,
        fourier=fourier,
        initial_guess=initial_guess,
        stability_method=KoopmanHillDAE(
            fourier, tol=1e-4, autonomous=False, tol_drazin=1e-6
        ),
    )

    if solver.verbose:
        print(
            f"Case {name_case}, N = {fourier.N_HBM}, smoothing = {smoothing}: -- Residual before solving: {np.linalg.norm(hbm.residual(update=True), np.inf)}"
        )

    solver.solve_equation(hbm, unknown="X")

    if solver.verbose:
        print(
            f"Case {name_case}, N = {fourier.N_HBM}, smoothing = {smoothing}: -- Residual after solving: {np.linalg.norm(hbm.residual(update=True), np.inf)}"
        )

    return hbm


def iterate_over_N(
    name_case="C",
    smoothing=np.inf,
    Ns_HBM=(30, 40, 50),
    L_DFT=2**14,
    max_residual=1e-9,
) -> Generator[HBMEquationDAE, Any, None]:

    N_max = Ns_HBM[-1]
    description = get_description(name_case, smoothing, N_max, L_DFT)

    hbm_ref = None

    for N_HBM in Ns_HBM:
        # plt.close("all")
        print("--------------------------------------------------------------------")
        print(f"solving N = {N_HBM} for {description}")
        fourier = Fourier(N_HBM, L_DFT, n_dof=5, real_formulation=True)
        hbm = solve_hbm(name_case, smoothing, fourier, hbm_ref)

        if np.linalg.norm(hbm.residual(update=False)) > max_residual:
            print(
                f"Residual of solved solution is above threshold: {np.linalg.norm(hbm_ref.residual(update=False))} > {max_residual}"
            )
            # . Stopping iteration."
            # )
            # break
        else:
            hbm_ref = hbm
            yield hbm


def plot_and_save(hbms, description, tol_drazin=1e-7, path=""):
    print(f"a posteriori analysis {description}")

    # Drazin and Drazin ratio plots
    _, ax_drazin = plt.subplots(1, 1)
    _, ax_drazin_ratio = plt.subplots(1, 1)

    plot_drazin_and_ratio(
        hbms=hbms,
        tol_drazin=tol_drazin,
        description=description,
        ratio_limits=[3 / 5, 4 / 5],
        ax_drazin=ax_drazin,
        ax_ratio=ax_drazin_ratio,
        path_drazin="plots/",
        path_ratio=f"plots/",
    )

    # save the HBM results to a file
    to_csv(hbms, f"{path}HBM_results_{description}.csv")

    # Plot Floquet multipliers
    plot_FM_convergence(hbms, description, path=f"{path}FMs_{description}.tikz")

    plot_FM_error(
        hbms,
        hbms[-1].eigenvalues,
        description,
        path=f"{path}FM_error_{description}.tikz",
    )

    # Plot HBM convergence
    plot_hbm_convergence(
        hbms, description, path=f"{path}HBM_convergence_time_{description}.tikz"
    )

    print(
        "==============================================================================================="
    )


if __name__ == "__main__":
    main(
        cases=["B"],
        smoothings=[50, np.inf],
        Ns_HBM=(30,),
        Ns_plot=(30,),
        L_DFT=2**14,
        max_residual=1e-9,
    )
    plt.show()
