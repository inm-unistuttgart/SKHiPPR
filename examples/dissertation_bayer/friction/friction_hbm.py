"""FRC of frictional oscillator. See Schütz (2025), Bachelor's thesis, and Legrand2023."""

from copy import copy
from typing import Any
from collections.abc import Generator

import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

from scipy.io import loadmat


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
                        plot_hbm_result(
                            hbm,
                            description_plot,
                            path=f"examples/dissertation_bayer/friction/plots/",
                        )
                        save_result_with_hill_matrix(
                            hbms[-1],
                            description_plot,
                            "examples/dissertation_bayer/friction/data/",
                        )

                    print(
                        "---------------------------------------------------------------------"
                    )
                    print(f"smoothing = {smoothing}:")
                    print(hbm.eigenvalues)
                    print(
                        "---------------------------------------------------------------------"
                    )

                    # KH_other_N(hbm, N_other=10, description=description)
                    hbm.hill_matrix(update=True)
            except MemoryError as ME:
                plt.close("all")
                print(f"Memory overflow: {ME}")
                continue

            if False:  # len(Ns_HBM) > 1:
                plot_and_save(
                    hbms,
                    description_plot,
                    tol_drazin=1e-7,
                    path="examples/dissertation_bayer/friction/plots/",
                )
            else:
                # to_csv(
                #     hbms,
                #     f"examples/dissertation_bayer/friction/data/HBM_results_{description}.csv",
                # )

                plot_FM_convergence(
                    hbms,
                    description,
                    path=f"examples/dissertation_bayer/friction/plots/FMs_{description}.tikz",
                )

                _, ax = plt.subplots(1, 1)

                # plot_drazin_and_ratio(
                #     hbms=hbms,
                #     tol_drazin=1e-7,
                #     description=description,
                #     ratio_limits=[3 / 5, 4 / 5],
                #     ax_drazin=ax,
                #     ax_ratio=None,
                #     path_drazin=f"examples/dissertation_bayer/friction/plots/",
                #     path_ratio=None,
                # )


def fundamat_over_time(
    hbm: HBMEquationDAE, L=101, rank_tol=1e-8, description=None, path="", path_ref=""
):
    """Determine the fundamental solution matrix over time by evaluating Koopman-Hill not only aht the period."""
    ts = np.linspace(0, 1, L, endpoint=True)
    Phis = np.zeros((hbm.fourier.n_dof, hbm.fourier.n_dof, len(ts)), dtype=complex)
    hbm.hill_matrix(update=True)
    for k, t in enumerate(ts):
        print(f"{k}/{len(ts)}: t/T = {t:.3f}")
        Phis[:, :, k] = hbm.stability_method.fundamental_matrix(
            hbm=hbm, t_over_period=t, update=False
        )

    if description is not None:
        ax = np.empty((hbm.fourier.n_dof, hbm.fourier.n_dof), dtype=object)
        # Plot all entries of fundamental solution matrix over time
        # _, ax = plt.subplots(hbm.fourier.n_dof, hbm.fourier.n_dof)
        for i in range(hbm.fourier.n_dof):
            for j in range(hbm.fourier.n_dof):
                _, ax[i, j] = plt.subplots(1, 1)
                ax[i, j].plot(ts, Phis[i, j, :], label="real")
                # ax[i, j].plot(ts, np.imag(Phis[i, j, :]), "--", label="imag")
                ax[i, j].set_ylabel(f"Phi[{i},{j}]")
                ax[i, j].set_xlabel("t/T")

        if path_ref != "":
            dict_vars = loadmat(path_ref)
            Phis_ref = dict_vars["Phis"]
            Phis_dae = dict_vars["Phis_dae"]
            ts_ref = dict_vars["ts"].flatten()
            ts_ref /= (2 * np.pi) / hbm.omega
            for i in range(hbm.fourier.n_dof - 1):
                for j in range(hbm.fourier.n_dof - 1):
                    ax[i, j].plot(
                        ts_ref,
                        Phis_ref[i, j, :],
                        "--",
                        label="ref",
                    )
            for i in range(hbm.fourier.n_dof):
                for j in range(hbm.fourier.n_dof):
                    ax[i, j].plot(
                        ts_ref,
                        Phis_dae[i, j, :],
                        ":",
                        label="ref DAE",
                    )

        ax[0, 0].legend()
        ax[0, 0].set_title(f"fundamental matrix {description}")

        # Plot the rank of the fundamental solution matrix over time
        ranks = np.zeros(len(ts), dtype=int)
        _, ax_rank = plt.subplots(1, 1)

        for k in range(len(ts)):
            Phi = Phis[:, :, k]
            ranks[k] = np.linalg.matrix_rank(Phi, tol=rank_tol)

        ax_rank.plot(ts, ranks)
        ax_rank.set_title("Rank of fundamental solution matrix over time")
        ax_rank.set_xlabel("t/T")

        if path != "":
            for i in range(hbm.fourier.n_dof):
                for j in range(hbm.fourier.n_dof):
                    plt.sca(ax[i, j])
                    tikzplotlib.save(
                        f"{path}fundamental_solution_{description}_{i}{j}.tikz"
                    )

    return Phis, ts


def plot_J_over_time(hbm: HBMEquationDAE, description=None, path=""):
    """Plot the Jacobian of the system over time."""
    Js = hbm.ode_samples()
    ts = hbm.fourier.time_samples(omega=2 * np.pi)

    if description is not None:
        # Plot all entries of Jacobian over time
        ax = np.empty((hbm.fourier.n_dof, hbm.fourier.n_dof), dtype=object)
        # _, ax = plt.subplots(hbm.fourier.n_dof, hbm.fourier.n_dof)
        for i in range(hbm.fourier.n_dof):
            for j in range(hbm.fourier.n_dof):
                _, ax[i, j] = plt.subplots(1, 1)
                ax[i, j].plot(ts, np.real(Js[i, j, :]), label="real")
                ax[i, j].set_ylabel(f"J[{i},{j}]")
                ax[i, j].set_xlabel("t/T")

        ax[0, 0].legend()
        ax[0, 0].set_title(f"J(t) matrix {description}")

        if path != "":
            for i in range(hbm.fourier.n_dof):
                for j in range(hbm.fourier.n_dof):
                    plt.sca(ax[i, j])
                    tikzplotlib.save(f"{path}J_{description}_{i}{j}.tikz")


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
        path_drazin=path,
        path_ratio=path,
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

    # fourier = Fourier(N_HBM=30, L_DFT=2**13, n_dof=5, real_formulation=True)
    # name_case = "Schuetz2"

    # for smoothing in [np.inf, 200]:
    #     hbm = solve_hbm(name_case, smoothing=smoothing, fourier=fourier)
    #     plot_J_over_time(
    #         hbm,
    #         description=f"J_over_time_{name_case}_inf_N{fourier.N_HBM}_L{fourier.L_DFT}_smoothing{smoothing}",
    #         path="examples/dissertation_bayer/friction/plots/",
    #     )
    #     Phis = fundamat_over_time(
    #         hbm,
    #         L=200,
    #         description=f"Funda_mat_{name_case}_N{fourier.N_HBM}_L{fourier.L_DFT}_smoothing{smoothing}",
    #         path="examples/dissertation_bayer/friction/plots/",
    #         path_ref=f"examples/dissertation_bayer/friction/data/Phi_t_ref_{name_case}.mat",
    #     )
    main(
        cases=["A", "B"],
        smoothings=[np.inf],
        Ns_HBM=np.arange(1, 401),
        Ns_plot=(400,),
        L_DFT=2**12,
        max_residual=1e-9,
    )
    plt.show()
