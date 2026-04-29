"""FRC of frictional oscillator. See Schütz (2025), Bachelor's thesis, and Legrand2023."""

from typing import Any, Generator

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

from skhippr.visualization.cycles import (
    plot_period,
    plot_phase,
    plot_floquet_multipliers,
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
        plt.close("all")
        print("--------------------------------------------------------------------")
        print(f"solving N = {N_HBM} for {description}")
        fourier = Fourier(N_HBM, L_DFT, n_dof=5, real_formulation=True)
        hbm = solve_hbm(name_case, smoothing, fourier, hbm_ref)

        # hbms.append(hbm_ref)
        # figs = plot_and_save(hbms, Ns_HBM[: len(hbms)], description, tol_drazin)

        if np.linalg.norm(hbm_ref.residual(update=False)) > max_residual:
            print(
                f"Residual of reference solution is above threshold: {np.linalg.norm(hbm_ref.residual(update=False))} > {max_residual}. Stopping iteration."
            )
            break
        else:
            hbm_ref = hbm
            yield hbm


def plot_and_save_solution(hbm, description, idx=(0, 1), path_export=None):
    ax = None
    for i in idx:
        ax = plot_phase(hbm, idx=i, ax=ax, label=f"x_{i}")
    ax.set_title(f"{description}")
    ax.legend()
    if path_export is not None:
        tikzplotlib.save(f"{path_export}{description}.tikz")


def plot_hbm_result(hbm, description, path="plots/"):
    path = "plots/"
    r = np.linalg.norm(hbm.residual(update=False))

    # position, velocity, lambda in three plots
    ax_pos = plot_and_save_solution(hbm, f"position_{description}", (0, 1), path)
    ax_vel = plot_and_save_solution(hbm, f"velocity_{description}", (0, 1), path)
    ax_force = plot_and_save_solution(hbm, f"lambda_{description}", (0, 1), path)

    # force law
    ax_forcelaw = plot_phase(hbm, idx=(3, 4))
    ax_forcelaw.set_title(f"force_law_{description}")

    ax_forcelaw.set_xlabel("x3")
    ax_forcelaw.set_ylabel("lambda")
    ax_forcelaw.set_title(f"force law {description} r = {r}")
    ax_forcelaw.legend()
    tikzplotlib.save(f"{path}forcelaw_{description}.tikz")

    return ax_pos, ax_vel, ax_force, ax_forcelaw

    # np.savetxt(f"X_{description}.csv", hbm.X, delimiter=";")


def plot_and_save(hbms, Ns_HBM, description, tol_drazin=1e-7):
    figs = []
    hbm_ref = hbms[-1]
    print(f"a posteriori analysis {description}")

    FM_ref = sort_FMs(hbm_ref.eigenvalues)

    results_to_csv = np.zeros((len(Ns_HBM), hbm_ref.X.shape[0] + 2))
    header_components = lambda my_text: [
        f"X_{my_text}_{l}" for l in range(hbm_ref.fourier.n_dof)
    ]
    header_const = header_components("const")
    header_cos = []
    header_sin = []
    for k in range(1, hbm_ref.fourier.N_HBM + 1):
        header_cos += header_components(f"cos{k}")
        header_sin += header_components(f"sin{k}")
    header_csv = ["N_HBM", "HBM residual"] + header_const + header_cos + header_sin
    header_csv = ";".join(header_csv)

    e_hbm = np.zeros((5, len(Ns_HBM)))
    e_hbm_fourier = np.zeros((5, len(Ns_HBM)))
    e_stab = np.zeros((5, len(Ns_HBM)))
    FMs_all = np.zeros((5, len(Ns_HBM)), dtype=complex)

    # Drazin and Drazin ratio plots
    fig, ax_drazin = plt.subplots(1, 1)
    figs.append(fig)
    fig, ax_drazin_ratio = plt.subplots(1, 1)
    figs.append(fig)

    plot_drazin_and_ratio(
        hbms=hbms,
        tol_drazin=tol_drazin,
        description=description,
        ratio_limits=[3 / 5, 4 / 5],
        ax_drazin=ax_drazin,
        ax_ratio=ax_drazin_ratio,
        path_drazin=None,
        path_ratio=f"plots/",
    )

    for k, hbm in enumerate(hbms):

        x_time = hbm.x_time()
        X_comp = hbm_ref.fourier.DFT(x_time)
        X_comp[np.abs(X_comp) <= 1e-17] = 0
        e_comp = np.reshape(hbm_ref.X - X_comp, (5, -1), order="F")

        results_to_csv[k, 0] = hbm.fourier.N_HBM
        results_to_csv[k, 1] = np.linalg.norm(hbm.residual(update=False))
        results_to_csv[k, 2:] = X_comp

        e_hbm[:, k] = np.max(np.abs(x_time - hbm_ref.x_time()), axis=1)
        e_hbm_fourier[:, k] = np.linalg.norm(e_comp, axis=1)
        FMs = sort_FMs(hbm.eigenvalues)
        FMs_all[:, k] = FMs
        e_stab[:, k] = np.abs(FMs - FM_ref)

    # np.savetxt(
    #     f"\\\\inm-cifs.tik.uni-stuttgart.de\\users\\ac127316\\Research\\data\\2026_diss_friction\\X_{description}.csv",
    #     results_to_csv,
    #     delimiter=";",
    #     header=header_csv,
    # )
    np.savetxt(
        f"X_{description}.csv",
        results_to_csv,
        delimiter=";",
        header=header_csv,
    )

    # Plot Floquet multipliers
    fig, ax = plt.subplots(1, 1)
    phis = np.linspace(0, 2 * np.pi, 250)
    ax.plot(np.cos(phis), np.sin(phis))
    for l in range(FMs_all.shape[0]):
        ax.plot(np.real(FMs_all[l, :]), np.imag(FMs_all[l, :]), "-x", label=f"FM {l}")
    ax.plot(np.real(FM_ref), np.imag(FM_ref), "o", label=f"ref(N={N_max})")
    ax.set_aspect("equal")
    ax.set_title(description)
    ax.legend(loc="upper left")
    # tikzplotlib.save(
    #     f"\\\\inm-cifs.tik.uni-stuttgart.de\\users\\ac127316\\Research\\data\\2026_diss_friction\\FMs_conv_{description}.tikz"
    # )
    tikzplotlib.save(f"FMs_conv_{description}.tikz")
    figs.append(fig)

    # Plot HBM convergence in time
    fig, ax = plt.subplots(1, 1)
    for l in range(e_hbm.shape[0]):
        ax.semilogy(Ns_HBM[:-1], e_hbm[l, :-1], label=f"x{l}")
    ax.legend(loc="best")
    ax.set_xlabel("N")
    ax.set_ylabel("error HBM")
    ax.set_title(f"HBM convergence {description}")
    # tikzplotlib.save(
    #     f"\\\\inm-cifs.tik.uni-stuttgart.de\\users\\ac127316\\Research\\data\\2026_diss_friction\\HBM_error_{description}.tikz"
    # )
    tikzplotlib.save(f"HBM_error_{description}.tikz")
    figs.append(fig)

    # Plot HBM convergence in freq domain
    fig, ax = plt.subplots(1, 1)
    for l in range(e_hbm_fourier.shape[0]):
        ax.semilogy(Ns_HBM[:-1], e_hbm_fourier[l, :-1], label=f"x{l}")
    ax.legend(loc="best")
    ax.set_xlabel("N")
    ax.set_ylabel("error HBM FCs")
    ax.set_title(f"HBM FC convergence {description}")
    # tikzplotlib.save(
    #     f"\\\\inm-cifs.tik.uni-stuttgart.de\\users\\ac127316\\Research\\data\\2026_diss_friction\\HBM_FC_error_{description}.tikz"
    # )
    tikzplotlib.save(f"HBM_FC_error_{description}.tikz")
    figs.append(fig)

    # Plot FM convergence
    fig, ax = plt.subplots(1, 1)
    for l in range(e_stab.shape[0]):
        ax.semilogy(Ns_HBM[:-1], e_stab[l, :-1], label=f"FM {l}")
    ax.legend(loc="best")
    ax.set_xlabel("N")
    ax.set_ylabel("error FMs")
    ax.set_title(f"FM convergence {description}")
    # tikzplotlib.save(
    #     f"\\\\inm-cifs.tik.uni-stuttgart.de\\users\\ac127316\\Research\\data\\2026_diss_friction\\FMs_error_{description}.tikz"
    # )
    tikzplotlib.save(f"FMs_error_{description}.tikz")
    figs.append(fig)

    print(
        "==============================================================================================="
    )
    return figs


def sort_FMs(FMs, significant_digits=2):

    FMs_rounded = np.zeros_like(FMs)
    for k, FM in enumerate(FMs):
        FMs_rounded[k] = round_to_significant_digits(
            FM, significant_digits=significant_digits
        )
    idx_sort = np.lexsort((np.angle(FMs_rounded), np.abs(FMs_rounded)))
    FMs = FMs[idx_sort]
    FMs_rounded = FMs_rounded[idx_sort]
    # Separate complex and real eigenvalues
    FMs = np.hstack((FMs[np.imag(FMs_rounded) == 0], FMs[np.imag(FMs_rounded) != 0]))
    return FMs


def plot_everything():

    Ns_HBM = [40]
    L_DFT = 2**13

    smoothings = [np.inf]

    for name_case in ["C"]:
        hbm = None

        for k, N_HBM in enumerate(Ns_HBM):
            fourier = Fourier(N_HBM, L_DFT, n_dof=5, real_formulation=True)

            for l, alpha in enumerate(smoothings):
                hbm = solve_hbm(name_case, alpha, fourier, hbm)

                _, axs = plt.subplots(2, 2)
                x_time = hbm.x_time()

                for i in range(2):
                    axs[i][0].plot(x_time[i, :], x_time[i + 2, :], "-")
                    axs[i][0].set_title(f"Phase plot of x_[{i}]")
                    axs[i][0].set_ylabel(f"dx_[{i}]")
                    axs[i][0].set_xlabel(f"x_[{i}]")

                floquet_multipliers = hbm.eigenvalues
                axs[0][1].plot(
                    np.real(floquet_multipliers), np.imag(floquet_multipliers), "x"
                )
                axs[0][1].set_title("Floquet multipliers")
                axs[0][1].plot(
                    np.cos(hbm.fourier.time_samples_normalized),
                    np.sin(hbm.fourier.time_samples_normalized),
                    "k",
                )
                axs[0][1].axis("equal")
                axs[0][0].set_title(
                    f"Friction oscillator N_HBM = {N_HBM}, alpha = {alpha}"
                )

                _, axs = plt.subplots(hbm.n_dof, 1)
                x_time = hbm.x_time()
                for i in range(hbm.n_dof):
                    axs[i].plot(hbm.fourier.time_samples(hbm.omega), x_time[i, :])
                axs[0].set_title(
                    f"Friction oscillator N_HBM = {N_HBM}, alpha = {alpha}"
                )


def plot_frc():

    solver = ScipyRootSolver(
        tolerance=1e-7, max_iterations=100, verbose=True, use_fprime=True, method="lm"
    )

    # BA Schütz case 2 (p. 50)
    masses = [1, 1]
    g = 10
    stiffnesses = [1, 1]
    dampings = [0.02, 0.02]
    forcings = [20, 0]
    omega = 0.1
    phases = [0.5 * np.pi, 0]
    mu = 0.9
    smoothing = 10
    prox_parameter = 1

    # Systems
    dae_smooth = SmoothedFrictionOscillator(
        stiffnesses=stiffnesses,
        dampings=dampings,
        masses=masses,
        g=g,
        mu=mu,
        forcing_amplitudes=forcings,
        forcing_phases=phases,
        smoothing=smoothing,
    )

    dae_nonsmooth = FrictionOscillator(
        stiffnesses=stiffnesses,
        dampings=dampings,
        masses=masses,
        g=g,
        mu=mu,
        forcing_amplitudes=forcings,
        forcing_phases=phases,
        prox_parameter=prox_parameter,
    )

    fourier = Fourier(
        N_HBM=60, L_DFT=2000, n_dof=dae_smooth.n_dof, real_formulation=True
    )

    fig, axs = plt.subplots(1, 2)
    alphas = [smoothing, "nonsmooth"]

    for k, dae in enumerate([dae_smooth, dae_nonsmooth]):

        if k == 0:
            initial_guess = 0.1 * np.random.rand(dae.n_dof * (2 * fourier.N_HBM + 1))
        else:
            initial_guess = hbm.X

        hbm = HBMEquationDAE(
            dae,
            omega,
            fourier=fourier,
            initial_guess=initial_guess,
            stability_method=KoopmanHillDAE(
                fourier, tol=1e-4, autonomous=False, tol_drazin=1e-6
            ),
        )

        sys = EquationSystem(
            equations=[hbm], unknowns=["X"], equation_determining_stability=hbm
        )

        solver.verbose = True
        solver.solve(sys)
        solver.verbose = False

        for branch_point in pseudo_arclength_continuator(
            initial_system=sys,
            solver=solver,
            continuation_parameter="omega",
            stepsize=0.1,
            stepsize_range=[0.001, 2],
            num_steps=10,
            verbose=True,
        ):
            if branch_point.stable:
                color = "r"
            else:
                color = "b"
            axs[k].plot(
                branch_point.omega,
                np.max(np.abs(branch_point.equations[0].x_time()[0, :])),
                f"{color}.",
            )

            if branch_point.omega > 1:
                break

        axs[k].set_title(f"Friction oscillator FRC smoothing = {alphas[k]}")
        axs[k].set_xlabel("omega")
        axs[k].set_ylabel("|x[0]| max")


if __name__ == "__main__":
    N_min = 10
    N_max = 1200
    Ns = [
        int(N)
        for N in np.unique(np.round(np.logspace(np.log10(N_min), np.log10(N_max), 60)))
    ]
    # Ns = [80]
    # Ns = np.arange(1, N_max + 1)
    print(Ns)
    # Ns = Ns + [N_max + k for k in range(1, 11)]
    for name_case in ["B"]:
        for smoothing in [np.inf]:  # np.inf case B fehlt noch
            # plot_and_export_hbm(name_case, smoothing=smoothing, N_HBM=80, L_DFT=4096)
            try:
                iterate_over_N(
                    name_case,
                    Ns_HBM=Ns,
                    L_DFT=2**13,
                    smoothing=smoothing,
                    max_residual=1e-5,
                )
                plt.close("all")
            except MemoryError:
                plt.close("all")
                continue
    # plot_everything()
    # plot_frc()
    # plt.show()
