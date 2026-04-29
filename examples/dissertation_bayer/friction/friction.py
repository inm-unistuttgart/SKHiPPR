"""FRC of frictional oscillator. See Schütz (2025), Bachelor's thesis, and Legrand2023."""

import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
from scipy.linalg import (
    lu_factor,
    lu_solve,
)


from skhippr.odes.daes import FrictionOscillator, SmoothedFrictionOscillator
from skhippr.Fourier import Fourier, round_to_significant_digits
from skhippr.solvers.newton import ScipyFsolveSolver, NewtonSolver, ScipyRootSolver
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.cycles.hbm import HBMEquationDAE
from skhippr.solvers.continuation import pseudo_arclength_continuator

from skhippr.stability.KoopmanHillProjection import (
    KoopmanHillDAE,
    drazin,
)

from friction_init import init_oscillator


def solve_hbm(name_case, smoothing=np.inf, fourier=None, hbm_ref=None, solver=None):
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
        X_old = hbm_ref.X
        X_old = np.reshape(X_old, (dae.n_dof, -1), order="F")
        Lambda_old = X_old[-1, :]

        initial_guess_lambda = np.zeros(2 * fourier.N_HBM + 1)

        idx_cos_end_old = min(hbm_ref.fourier.N_HBM + 1, fourier.N_HBM + 1)
        idx_cos_end_new = idx_cos_end_old

        idx_sin_start_old = hbm_ref.fourier.N_HBM + 1
        idx_sin_end_old = idx_cos_end_old + hbm_ref.fourier.N_HBM

        idx_sin_start_new = fourier.N_HBM + 1
        idx_sin_end_new = idx_cos_end_old + fourier.N_HBM

        initial_guess_lambda[:idx_cos_end_new] = Lambda_old[:idx_cos_end_old]
        initial_guess_lambda[idx_sin_start_new:idx_sin_end_new] = Lambda_old[
            idx_sin_start_old:idx_sin_end_old
        ]

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

    # Solve actual problem
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


def plot_and_export_hbm(name_case="C", smoothing=np.inf, N_HBM=160, L_DFT=2**14):
    fourier = Fourier(N_HBM, L_DFT, n_dof=5, real_formulation=True)
    hbm = solve_hbm(name_case, smoothing, fourier)
    x_time = hbm.x_time()
    ts = hbm.fourier.time_samples(hbm.omega)
    residual = np.linalg.norm(hbm.residual(update=True))

    description = f"case-{name_case}-N-{N_HBM}-smoothing-{smoothing}"

    # Plot position
    _, ax = plt.subplots(1, 1)
    ax.plot(ts, x_time[0, :], label="x0")
    ax.plot(ts, x_time[1, :], label="x1")
    ax.set_xlabel("t")
    ax.set_ylabel("position")
    ax.set_title(f"position {description} r = {residual}")
    ax.legend()
    tikzplotlib.save(f"plots/position_{description}.tikz")

    # Plot velocity
    _, ax = plt.subplots(1, 1)
    ax.plot(ts, x_time[2, :], label="x2")
    ax.plot(ts, x_time[3, :], label="x3")
    ax.set_xlabel("t")
    ax.set_ylabel("velocity")
    ax.set_title(f"velocity {description} r = {residual}")
    ax.legend()
    tikzplotlib.save(f"plots/velocity_{description}.tikz")

    # Plot tangential force
    _, ax = plt.subplots(1, 1)
    ax.plot(ts, x_time[4, :], label="lambda")
    ax.set_xlabel("t")
    ax.set_ylabel("lambda")
    ax.set_title(f"lambda {description} r = {residual}")
    ax.legend()
    tikzplotlib.save(f"plots/lambda_{description}.tikz")

    # Plot force law
    _, ax = plt.subplots(1, 1)
    ax.plot(x_time[3, :], x_time[4, :], label="lambda")
    ax.set_xlabel("x3")
    ax.set_ylabel("lambda")
    ax.set_title(f"force law {description} r = {residual}")
    ax.legend()
    tikzplotlib.save(f"plots/forcelaw_{description}.tikz")

    np.savetxt(f"X_{description}.csv", hbm.X, delimiter=";")


def convergence_study_N(
    name_case="C",
    smoothing=np.inf,
    Ns_HBM=(30, 40, 50),
    L_DFT=2**14,
    tol_drazin=1e-7,
    max_residual=1e-9,
):

    N_max = Ns_HBM[-1]
    description = f"{name_case}-Nmax{N_max}-smoothing{smoothing}-L{L_DFT}"

    hbms = []
    figs = []
    hbm_ref = None

    for N_HBM in Ns_HBM:
        for fig in figs:
            plt.close(fig)
        print("--------------------------------------------------------------------")
        print(f"solving N = {N_HBM} for {description}")
        fourier = Fourier(N_HBM, L_DFT, n_dof=5, real_formulation=True)
        hbm_ref = solve_hbm(name_case, smoothing, fourier, hbm_ref)

        hbms.append(hbm_ref)
        figs = plot_and_save(hbms, Ns_HBM[: len(hbms)], description, tol_drazin)

        if np.linalg.norm(hbm_ref.residual(update=False)) > max_residual:
            hbm_ref = hbms[-1]
            break


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

    drazin_ratios = np.zeros(len(Ns_HBM))
    fig, ax_drazin = plt.subplots(1, 1)
    figs.append(fig)

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

        # Drazin inverse analysis
        drazin_ratios[k] = analyze_drazin(hbm, ax_drazin, tol_drazin=tol_drazin)

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

    ax_drazin.axhline(tol_drazin, linestyle="--")
    ax_drazin.set_xlabel("n*(2*N+1)")
    ax_drazin.set_ylabel("magnitude of eigenvalue")
    ax_drazin.set_title(f"Drazin eigenvalues {description}")
    # tikzplotlib.save(
    #     f"\\\\inm-cifs.tik.uni-stuttgart.de\\users\\ac127316\\Research\\data\\2026_diss_friction\\drazin_{description}.tikz"
    # )
    tikzplotlib.save(f"drazin_{description}.tikz")

    fig, ax_drazin_ratio = plt.subplots(1, 1)
    ax_drazin_ratio.plot(Ns_HBM, drazin_ratios, "-x")
    ax_drazin_ratio.axhline(4 / 5, linestyle="--")
    ax_drazin_ratio.axhline(3 / 5, linestyle="--")
    ax_drazin_ratio.set_title(f"Drazin ratio {description}")
    #
    tikzplotlib.save(f"drazin_ratio_{description}.tikz")
    figs.append(fig)

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


def analyze_drazin(
    hbm: HBMEquationDAE,
    ax=None,
    tol_cond=1e6,
    tol_drazin=1e-7,
):

    hill_matrix = hbm.hill_matrix(update=True)
    mass_matrix = hbm.M()

    # Copy&pasted from KoopmanHillDAE.generalized_exponential()
    a_vals = [1.0, 10.0, 0.1, 100, 0.01, 1000, 0.001]
    success = False
    for a in a_vals:
        pencil = a * mass_matrix - hill_matrix
        if np.linalg.cond(pencil) < tol_cond:
            success = True
            # print(f"N = {N}: a = {a}")
            break
    if not success:
        raise RuntimeError(
            f"Could not find suitable scaling factor 'a' for Drazin inverse with condition < {tol_cond}."
        )

    pencil_lu = lu_factor(a * mass_matrix - hill_matrix)
    pencil_M = lu_solve(pencil_lu, mass_matrix)
    _, ratio = drazin(pencil_M, tol_drazin, ax_plot=ax)
    # print(f"Ratio of Drazin inverse: {ratio}")
    return ratio


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
                convergence_study_N(
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
