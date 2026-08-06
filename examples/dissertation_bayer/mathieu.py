"""Mathieu equation examples for dissertation by Fabia Bayer, 2026"""

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import tqdm
from scipy.linalg import expm


from skhippr.odes.ltp import MathieuODE, TruncatedMeissner
from skhippr.solvers.newton import NewtonSolver
from skhippr.Fourier import Fourier
from skhippr.cycles.hbm import HBMEquation
from skhippr.stability.ClassicalHill import ClassicalHill
from skhippr.stability.KoopmanHillProjection import (
    KoopmanHillProjection,
    KoopmanHillSubharmonic,
)
from skhippr.equations.PseudoSpectrumEquation import compute_pseudospectrum

solver = NewtonSolver()


def plot_all_hill_EVs(
    ode,
    N_HBM=10,
    ax_FE=None,
    ax_FM=None,
    save=False,
    vals_eigenvector=(),
    real_formulation=False,
    **plot_args,
):
    fourier = Fourier(N_HBM=N_HBM, L_DFT=1024, n_dof=2)
    # ode = TruncatedMeissner(
    #     t=0,
    #     x=np.array([0.0, 0.0]),
    #     a=delta,
    #     b=epsilon,
    #     omega=omega,
    #     damping=damping,
    #     N_harmonics=20,
    # )
    omega = ode.omega
    hbm = HBMEquation(
        ode,
        omega=omega,
        fourier=fourier,
        initial_guess=np.zeros((2 * N_HBM + 1) * 2),
        stability_method=ClassicalHill(fourier, "symmetry"),
    )
    solver = NewtonSolver(verbose=True)
    solver.solve_equation(hbm, "X")

    hill_matrix = hbm.hill_matrix(real_formulation=real_formulation)
    print(hill_matrix.shape)

    eigenvalues, eigenvectors = np.linalg.eig(hill_matrix)

    if real_formulation:
        eigenvectors = fourier.T_to_cplx_from_real @ eigenvectors

    if ax_FE is None:
        _, ax_FE = plt.subplots(1, 1)

    ax_FE.plot(np.real(eigenvalues), np.imag(eigenvalues), ".", **plot_args)
    if save:
        tikzplotlib.save(
            f"examples/dissertation_bayer/plots/mathieu_spurious_delta_{ode.a}_epsilon_{ode.b}_damping_{ode.damping}.tikz"
        )

    if ax_FM is not None:
        FMs = np.exp(eigenvalues * 2 * np.pi / omega)
        ax_FM.plot(np.real(FMs), np.imag(FMs), ".", **plot_args)
        FMs_choice = hbm.eigenvalues
        ax_FM.plot(
            np.real(FMs_choice), np.imag(FMs_choice), "x", mfc="none", **plot_args
        )
        ax_FM.plot()

    for val in vals_eigenvector:
        idx_min = np.argmin(np.abs(eigenvalues - val))
        eigenvector = eigenvectors[:, idx_min]
        eigenvector = np.reshape(eigenvector, shape=(ode.n_dof, -1), order="F")
        norms_ev = np.linalg.norm(eigenvector, ord=2, axis=0)
        _, ax = plt.subplots(1, 1)
        ax.bar(np.arange(-fourier.N_HBM, fourier.N_HBM + 1), norms_ev)
        plt.title(f"eigenvector norm to eigenvalue {eigenvalues[idx_min]}")
        tikzplotlib.save(
            f"examples/dissertation_bayer/plots/mathieu_eigenvectors_{np.real(val)}_{np.imag(val)}.tikz"
        )
        w = hbm.stability_method._weighted_mean(
            eigenpair=(eigenvalues[idx_min], eigenvectors[:, idx_min])
        )
        print(f"alpha= {eigenvalues[idx_min]}:weighted mean =0.5 + {w - 0.5}")

    return hill_matrix


def plot_mathieu_pseudospectrum(ode, N_HBM=10):
    # Plot sorted eigenvalues of Hill matrix
    hill_mat = plot_all_hill_EVs(ode, N_HBM=10)
    FE_all, _ = np.linalg.eig(hill_mat)

    # Consider LTI variant
    lti = ...  # TO DO
    A = lti.derivative(variable="x", update=True)
    hill_mat_A = np.kron(np.eye(2 * N_HBM + 1), A) + np.kron(
        np.diag(np.arange(-N_HBM, N_HBM + 1)), -1j * ode.omega * np.eye(2)
    )
    E = np.linalg.norm(hill_mat_A - hill_mat, ord=2)
    print(f"E for pseudospectrum: {E}")

    plt.figure()
    plt.plot(np.real(FE_all), np.imag(FE_all), "kx")
    alphas, _ = np.linalg.eig(a=A)
    for alpha in alphas:
        # for k in range(-N_HBM, N_HBM + 1):
        for k in [0]:
            bdry = compute_pseudospectrum(
                hill_mat_A,
                E,
                z_init=alpha + 1j * k * ode.omega,
                verbose=True,
                tolerance=1e-11,
                max_step=0.001,
            )
            print(f"Pseudospectrum N = {k} computed.")
            plt.plot(np.real(alpha), np.imag(alpha) + k * ode.omega, "r.")
            plt.plot(np.real(bdry), np.imag(bdry), "b")


def continuity_sort(eigenvalues_old, eigenvalues_new, omega):

    eigenvalues_sorted = []
    idx_all = np.arange(len(eigenvalues_new))

    if len(eigenvalues_old) < len(eigenvalues_new):
        N_HBM = int(0.5 * (len(eigenvalues_new) / len(eigenvalues_old) - 1))
        for k in zigzag(N_HBM):
            for eigenvalue_old in eigenvalues_old:
                FE = eigenvalue_old + k * omega * 1j
                idx_candidates = idx_all[
                    np.abs(np.imag(eigenvalues_new) - np.imag(eigenvalue_old))
                    < 0.5 * omega
                ]
                if len(idx_candidates) == 0:
                    idx_candidates = idx_all
                idx_new = np.argmin(np.abs(FE - eigenvalues_new[idx_candidates]))
                idx_new = idx_candidates[idx_new]
                eigenvalues_sorted.append(eigenvalues_new[idx_new])
                eigenvalues_new[idx_new] = np.inf

    else:
        for FE in eigenvalues_old:
            idx_candidates = idx_all[
                np.abs(np.imag(eigenvalues_new) - np.imag(FE)) < 0.5 * omega
            ]
            if len(idx_candidates) == 0:
                idx_candidates = idx_all
            idx_new = np.argmin(np.abs(FE - eigenvalues_new[idx_candidates]))
            idx_new = idx_candidates[idx_new]
            eigenvalues_sorted.append(eigenvalues_new[idx_new])
            eigenvalues_new[idx_new] = np.inf

    eigenvalues_sorted = np.array(eigenvalues_sorted)
    eigenvalues_new[:] = eigenvalues_sorted
    return eigenvalues_sorted


def zigzag(N):
    yield 0
    for i in range(1, N + 1):
        yield i
        yield -i


def plot_eigenvalues_continuously(
    hbm, name_param, vals_param, ax_cont=None, ax_FE=None, ax_FM=None, solver=None
):

    if solver is None:
        solver = NewtonSolver(verbose=False)

    if ax_cont is None:
        _, ax_cont = plt.subplots(1, 1)

    eigenvalues_old = []
    factors = (vals_param - np.min(vals_param)) / (
        np.max(vals_param) - np.min(vals_param)
    )

    eigenvalues_all = np.zeros((len(hbm.X), len(vals_param)), dtype=complex)

    for k, (param, factor) in enumerate(zip(vals_param, factors)):
        print(f"{name_param} = {param}")
        setattr(hbm, name_param, param)
        solver.solve_equation(hbm, "X")

        if len(eigenvalues_old) == 0:
            # old eigenvalues derived from constant case
            J_coeffs = hbm.ode_coeffs()
            if hbm.fourier.real_formulation:
                A = J_coeffs[:, :, 0]
            else:
                A = J_coeffs[:, :, hbm.fourier.N_HBM]
            eigenvalues_old, _ = np.linalg.eig(A)

        eigenvalues, eigenvectors = np.linalg.eig(hbm.hill_matrix(update=True))
        FEs, _ = hbm.stability_method.hill_EVP(hbm, visualize=False)

        color = (factor, 0, factor)

        if ax_FE is not None:
            ax_FE.plot(np.real(eigenvalues), np.imag(eigenvalues), ".", color=color)
            ax_FE.plot(np.real(FEs), np.imag(FEs), "o", color=color, mfc="none")

        if ax_FM is not None:
            FMs_all = np.exp(eigenvalues * 2 * np.pi / hbm.omega)
            FMs = np.exp(FEs * 2 * np.pi / hbm.omega)
            ax_FM.plot(np.real(FMs_all), np.imag(FMs_all), ".", color=color)
            ax_FM.plot(np.real(FMs), np.imag(FMs), "o", color=color, mfc="none")

        eigenvalues = continuity_sort(eigenvalues_old, eigenvalues, hbm.omega)
        eigenvalues_all[:, k] = eigenvalues
        eigenvalues_old = eigenvalues

        ax_cont.cla()
        for ii in range(len(eigenvalues_all[:, 0])):
            ax_cont.plot(
                np.real(eigenvalues_all[ii, : (k + 1)]),
                np.imag(eigenvalues_all[ii, : (k + 1)]),
                ".",
            )
        pass


def last_try_accuracy(N_max, delta, epsilon, damping, omega):

    ode = MathieuODE(0, np.array([0, 0]), a=delta, b=epsilon, damping=damping, omega=2)

    # Reference
    print(f"Reference solution...")
    fourier_ref = Fourier(2 * N_max, 2 * 1024, 2, real_formulation=False)
    hbm_ref = HBMEquation(
        ode,
        ode.omega,
        fourier_ref,
        initial_guess=np.zeros(
            (ode.n_dof * (2 * fourier_ref.N_HBM + 1)), dtype=complex
        ),
        period_k=1,
        stability_method=None,
    )
    stabm_ref = KoopmanHillSubharmonic(fourier_ref)
    print(f"Residual: {np.linalg.norm(hbm_ref.residual(update=True))}")
    hbm_ref.hill_matrix(update=True)
    FMs_ref = stabm_ref.determine_eigenvalues(hbm=hbm_ref)
    print(f"done. FMs: {FMs_ref}")

    def hbm_direct(fourier):
        stab_method = KoopmanHillProjection(fourier)
        ode = MathieuODE(
            t=0, x=np.array([0, 0]), a=delta, b=epsilon, omega=omega, damping=damping
        )

        initial_guess = np.zeros(fourier.n_dof * (2 * fourier.N_HBM + 1))

        return HBMEquation(
            ode=ode,
            omega=omega,
            fourier=fourier,
            initial_guess=initial_guess,
            period_k=1,
            stability_method=stab_method,
        )

    def hbm_subh(fourier):
        stab_method = KoopmanHillSubharmonic(fourier)
        ode = MathieuODE(
            t=0, x=np.array([0, 0]), a=delta, b=epsilon, omega=omega, damping=damping
        )

        initial_guess = np.zeros(fourier.n_dof * (2 * fourier.N_HBM + 1))

        return HBMEquation(
            ode=ode,
            omega=omega,
            fourier=fourier,
            initial_guess=initial_guess,
            period_k=1,
            stability_method=stab_method,
        )

    def hbm_ones(fourier):
        stab_method = KoopmanHillProjection(fourier)
        C = np.ones((1, 2 * fourier.N_HBM + 1))
        C[1::2] = -1
        stab_method.C = np.kron(C, np.eye(fourier.n_dof))

        ode = MathieuODE(
            t=0, x=np.array([0, 0]), a=delta, b=epsilon, omega=omega, damping=damping
        )

        initial_guess = np.zeros(fourier.n_dof * (2 * fourier.N_HBM + 1))

        return HBMEquation(
            ode=ode,
            omega=omega,
            fourier=fourier,
            initial_guess=initial_guess,
            period_k=2,
            stability_method=stab_method,
        )

    def hbm_p2_direct(fourier):
        stab_method = KoopmanHillProjection(fourier)

        ode = MathieuODE(
            t=0, x=np.array([0, 0]), a=delta, b=epsilon, omega=omega, damping=damping
        )

        initial_guess = np.zeros(fourier.n_dof * (2 * fourier.N_HBM + 1))

        return HBMEquation(
            ode=ode,
            omega=omega,
            fourier=fourier,
            initial_guess=initial_guess,
            period_k=2,
            stability_method=stab_method,
        )

    errors = np.ones((4, N_max + 1))
    plt.figure()
    for idx, (hbm_template, t_over_period) in enumerate(
        zip((hbm_direct, hbm_subh, hbm_p2_direct, hbm_ones), (1, 1, 1, 1))
    ):
        errors[idx, :] = error_over_N(hbm_template, t_over_period, N_max, FMs_ref)
        plt.semilogy(errors[idx, :])
        plt.legend(
            ["direct", "subh", "direct with half omega", "alternating with half omega"]
        )
    tikzplotlib.save(
        f"examples/dissertation_bayer/plots/mathieu_delta_{ode.a}_epsi_{ode.b}_subh_accuracy.tikz"
    )
    plt.show()


def error_over_N(hbm_template, t_over_period, N_max, FMs_ref):
    error = np.ones(N_max + 1)
    for N_HBM in range(N_max + 1):
        fourier = Fourier(N_HBM=N_HBM, L_DFT=1024, n_dof=2, real_formulation=False)
        hbm = hbm_template(fourier)
        error[N_HBM] = error_individual(hbm, t_over_period, FMs_ref)
    return error


def error_individual(hbm, t_over_period, FMs_ref):
    if hbm.fourier.real_formulation:
        raise ValueError("This function only works for complex formulation!")

    W = np.kron(np.ones((2 * hbm.fourier.N_HBM + 1, 1)), np.eye(2))
    if np.linalg.norm(hbm.residual(update=True)) > 0:
        raise ValueError(
            f"Passed un-solved HBM equation with residual {hbm.residual(update=False)}"
        )
    hbm.hill_matrix(update=True)

    Phi = hbm.stability_method.fundamental_matrix(t_over_period, hbm)
    FMs, _ = np.linalg.eig(Phi)
    return min(np.max(np.abs((FMs - FMs_ref))), np.max(np.abs((FMs[::-1] - FMs_ref))))


if __name__ == "__main__":

    last_try_accuracy(N_max=20, delta=1.5, epsilon=1, damping=0.01, omega=2)
    # # # Plot first demo case
    # # epsilon = 3
    # # delta = 0.15
    # # ode = MathieuODE(t=0, x=np.array([0, 0]), a=delta, b=epsilon, damping=0.1, omega=1)
    # # plot_all_hill_EVs(
    # #     ode=ode,
    # #     N_HBM=10,
    # #     save=True,
    # #     ax_FE=None,
    # #     vals_eigenvector=[0.1j, 8j],
    # #     real_formulation=False,
    # # )
    # # # plot_mathieu_pseudospectrum(epsilon=3, delta=2)

    # # # Unstable case with negative Floquet multipliers
    # # epsilon = 2
    # # delta = 0.15
    # # ode = MathieuODE(t=0, x=np.array([0, 0]), a=delta, b=epsilon, damping=0.1, omega=1)
    # # plot_all_hill_EVs(
    # #     ode=ode,
    # #     N_HBM=7,
    # #     save=True,
    # #     ax_FE=None,
    # #     vals_eigenvector=[-0.75 + 0.5j, -0.75 - 0.5j, 0.5 + 0.5j, 0.5 - 0.5j],
    # #     real_formulation=True,
    # # )

    # # Plot varying epsilon case
    # fig, ax = plt.subplots(2, 2)
    # phis = np.linspace(0, 2 * np.pi, 200)
    # ax[0, 1].plot(np.cos(phis), np.sin(phis), "k")
    # ax[0, 1].set_aspect("equal")
    # ax[0, 1].set_xlim([-1.5, 1.5])
    # ax[0, 1].set_ylim([-1.5, 1.5])
    # ax[0, 0].set_xlim([-1, 1])

    # delta = 12
    # epsilons = np.linspace(0, 12, 200)
    # fourier = Fourier(N_HBM=3, real_formulation=False, L_DFT=1024, n_dof=2)

    # ode = MathieuODE(t=0, x=np.array([0.0, 0.0]), b=epsilons[0], a=delta)
    # hbm = HBMEquation(
    #     ode=ode,
    #     omega=ode.omega,
    #     fourier=fourier,
    #     initial_guess=np.zeros(ode.n_dof * (2 * fourier.N_HBM + 1)),
    #     stability_method=ClassicalHill(fourier, "symmetry"),
    # )

    # plot_eigenvalues_continuously(
    #     hbm, "b", epsilons, ax[1, 0], ax_FE=ax[0, 0], ax_FM=ax[0, 1]
    # )

    # delta_max = 10
    # epsilon_max = 20
    # for factor in np.linspace(0, 1, 10):
    #     ode = MathieuODE(
    #         t=0,
    #         x=np.array([0, 0]),
    #         b=factor * epsilon_max,
    #         a=delta_max,
    #         damping=0.001,
    #         omega=1,
    #     )
    #     plot_all_hill_EVs(
    #         ode=ode,
    #         N_HBM=3,
    #         ax_FE=ax[0],
    #         ax_FM=ax[1],
    #         color=(factor, 0, factor),
    #         save=False,
    #     )
    #     # TODO plot eigenvalues continuously

    plt.show()
