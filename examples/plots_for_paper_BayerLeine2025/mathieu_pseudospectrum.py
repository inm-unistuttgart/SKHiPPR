import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tikzplotlib
import numpy as np
from scipy.integrate import solve_ivp
from tqdm import trange, tqdm

import csv

from skhippr.Fourier import Fourier
from skhippr.odes.ltp import MathieuODE
from skhippr.cycles.hbm import HBMEquation, HBMSystem

from skhippr.stability.AbstractStabilityHBM import AbstractStabilityHBM
from skhippr.stability.KoopmanHillProjection import (
    KoopmanHillProjection,
    KoopmanHillSubharmonic,
)


from skhippr.equations.PseudoSpectrumEquation import (
    does_pseudospectrum_include,
    compute_pseudospectrum,
)


def optimal_error_bound(hbm: HBMEquation, t: float, subharmonic: bool = False):
    """
    Computes the optimal error bound for the solution of a Mathieu-type equation using the Harmonic Balance Method (HBM).
    The error bound is calculated based on the norms of the Fourier coefficient matrices of the system, assuming that the Fourier coefficient matrices have finite support and only J_0 and J_1 are nonzero.
    The formulas for the error are derived in Bayer & Leine (2025).

    Parameters
    ----------

    hbm : HBMEquation
        An instance of the HBMEquation class representing the system, containing the ODE and Fourier parameters.
    t : float
        The time parameter at which the error bound is evaluated.

    Returns
    -------

    E : float
        The computed optimal error bound for the given system and time.

    Notes
    -----

    - The function temporarily sets the parameter `b` of the ODE to zero to compute the norm of the matrix J_0.
    - The error bound formula depends on the relationship between `beta` (norm of J_0), `gamma` (norm of J_1),
      the number of harmonics `N`, and the time `t`.
    """

    # Determine norms of Fourier coeff matrices J_0, J_{\pm 1}
    b = hbm.ode.b
    hbm.ode.b = 0
    J_0 = hbm.ode.closed_form_derivative("x")
    hbm.ode.b = b
    beta = np.linalg.norm(J_0, 2)

    J_1 = np.array([[0, 0], [-0.5 * b, 0]])
    gamma = np.linalg.norm(J_1)

    # Optimal error bound - see Bayer&Leine, 2025
    N = hbm.fourier.N_HBM
    if subharmonic:
        N = 2 * N

    if beta < N / (4 * t):
        E = (8 * gamma * t / N) ** N * (np.exp(N) - 1)
    else:
        E = (2 * gamma / beta) ** N * (np.exp(4 * beta * t) - 1)

    return E


def plot_FMs_with_guarantee(ode, N, subh, ax=None, color=None, **kwargs):
    fourier = Fourier(N_HBM=N, L_DFT=10 * N, n_dof=2)
    X = np.zeros(2 * (2 * N + 1))
    hbm = HBMEquation(ode, ode.omega, fourier, X)
    hbm.residual(update=True)

    if subh:
        stability_method = KoopmanHillSubharmonic(fourier=fourier)
    else:
        stability_method = KoopmanHillProjection(fourier=fourier)

    if ax is None:
        ax = plot_FM(hbm, stability_method)

    E = optimal_error_bound(hbm, hbm.T_solution, subh)

    print(f"N = {fourier.N_HBM}: E = {E}")

    Phi_T = stability_method.fundamental_matrix(t_over_period=1, hbm=hbm)
    FMs, _ = np.linalg.eig(Phi_T)

    if 1e-14 < E < 10:
        for k, FM in enumerate(FMs):
            z_pseudospectrum = compute_pseudospectrum(
                Phi_T, epsilon=E, z_init=FM, verbose=False, max_step=2e-4
            )
            if k == 0:
                label = f"N = {N}"
            else:
                label = None
            ax.plot(
                np.real(z_pseudospectrum),
                np.imag(z_pseudospectrum),
                label=label,
                **kwargs,
            )

    return ax


def plot_FM(hbm: HBMEquation, method: AbstractStabilityHBM = None):
    """Plot Floquet multipliers and unit circle"""

    _, ax = plt.subplots(1, 1)
    phis = np.linspace(0, 2 * np.pi, 1000)
    ax.plot(np.cos(phis), np.sin(phis), color="gray")
    ax.set_aspect("equal")

    if method is None:
        method = hbm.stability_method

    FMs = method.determine_eigenvalues(hbm)
    ax.plot(np.real(FMs), np.imag(FMs), "rx", label="Floquet multipliers")

    return ax


def plot_near_PD():
    omega = 1
    # a_vals = (-0.3549, -0.35485) # for omega = 2
    a_vals = (0.25 * omega**2 * -0.3549, 0.25 * omega**2 * -0.35485)  # for omega = 1
    for a in a_vals:
        # ode = MathieuODE(0, np.array([0, 0]), a, 2.4, omega=2)
        ode = MathieuODE(0, np.array([0, 0]), a, 0.25 * omega**2 * 2.4, omega=omega)

        n_periods = 5
        # Simulate ODE at this configuration over 10 periods
        print(f"Wait very long: Integrating over {n_periods} periods")
        T = 2 * np.pi / ode.omega
        sol = solve_ivp(
            ode.dynamics,
            (0, n_periods * T),
            [0, 0.05],
            dense_output=True,
            rtol=1e-13,
            atol=1e-13,
        )
        plt.figure()
        plt.plot(sol.t / T, sol.y[0])
        plt.xlabel("t/T")
        plt.ylabel("y_0")
        plt.title(f"Evolution for a = {a}")
        tikzplotlib.save(f"examples/convergence/evolution_a_{a}_b_{ode.b}.tikz")

        ax = None
        styles = ("--", "-", "-.")
        for N, style in zip([50, 51, 52], styles):
            ax = plot_FMs_with_guarantee(
                ode, N, subh=False, ax=ax, linestyle=style, color="k"
            )
        ax.set_title(f"Floquet multipliers and confidence regions for a = {a}")
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.legend()
        ax.set_xlim([-1.1, -0.9])
        ax.set_ylim([-0.13, 0.13])
        tikzplotlib.save(f"examples/convergence/FMs_a_{a}_b_{ode.b}.tikz")


def plot_N_over_a(ode, a_values, E_des=1e-6, fourier_ref=None, subh=True):

    unit_circle = np.exp(1j * np.linspace(0, 2 * np.pi, 250))
    real_axis = np.linspace(-1.1, 1.1, 250)

    if fourier_ref is None:
        fourier_ref = Fourier(N_HBM=100, L_DFT=1024, n_dof=2, real_formulation=True)

    N_vals = np.nan * np.ones((4, len(a_values)))
    T = 2 * np.pi / ode.omega
    if subh:
        StabClass = KoopmanHillSubharmonic
    else:
        StabClass = KoopmanHillProjection

    stabchanges = []
    last_stable = None

    for k, a in enumerate(a_values):
        ode.a = a
        hbm_ref = HBMEquation(
            ode,
            ode.omega,
            fourier_ref,
            initial_guess=np.zeros(2 * (2 * fourier_ref.N_HBM + 1)),
            stability_method=StabClass(fourier_ref, tol=1e-10),
        )
        hbm_ref.residual(update=True)
        hbm_ref.derivative("X", update=True)
        Phi_T_ref = hbm_ref.stability_method.fundamental_matrix(
            t_over_period=1, hbm=hbm_ref
        )
        FMs_ref = hbm_ref.stability_method.determine_eigenvalues(hbm_ref)
        stable_ref = hbm_ref.stability_criterion(FMs_ref)

        print(FMs_ref)

        print(f"{k}/{len(a_values)}: a = {a}")

        if k > 0 and last_stable != stable_ref:
            stabchanges.append((a + a_values[k - 1]) / 2)
            print("Stability change!")

        last_stable = stable_ref

        for N in trange(1, 150):

            # Set up and initialize Hill matrix
            fourier = fourier_ref.__replace__(N_HBM=N)
            hbm = HBMEquation(
                ode,
                ode.omega,
                fourier,
                initial_guess=np.zeros(2 * (2 * N + 1)),
                stability_method=StabClass(fourier, tol=1e-10),
            )
            hbm.residual(update=True)
            hbm.derivative("X", update=True)  # to initialize the Hill matrix correctly

            Phi_T = hbm.stability_method.fundamental_matrix(1, hbm)
            FMs = hbm.stability_method.determine_eigenvalues(hbm)
            stable = hbm.stability_criterion(FMs)

            # 0. guaranteed error
            if np.isnan(N_vals[0, k]) or np.isnan(N_vals[1, k]):
                E_opt = optimal_error_bound(hbm, t=T, subharmonic=subh)
                if E_opt < E_des and np.isnan(N_vals[0, k]):
                    N_vals[0, k] = N

            # 1. guaranteed stability
            if np.isnan(N_vals[1, k]):
                if stable:
                    to_check = real_axis
                else:
                    to_check = unit_circle

                if not does_pseudospectrum_include(Phi_T, E_opt, unit_circle):
                    N_vals[1, k] = N

            # 2.  Actual error
            if np.isnan(N_vals[2, k]):
                E_num = np.linalg.norm(Phi_T - Phi_T_ref)
                if E_num < E_des:
                    N_vals[2, k] = N

            # 3. Actual stability
            if np.isnan(N_vals[3, k]) and stable == stable_ref:
                N_vals[3, k] = N

            if not any(np.isnan(N_vals[:, k])):
                break

    plt.figure()
    plt.vlines(
        stabchanges,
        0,
        np.max(N_vals),
        linestyles="--",
        colors="gray",
        label="stability change",
    )
    plt.plot(a_values, N_vals[0, :], label=f"Guaranteed E < {E_des}")
    plt.plot(a_values, N_vals[1, :], label=f"Guaranteed stability info")
    plt.plot(a_values, N_vals[2, :], label=f"Numerical E < {E_des}")
    plt.plot(a_values, N_vals[3, :], label=f"Numerical stability info")

    plt.title(f"Mathieu equation b = {ode.b}, omega = {ode.omega}")
    plt.xlabel("a")
    plt.ylabel("b")
    plt.legend()

    tikzplotlib.save(f"mathieu_traversing_a_b_{ode.b}_omega_{ode.omega}.tex")


def plot_ince_strutt(ode, a_values, b_values, fourier, subh=True, logscale=True):

    unit_circle = np.exp(1j * np.linspace(0, 2 * np.pi, 250))
    real_axis = np.linspace(-1.1, 1.1, 250)

    if subh:
        stability_method = KoopmanHillSubharmonic(fourier, tol=1e-10)
    else:
        stability_method = KoopmanHillProjection(fourier, tol=1e-10)

    lambda_max = np.nan * np.ones((len(b_values), len(a_values)))
    E_opt = np.nan * np.ones_like(lambda_max)
    guaranteed = np.nan * np.ones_like(lambda_max)

    X = np.zeros(2 * (2 * fourier.N_HBM + 1))
    T = 2 * np.pi / ode.omega

    # Write header row with a_values if first b iteration
    with open("ince_strutt_FM.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(["b\\a"] + list(a_values))
    with open("ince_strutt_guaranty.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(["b\\a"] + list(a_values))
    with open("ince_strutt_E_opt.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(["b\\a"] + list(a_values))

    for idx_b, b in enumerate(b_values):
        ode.b = b
        print(f"{idx_b}/{len(b_values)}: b = {b}")
        for idx_a, a in enumerate(tqdm(a_values)):
            ode.a = a

            # Set up and initialize Hill matrix
            hbm = HBMEquation(
                ode,
                ode.omega,
                fourier,
                initial_guess=X,
                stability_method=stability_method,
            )
            hbm.residual(update=True)
            hbm.determine_stability(update=True)
            FMs = hbm.eigenvalues
            Phi_T = stability_method.fundamental_matrix(t_over_period=1, hbm=hbm)

            E_opt[idx_b, idx_a] = optimal_error_bound(hbm, t=T, subharmonic=subh)
            lambda_max[idx_b, idx_a] = np.max(np.abs(FMs))

            if hbm.stable:
                to_check = real_axis
            else:
                to_check = unit_circle
            guaranty = not does_pseudospectrum_include(
                Phi_T, E_opt[idx_b, idx_a], to_check
            )
            if guaranty:
                if hbm.stable:
                    guaranteed[idx_b, idx_a] = 1
                else:
                    guaranteed[idx_b, idx_a] = -1

        # After each b iteration, append the newly added row to a CSV file
        with open("ince_strutt_FM.csv", "a", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=";")
            writer.writerow([b] + list(lambda_max[idx_b, :]))

        with open("ince_strutt_guaranty.csv", "a", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=";")
            writer.writerow([b] + list(guaranteed[idx_b, :]))

        with open("ince_strutt_E_opt.csv", "a", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=";")
            writer.writerow([b] + list(guaranteed[idx_b, :]))

    shade = np.maximum(1, np.minimum(lambda_max, 10))  # np.maximum(0, magnitude_fm)

    if logscale:
        norm = "log"
    else:
        norm = "linear"

    # Show image
    fig, ax = plt.subplots()
    image = ax.pcolormesh(a_values, b_values, shade, cmap="gray_r", norm=norm)
    cbar = fig.colorbar(image, label="largest FM (magnitude)")

    # Plot overlay for guarantee
    cmap = ListedColormap(
        [
            (0, 0, 1, 0.5),  # blue (for 1)
            (0, 0, 0, 0),  # transparent (for 0 or nan)
            (1, 0, 0, 0.5),  # red (for -1)
        ]
    )
    ax.pcolormesh(a_values, b_values, guaranteed, cmap=cmap)

    ax.set_xlabel("$a$")
    ax.set_ylabel("$b$")
    cbar.set_ticks((1, 10), labels=("$\\leq 10^0$ (stable)", "$\\geq 10^1$"))

    tikzplotlib.save("ince-strutt.tex")
    return ax


if __name__ == "__main__":

    plot_near_PD()

    ode = MathieuODE(0, np.array([0, 0]), a=0.95, b=1.05, omega=1)
    # plot_N_over_a(ode, [1, 2, 3])

    a_values = np.linspace(0.995, 1.01, 400)
    b_values = np.linspace(0, 0.15, 600)
    fourier = Fourier(N_HBM=95, L_DFT=2048, n_dof=2)

    # plot_ince_strutt(ode, a_values, b_values, fourier=fourier, subh=False)

    plt.show()
