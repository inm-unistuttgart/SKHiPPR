"""Van der Pol oscillator: continuation w.r.t. nu and animation of the resulting phase portrait."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import tikzplotlib

# ODE
from skhippr.odes.autonomous import Vanderpol

# FFT configuration
from skhippr.Fourier import Fourier

# HBM and stability
from skhippr.cycles.hbm import HBMSystem
from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic

# Solution procedure
from skhippr.solvers.newton import NewtonSolver
from skhippr.solvers.continuation import pseudo_arclength_continuator

# only for type hinting
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.odes.AbstractODE import AbstractODE
from skhippr.solvers.continuation import BranchPoint

from skhippr.visualization.cycles import plot_phase, plot_floquet_multipliers

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.size"] = 12
# plt.rcParams["text.usetex"] = True
# cm = 1 / 2.54  # cm in inches
# plt.rcParams["figure.figsize"] = (7 * cm, 7 * cm)
# plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plt.cm.Dark2.colors)


def main():
    """Demonstration of the continuation of the Van der Pol oscillator w.r.t. nu and animation of the resulting phase portrait.
    This function performs the following steps:

    #. Setup of the :py:class:`~skhippr.odes.autonomous.Vanderpol` ODE
    #. Setup of a :py:class:`~skhippr.cycles.hbm.HBMSystem` object with the :py:class:`~skhippr.odes.autonomous.Vanderpol` ode and a :py:class:`~skhippr.Fourier.Fourier` object, encoding both the HBM equations and the phase anchor.
    #. Continuation of the HBM system w.r.t. nu using the :py:func:`~skhippr.solvers.continuation.pseudo_arclength_continuator` and a :py:class:`~skhippr.solvers.newton.NewtonSolver`
    #. Analysis of the resulting branch of solutions, extracting time series, amplitudes, Floquet multipliers, and stability
    #. Visualization of the results, including an animation of the phase portrait and Floquet multipliers, as well as plots of amplitude and frequency w.r.t. nu
    """

    print("Van der Pol oscillator: continuation w.r.t. nu")

    # setup
    newton_solver = NewtonSolver(verbose=True)
    ode = Vanderpol(x=[2.0, 0.0], nu=0.1)
    hbm_system: EquationSystem = setup_hbm_system(ode, newton_solver)

    # continuation
    branch: list[BranchPoint] = []
    nu_range = (ode.nu, 6)
    newton_solver.verbose = False

    for branch_point in pseudo_arclength_continuator(
        initial_system=hbm_system,
        solver=newton_solver,
        stepsize=0.1,
        stepsize_range=(0.001, 0.4),
        initial_direction=1,
        num_steps=1000,
        continuation_parameter="nu",
        verbose=True,
    ):
        branch.append(branch_point)
        if not nu_range[0] <= branch_point.nu <= nu_range[1]:
            break

    # analysis
    xs_time, amplitudes, nus, stable, FMs, omegas = parse_branch(branch)

    # visualization
    animation = animate_phase_portrait_and_FMs(nus, xs_time, FMs)
    plot_with_stability(nus, amplitudes, stable, "$\\nu$", "$|x_1|$")
    plot_with_stability(nus, omegas, stable, "$\\nu$", "$\\omega$")

    plot_phases(branch)

    return animation


def plot_phases(branch):

    ax = None
    for k, bp in enumerate(branch):
        ratio = k / len(branch)
        col = (ratio, 0.2, 1 - ratio)
        ax = plot_phase(hbm=bp, ax=ax, idx=[0, 1], color=col)
    ax.set_title("Van der Pol oscillator")

    tikzplotlib.save("plots/vanderpol.tikz")


def setup_hbm_system(ode: AbstractODE, solver: NewtonSolver = None, N_HBM=45):

    omega_0 = 1
    fourier = Fourier(N_HBM=N_HBM, L_DFT=1000, n_dof=ode.n_dof, real_formulation=True)
    stability_method = KoopmanHillSubharmonic(fourier=fourier, tol=1e-4)
    X0 = generate_initial_condition(fourier, omega_0)

    hbm_system = HBMSystem(ode, omega_0, fourier, X0, stability_method=stability_method)
    if solver:
        solver.solve(hbm_system)
        print(
            f"Initial problem convergence: {hbm_system.solved}. omega = {hbm_system.omega}"
        )
    return hbm_system


def generate_initial_condition(fourier, omega_0):
    ts = fourier.time_samples(omega_0)
    x0_samples = np.vstack(
        (2 * np.cos(omega_0 * ts), -2 * omega_0 * np.sin(omega_0 * ts))
    )
    X0 = fourier.DFT(x0_samples)
    return X0


def parse_branch(branch: list[BranchPoint]):
    xs_time = [point.equations[0].x_time() for point in branch]
    amplitudes = np.array([np.max(x_time[0, :]) for x_time in xs_time])
    stable = np.array([point.stable for point in branch])
    nus = np.array([np.squeeze(point.nu) for point in branch])
    floquet_multipliers = [point.eigenvalues for point in branch]
    omegas = [np.squeeze(point.omega) for point in branch]

    return xs_time, amplitudes, nus, stable, floquet_multipliers, omegas


def plot_with_stability(x_values, y_values, stable, xlabel, ylabel):

    x_stable = np.where(stable, x_values, np.nan)
    x_unstable = np.where(~stable, x_values, np.nan)

    plt.figure()
    plt.plot(x_stable, y_values, "r-", label="stable")
    plt.plot(x_unstable, y_values, "b--", label="unstable")
    # plt.title(f"Amplitude of first DOF -- {solver}")
    plt.legend()
    plt.xlabel("$\\nu$")
    plt.ylabel(ylabel)


def animate_phase_portrait_and_FMs(nus, xs_time, floquet_multipliers):

    fig_anim, axs = plt.subplots(nrows=1, ncols=2)

    # Initially populate the plots
    (line_period,) = axs[0].plot(xs_time[0][0, :], xs_time[0][1, :])
    axs[0].axis("equal")
    title_phase = axs[0].set_title("Phase portrait")

    (plot_floquet_multiplier,) = axs[1].plot(
        np.real(floquet_multipliers[0]),
        np.imag(floquet_multipliers[0]),
        "x",
    )
    phis = np.linspace(0, 2 * np.pi)
    axs[1].plot(np.cos(phis), np.sin(phis), "k-")
    axs[1].axis("equal")
    title_FM = axs[1].set_title("Floquet multipliers")

    # Animation function
    def update(frame):
        line_period.set_xdata(xs_time[frame][0, :])
        line_period.set_ydata(xs_time[frame][1, :])
        plot_floquet_multiplier.set_xdata(np.real(floquet_multipliers[frame]))
        plot_floquet_multiplier.set_ydata(np.imag(floquet_multipliers[frame]))
        title_phase.set_text(f"Phase portrait: $\\nu$ = {nus[frame]:.2f}")
        title_FM.set_text(f"Floquet multipliers: $\\nu$ = {nus[frame]:.2f}")
        return (line_period, plot_floquet_multiplier, title_phase, title_FM)

    animation = FuncAnimation(
        fig=fig_anim, func=update, frames=len(xs_time), interval=20
    )
    return animation, xs_time


def plot_vanderpol_for_diss():
    nus = [1.0, 5.5]
    systems = []

    # setup
    solver = NewtonSolver(verbose=False)
    ax_phase, ax_FM_before = (
        None,
        None,
    )

    _, ax_FM_after = plt.subplots(1, 1)
    phis = np.linspace(0, 2 * np.pi)
    ax_FM_after.plot(np.cos(phis), np.sin(phis), color="gray")
    ax_FM_after.set_aspect("equal")
    ax_FM_after.set_xlabel("Re")
    ax_FM_after.set_ylabel("Im")

    nus_all = []
    errors = [[], [], [], []]

    hbm_system = setup_hbm_system(
        Vanderpol(x=np.array([2.0, 0]), nu=nus[0], t=0), solver=solver
    )
    for branch_point in pseudo_arclength_continuator(
        initial_system=hbm_system,
        solver=solver,
        stepsize=0.1,
        stepsize_range=(0.001, 0.1),
        initial_direction=1,
        num_steps=1000,
        continuation_parameter="nu",
        verbose=True,
    ):
        nus_all.append(branch_point.nu)

        # FOP multiplier is lambdas[1]
        lambdas = np.sort(branch_point.eigenvalues)
        errors[0].append(np.abs(lambdas[1] - 1))

        # Obtain monodromy matrix
        Phi_T = branch_point.equations[0].stability_method.fundamental_matrix(
            t_over_period=1, hbm=branch_point.equations[0]
        )
        x0 = branch_point.equations[0].x_time()[:, 0]
        v = branch_point.equations[0].ode.dynamics(x=x0)
        errors[1].append(np.linalg.norm(v - Phi_T @ v))

        # Wielandt deflation
        Phi_shift = Phi_T - ((v[:, np.newaxis] * v[np.newaxis, :]) / (sum(v * v)))
        lambdas_after, _ = np.linalg.eig(Phi_shift)

        # Sort to have least-magnitude FM at index 1
        idx_sort = np.lexsort((np.angle(lambdas_after), np.abs(lambdas_after)))
        lambdas_after = lambdas_after[idx_sort[::-1]]

        errors[2].append(np.abs(lambdas_after[1]))
        errors[3].append(np.abs(lambdas[0] - lambdas_after[0]))

        # Plot if desired

        if branch_point.nu > nus[0]:
            nus.pop(0)
            print(f"nu = {branch_point.nu}")

            # Phase plot
            ax_phase = plot_phase(branch_point, ax_phase, label=f"nu={branch_point.nu}")

            # FMs before
            ax_FM_before = plot_floquet_multipliers(branch_point, ax=ax_FM_before)
            print(branch_point.eigenvalues)

            # FMs after
            ax_FM_after.plot(np.real(lambdas_after), np.imag(lambdas_after), "x")

            if len(nus) == 0:
                break

    ax_phase.legend()

    _, ax_nu = plt.subplots(1, 1)
    ax_nu.semilogy(nus_all, errors[0], label="lambda_1 - 1")
    ax_nu.semilogy(nus_all, errors[1], label="eigvec error")
    ax_nu.semilogy(nus_all, errors[2], label="lambda_0")
    ax_nu.semilogy(nus_all, errors[3], label="lambda_2 - lambda_2,shift")
    ax_nu.legend()

    for k in range(4):
        tikzplotlib.save(f"plots/vanderpol_{k}.tikz")
        plt.close()


if __name__ == "__main__":
    # animation = main()
    main()
    # plot_vanderpol_for_diss()
    plt.show()
