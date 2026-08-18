"""Van der Pol oscillator: continuation w.r.t. nu and animation of the resulting phase portrait."""

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

from skhippr.odes.AbstractODE import AbstractODE


from examples.vanderpol_minimal import compute_frc

# Visualization
from skhippr.visualization.cycles import (
    plot_phase,
    animate_period,
    animate_floquet_multipliers,
    animate_phase,
)
from skhippr.visualization.continuation import plot_continuation
from skhippr.visualization.data_export import save_animation


def main():
    """Demonstration of the continuation of the Van der Pol oscillator w.r.t. nu and animation of the resulting phase portrait.
    This function performs the following steps:

    #. Setup of the :py:class:`~skhippr.odes.autonomous.Vanderpol` ODE
    #. Setup of a :py:class:`~skhippr.cycles.hbm.HBMSystem` object with the :py:class:`~skhippr.odes.autonomous.Vanderpol` ode and a :py:class:`~skhippr.Fourier.Fourier` object, encoding both the HBM equations and the phase anchor.
    #. Continuation of the HBM system w.r.t. nu using the :py:func:`~skhippr.solvers.continuation.pseudo_arclength_continuator` and a :py:class:`~skhippr.solvers.newton.NewtonSolver`
    #. Analysis of the resulting branch of solutions, extracting time series, amplitudes, Floquet multipliers, and stability
    #. Visualization of the results, including an animation of the phase portrait and Floquet multipliers, as well as plots of amplitude and frequency w.r.t. nu
    #. Saving the animation by passing a relative path as a string.
    """

    hbm, branch = compute_frc(nu_range=(0, 10), max_stepsize=0.5)

    # --- Create animations from the HBMEquations in the continuation branch ---
    _, animation0 = animate_phase(branch, scaling=False)
    _, animation1 = animate_period(branch, scaling=True)
    _, animation2 = animate_floquet_multipliers(
        branch, show_unit_circle=True, scaling=False
    )
    plot_continuation(
        branch,
        plot_fun=lambda point: point.omega,
        xlabel="nu",
        ylabel="omega",
        title="Van der Pol continuation",
    )

    # --- Export animations ---
    # Animations can be saved as a .gif and as video files such as .mp4.
    # Video formats require the user to have FFmpeg installed.
    # save_animation(animation0, "plots/vanderpol_animations/phase_animation.gif")

    # And, finally, a pretty phase plot
    ax = None
    for k, bp in enumerate(branch):
        ax = plot_phase(bp, ax=ax, label=None, color=plt.cm.viridis(k / len(branch)))

    return animation0, animation1, animation2


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
    animations = main()
    # plot_vanderpol_for_diss()
    plt.show()
