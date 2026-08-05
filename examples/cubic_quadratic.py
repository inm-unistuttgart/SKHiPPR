"""Demonstration / Template for creating a simple frequency response curve of a non-autonomous dynamical system with HBM.
This example uses an oscillator with quadratic and cubic nonlinearity, but ``ode`` can be instantiated as any non-autonomous :py:class:`~skhippr.odes.AbstractODE.AbstractODE`.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12
plt.rcParams["text.usetex"] = True
cm = 1 / 2.54  # cm in inches
plt.rcParams["figure.figsize"] = (7 * cm, 7 * cm)
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plt.cm.Dark2.colors)


# --- Fourier configuration ---
from skhippr.Fourier import Fourier

# --- Differential equation ---
from skhippr.odes.nonautonomous import CubicQuadratic

# --- HBM equation system ---
from skhippr.cycles.hbm import HBMEquation
from skhippr.equations.EquationSystem import EquationSystem

# --- Stability method ---
from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic

# --- Continuation ---
from skhippr.solvers.continuation import pseudo_arclength_continuator, BranchPoint

# --- Newton solver ---
from skhippr.solvers.newton import NewtonSolver


def main():
    """
    Demonstration for creating a simple frequency response curve of a non-autonomous dynamical system with HBM.

    This function performs the following steps:

    #. Creation of :py:class:`~skhippr.Fourier`, :py:class:`~skhippr.solvers.newton.NewtonSolver`, and :py:class:`~skhippr.stability.KoopmmanHillProjection.KoopmanHillSubharmonic` objects to collect method parameters.
    #. Instantiation of a :py:class:`~skhippr.odes.AbstractODE.AbstractODE` (here: :py:class:`~skhippr.odes.nonautonomous.Quadratic`) object which contains the ODE.
    #. Setup of an initial guess
    #. Setup and solution of the :py:class:`~skhippr.cycles.hbm.HBMEquation`, which formalizes the Harmonic Balance equations.
    #. Creation of an :py:class:`~skhippr.equations.EquationSystem.EquationSystem` containing only the HBM equations as input to the continuation method.
    #. Continuation of the frequency response curve using :py:func:`~skhippr.cycles.continuation.pseudo_arclength_continuator` and collecting the branch points
    #. Analyzing and plotting the branch points

    Returns
    -------

    None
    """

    # --- FFT, stability method and Newton solver configuration ---
    fourier = Fourier(N_HBM=25, L_DFT=300, n_dof=2, real_formulation=True)
    stability_method = KoopmanHillSubharmonic(fourier, tol=1e-4, autonomous=False)
    solver = NewtonSolver(verbose=True)

    gammas = np.linspace(0, 0.07, 15)

    all_freqs_stable = []
    all_freqs_unstbl = []
    all_amps = []

    beta = 0.35
    for gamma in gammas:

        # for beta in [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:

        # --- Instantiation of the ODE at initial point of branch ---
        ode = CubicQuadratic(
            t=0,
            x=[1.0, 0.0],
            alpha=1,
            beta=beta,
            gamma=gamma,
            delta=0.11,
            F=0.5,
            omega=0.01,
        )

        # --- Initial guess in time and frequency domain ---
        ts = fourier.time_samples(ode.omega)
        x0_samples = np.array(
            [np.cos(ode.omega * ts), -ode.omega * np.sin(ode.omega * ts)]
        )
        X0 = fourier.DFT(x0_samples)

        # --- Set up the Harmonic Balance equations
        hbm = HBMEquation(
            ode=ode,
            omega=ode.omega,
            fourier=fourier,
            initial_guess=X0,
            period_k=1,
            stability_method=stability_method,
        )

        # --- Solve initial HBM problem. ---
        # As hbm is here an AbstractEquation, we can use solver.solve_equation()
        # and specify the unknown variable explicitly.
        # solver.solve_equation() internally instantiates an EquationSystem.
        solver.solve_equation(equation=hbm, unknown="X")

        # --- Avoid that the solver prints a full report at each continuation step
        solver.verbose = False

        # --- Creation of EquationSystem ---
        # The solver needs an EquationSystem and not just an equation, so we have to create one.
        # Alternatively to this explicit creation, one can also instantiate a HBMSystem.
        initial_system = EquationSystem(
            equations=[hbm], unknowns=["X"], equation_determining_stability=hbm
        )

        # --- Preallocate the result of the FRC continuation ---
        # BranchPoints (a subclass of EquationSystem) extend the initial EquationSystem with one added equation for the tangency condition.
        frc: list[BranchPoint] = []

        # --- Iterate through the branch. Almost all arguments are optional. ---
        for branch_point in pseudo_arclength_continuator(
            initial_system=initial_system,
            solver=solver,
            stepsize=0.1,
            stepsize_range=(0.001, 0.1),
            continuation_parameter="omega",
            initial_direction=1,
            verbose=True,
            num_steps=4000,
        ):
            frc.append(branch_point)

            # break if omega exceeds maximum
            if branch_point.omega > 2.5:
                break

        # --- Analyze the branch. ---
        freqs = np.array([np.squeeze(point.omega) for point in frc])
        stable = np.array([point.stable for point in frc])

        # --- Determine maximum x_1 amplitude of every point on the branch. ---
        # the first equation of the BranchPoint equation system is the HBMequation, which provides x_time().
        amps = np.array([np.max(point.equations[0].x_time()[0, :]) for point in frc])

        # --- plot ---
        # plt.figure()

        freqs_stable = np.where(stable, freqs, np.nan)
        freqs_unstable = np.where(~stable, freqs, np.nan)

        all_freqs_stable.append(freqs_stable)
        all_freqs_unstbl.append(freqs_unstable)
        all_amps.append(amps)

        # plt.plot(freqs_stable, amps, "r", label="stable")
        # plt.plot(freqs_unstable, amps, "b--", label="unstable")

        # plt.xlabel("$\\omega$")
        # plt.ylabel("$|x_1|$")
        # plt.legend()
        # plt.title(
        #     f"FRC of nonlinear oscillator with stiffness ${ode.alpha}x + {ode.beta}x^2 + {ode.gamma}x^3$"
        # )

    fig_FM = plt.figure()
    phis = np.linspace(0, 2 * np.pi, 250)

    plt.plot(np.cos(phis), np.sin(phis), "k")

    for point in frc:
        if point.stable:
            col = "b."
        else:
            col = "r."

        plt.plot(np.real(point.eigenvalues), np.imag(point.eigenvalues), col)

    plt.title("Floquet multipliers")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for freqs, amps, gamma in zip(all_freqs_stable, all_amps, gammas):
        ax.plot(
            freqs,
            gamma * np.ones_like(freqs),
            amps,
            "r",
            label=f"stable",
        )

    for freqs, amps, gamma in zip(all_freqs_unstbl, all_amps, gammas):
        ax.plot(
            freqs,
            gamma * np.ones_like(freqs),
            amps,
            "b--",
            label=f"unstable",
        )

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_zlabel(r"$|x_1|$")
    ax.set_title(
        f"Frequency response curve \n  $ k(x) = {ode.alpha}x + {ode.beta}x + \\gamma x^2$"
    )
    # Only show unique labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())


if __name__ == "__main__":
    main()
    plt.show()
