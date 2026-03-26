"""
Analyze oscillations of the Duffing oscillator using SKHiPPR by performing
continuation along both excitation frequency and excitation amplitude.

"""

from collections.abc import Iterable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple # for legend label grouping
from tqdm import tqdm  # for the progress bar

# --- Fourier configuration ---
from skhippr.Fourier import Fourier

# --- System function ---
from skhippr.odes.nonautonomous import Duffing

# --- HBM solver ---
from skhippr.cycles.hbm import HBMSystem

# --- Stability method ---
from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic

# --- Continuation ---
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.solvers.continuation import pseudo_arclength_continuator, BranchPoint
from skhippr.solvers.newton import NewtonSolver


# --- Visualization ---
from skhippr.visualization.cycles import (
    plot_floquet_multipliers,
    plot_floquet_exponents,
    plot_phase
)
from skhippr.visualization.continuation import plot_continuation
from skhippr.visualization.data_export import save_tikz, save_pdf

def main():
    """
    Runs a demonstration for analyzing the Duffing oscillator with HBM, to illustrate SKHiPPR usage.

    This function performs the following steps:

    #. Parameter setup.
    #. Creation of a :py:class:`~skhippr.Fourier` object to collect FFT parameters.
    #. Creation of a :py:class:`~skhippr.stability.KoopmanHillProjection.KoopmanHillProjection` object which defines the stability method. Other :py:class:`~skhippr.stability._StabilityHBM` subclasses can interchangeably be used, such as:

        * :py:class:`~skhippr.stability.KoopmanHillProjection.KoopmanHillSubharmonic`
        * :py:class:`~skhippr.stability.ClassicalHill.ClassicalHill`
        * :py:class:`~skhippr.stability.SinglePass.SinglePassRK`

    #. Setup and solution of the initial :py:class:`~skhippr.cycles.hbm.HBMSystem`.
    #. Visualization of the initial solution and its stability properties.
    #. Continuation of the first force response curve.
    #. Generation of frequency response curves starting from points on the previous force response.
    #. Generation of additional force responses starting from points on the frequency response curve.
    #. 3D plotting of all computed response curves, including stability information.

    """

    # --- FFT, stability method and Newton solver configuration ---
    fourier = Fourier(N_HBM=25, L_DFT=300, n_dof=2, real_formulation=True)
    stability_method = KoopmanHillSubharmonic(fourier, tol=1e-4)
    solver = NewtonSolver(verbose=True)

    # --- Continuation domain ---
    omega_min, omega_max = 0.8, 12
    omegas_spark = [0.3, 0.8, 1.1, 1.7, 4]
    F_min, F_max = 0.05, 20
    Fs_spark = [1, 5, 10, 17]

    # --- Instantiation of the ODE at initial point ---
    ode = Duffing(
        t=0, x=[1.0, 0.0], alpha=1, beta=2, delta=0.16, F=F_min, omega=omega_min
    )

    # --- Initial guess in time and frequency domain ---
    ts = fourier.time_samples(omega_min)
    x0_samples = np.array([np.cos(ts * omega_min), -omega_min * np.sin(ts * omega_min)])
    X0 = fourier.DFT(x0_samples)

    # --- HBM equation system setup ---
    hbm_sys = HBMSystem(
        ode=ode,
        omega=ode.omega,
        fourier=fourier,
        initial_guess=X0,
        stability_method=stability_method,
    )

    # --- Optional: solve initial point and visualize ---
    solver.solve(hbm_sys)
    assert hbm_sys.solved
    visualize_solution(hbm_sys)

    # --- Force response continuation ---
    solver.verbose = False
    response_F = initial_force_response(solver, hbm_sys, F_min, F_max, verbose=True)

    # --- Frequency response curves from force response points ---
    responses_omega = continue_from_continuation_curve(
        solver=solver,
        curve=response_F,
        new_cont_param="omega",
        param_range=(omega_min, omega_max),
        values_spark=Fs_spark,
    )

    # --- Additional force responses from frequency response points ---
    responses_F = continue_from_continuation_curve(
        solver,
        responses_omega[0],
        new_cont_param="F",
        param_range=(F_min, F_max),
        values_spark=omegas_spark,
    )

    # --- Plot results ---
    ax = plot_all_responses(
        responses_omega,
        stable_label = "stable $\omega$ responses",
        unstable_label="unstable $\omega$ responses"
        )
    
    ax = plot_all_responses(
        responses_F,
        ax=ax,
        stable_color = "k",
        unstable_color = "y",
        stable_label = "stable F responses",
        unstable_label = "unstable F responses"
        )


def initial_force_response(
    solver, initial_system: HBMSystem, F_min: float, F_max: float, verbose=True
) -> list[BranchPoint]:
    """
    Continue the force response curve starting from the initial solution.
    """

    response_F = []
    for branch_point_F in pseudo_arclength_continuator(
        initial_system=initial_system,
        solver=solver,
        stepsize=0.1,
        stepsize_range=(0.01, 0.5),
        initial_direction=1,
        continuation_parameter="F",
        verbose=verbose,
    ):
        response_F.append(branch_point_F)
        if branch_point_F.F > F_max:
            break
    return response_F


def continue_from_continuation_curve(
    solver: NewtonSolver,
    curve: list[BranchPoint],
    new_cont_param: str,
    param_range: tuple[float, float],
    values_spark: list[float],
) -> list[list[BranchPoint]]:
    """
    Finds and computes frequency response curves emanating from specified points on a force response curve.

    This function iterates over a given continuation curve (list of :py:class:`~skhippr.cycles.continuation.BranchPoint` objects). When the continuation parameter first exceeds the next value in ``values_spark``, the method it initiates a new continuation along the different parameter ``key_cont`` (e.g., ``"omega"``). The resulting branches are collected and returned as a list of lists of :py:class:`~skhippr.cycles.continuation.BranchPoint` objects.

    Parameters
    ----------
    solver : NewtonSolver
        The solver to use for continuation.
    curve : list[BranchPoint]
        The initial continuation curve.
    new_cont_param : str
        The name of the new continuation parameter (e.g., `'"omega"'`).
    param_range : tuple[float, float]
        The minimum and maximum values for the (new) continuation parameter.
    values_spark : list[float]
        The values of the (old) continuation parameter of the initial curve at which to start new continuation curves along the new parameter.

    Returns
    -------
    responses: list[list[BranchPoint]]
        A list containing the new continuation curves.
    """
    responses = []
    old_cont_param = curve[0].unknowns[-1]
    param_min, param_max = param_range

    for branch_point_init in curve:
        if not values_spark:
            break

        value_old_param = getattr(branch_point_init, old_cont_param)

        if value_old_param > values_spark[0]:
            values_spark.pop(0)
            responses.append([])

            # Create EquationSystem from the branch point, removing the previous anchor equation.
            initial_system = EquationSystem(
                equations=branch_point_init.equations[:-1],
                unknowns=branch_point_init.unknowns[:-1],
                equation_determining_stability=branch_point_init.equation_determining_stability,
            )

            # setup progress bar
            pbf = f"{new_cont_param} curve at {old_cont_param} = {np.squeeze(value_old_param):5.2f}: {new_cont_param} = {{n:5.2f}} |{{bar}}| "
            with tqdm(total=param_max, bar_format=pbf) as progress_bar:

                # do the continuation
                for branch_point in pseudo_arclength_continuator(
                    initial_system,
                    solver=solver,
                    stepsize=0.1,
                    stepsize_range=(0.01, 0.25),
                    initial_direction=1,
                    continuation_parameter=new_cont_param,
                    verbose=False,
                    num_steps=10000,
                ):
                    progress_bar.n = np.squeeze(getattr(branch_point, new_cont_param))
                    progress_bar.refresh()
                    responses[-1].append(branch_point)
                    if not (
                        param_min <= getattr(branch_point, new_cont_param) <= param_max
                    ):
                        break
    return responses

def plot_all_responses(
    responses: list[list[BranchPoint]],
    ax = None,
    **plot_kwargs
):
    """
    Plot a 3D curve of branch points with stability information.

    Visualize a list of continuation branches in 3D space, where the axes represent
    the frequency (``omega``), forcing amplitude (``F``), and the maximum absolute value of the first
    state variable (``|x_1|``). 
    
    Illustrates creating a 3D ``plot_fun`` that can be passed to :py:func:`~skhippr.visualization.continuation.plot_continuation`.

    Parameters
    ----------
    responses : Iterable[Iterable[BranchPoint]]
        An :py:class:`collections.abc.Iterable" of several continuation curves to plot. Every :py:class:`~skhippr.cycles.continuation.BranchPoint` must have the attributes ``point.F`` and ``point.omega``.
    ax : matplotlib.axes._subplots.Axes3DSubplot, optional
        Existing 3D axes to plot on. If ``None``, a new figure and axes are created. Defaults to ``None``.
    **plot_kwargs
        Additional keyword arguments passed to ``plot_continuation``.

    Returns
    -------
    ax: matplotlib.axes._subplots.Axes3DSubplot
        The 3D axes with the plotted data.
    """
    def plot_fun(point: BranchPoint) -> np.ndarray:
        omega = point.equations[0].omega
        F = point.equations[0].F
        amplitude = np.max(np.abs(point.equations[0].x_time()[0, :]))
        
        # --- Returns the shapes: ``(1,)``, ``(1,)``, ``()`` --- 
        # This is a valid configuration for ``plot_continuation``, since ``np.squeeze(element)``would return the same shape for each.
        # Another functional output would be: ``return (omega, F, amplitude)``.
        return omega, F, amplitude
    
    for response in responses:
        ax = plot_continuation(
        branch = response, 
        plot_fun = plot_fun,
        ax = ax,
        **plot_kwargs, 
        xlabel= "omega",
        ylabel="F",
        zlabel="|x_1|"
        )
    ax.set_title("Responses")
    return ax

def visualize_solution(system: HBMSystem):
    """
    Visualizes and analyzes properties of one solved :py:class:`~skhippr.cycles.hbm.HBMSystem`.

    This function generates two subplots:

    #. A phase plot of the solution, plotting the first two state variables over time.
    #. A plot of the Floquet multipliers in the complex plane, along with the unit circle for reference.

    Parameters
    ----------
    system : HBMSystem
        The :py:class:`~skhippr.cycles.hbm.HBMSystem` object containing the problem formulation and the solution.

    Returns
    -------
        None
    """

    _, axs = plt.subplots(nrows=1, ncols=3)
    axs[0] = plot_phase(hbm = system, ax = axs[0])
    axs[0].set_title("Phase plot of solution")
    axs[0].set_ylabel("x_1")
    axs[0].set_xlabel("x_0")
    fourier = system.equations[0].fourier
    axs[1] = plot_floquet_multipliers(hbm = system, ax = axs[1])
    axs[1].set_title("Floquet multipliers")
    axs[1].plot(
        np.cos(fourier.time_samples_normalized),
        np.sin(fourier.time_samples_normalized),
        "k",
    )
    axs[1].axis("equal")
    axs[2] = plot_floquet_exponents(hbm = system, ax = axs[2])
    axs[2].set_title("Floquet exponents")

if __name__ == "__main__":
    main()
    plt.show()
