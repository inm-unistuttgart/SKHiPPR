"""
Demonstration of convergence of Koopman-Hill projection method for the Mathieu equation.

This script analyzes the convergence of the Koopman-Hill projection stability method applied to the :py:class:`~skhippr.odes.ltp.MathieuODE` and :py:class:`~skhippr.odes.ltp.SmoothedMeissner` equations around their  equilibrium at zero.

The convergence is studied by increasing the number of harmonics (N_HBM) in the Harmonic Balance Method (HBM) and comparing the resulting monodromy matrix to a reference solution obtained via the shooting method.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

from skhippr.solvers.newton import NewtonSolver
from skhippr.odes.ltp import MathieuODE, SmoothedMeissner
from skhippr.Fourier import Fourier
from skhippr.cycles.hbm import HBMSystem
from skhippr.cycles.shooting import ShootingSystem, ShootingBVP
from skhippr.stability.KoopmanHillProjection import KoopmanHillProjection


def analyze_N_mathieu(N_max=60, csv_path=None):
    """Analyze the convergence of the Koopman-Hill projection method for the Mathieu equation. Optionally, results can be saved to a csv file."""
    mathieu = MathieuODE(
        t=0, x=np.array([0.0, 0.0]), a=4, b=0.2, omega=1, damping=0.005
    )
    solver = NewtonSolver(verbose=False)
    if csv_path:
        initialize_csv(csv_path, N_max=N_max, key_param=None)
    analyze_N_convergence(solver=solver, ode=mathieu, N_max=N_max, csv_path=csv_path)


def analyze_smoothness_effects(N_max=30, csv_path=None, smoothing=None):
    """
    Analyze and visualize the effects of the smoothing parameter on the convergence of a Hill-type system.

    For ``smoothing`` ==1, the :py:class:`~skhippr.odes.ltp.SmoothedMeissner` ODE coincides with Mathieu's equation and for ``smoothing`` ==0 with the Meissner equation.
    This function iterates over a range of smoothing parameter values, runs a convergence analysis for each,
    and plots the corresponding excitation and the convergence behavior. Optionally, results can be saved to a CSV file.

    Parameters
    ----------

    N_max : int, optional
        Maximum number of harmonics to consider in the convergence analysis (default is 30).
    csv_path : str or None, optional
        Path to a CSV file where results will be saved. If None, results are not saved (default is None).
    smoothing : array-like or None, optional
        Sequence of smoothing parameter values to analyze. If None, a default sequence from 1 to 0 is used.
    """
    solver = NewtonSolver()
    fourier_ref = Fourier(N_HBM=1, L_DFT=1026, n_dof=2, real_formulation=True)
    if smoothing is None:
        smoothing = np.flip(np.linspace(0, 1, 7, endpoint=True))

    ode = SmoothedMeissner(
        0, np.array([0.0, 0.0]), smoothing[0], a=4, b=0.2, omega=1, damping=0.005
    )
    T = 2 * np.pi / ode.omega

    if csv_path:
        initialize_csv(csv_path, N_max=N_max, key_param="smoothing")

    params_plot = dict()
    ax_conv, axs, _ = setup_plot(ax=None, lambdas_ref=0)
    ax_time = axs[1]
    ax_time.clear()
    ax_time.set_aspect("auto")
    ax_time.set_xlim((0, 0.5))
    ax_time.set_ylim(ode.a - 1.2 * ode.b, ode.a + 1.2 * ode.b)
    ax_time.set_xlabel("t/T")

    for k, eps in enumerate(smoothing):
        params_plot["color"] = f"C{k}"
        params_plot["label"] = f"$\\epsilon = {eps}$"
        ode.smoothing = eps
        hbm = analyze_N_convergence(
            solver=solver,
            ode=ode,
            fourier_ref=fourier_ref,
            Phi_T_ref=None,
            N_max=N_max,
            params_plot=params_plot,
            ax_conv=ax_conv,
            csv_path=csv_path,
            parameter="smoothing",
        )

        # Plot the considered function and the identified samples
        ts = fourier_ref.time_samples_normalized

        g = -ode.a - ode.b * ode.g_fcn(ts)

        J_time = hbm.equations[0].ode_samples()
        ax_time.plot(ts / T, g, "k-")
        ax_time.plot(ts / T, -J_time[1, 0, :].squeeze(), ".", **params_plot)

    ax_time.legend()
    ax_conv.legend()


def analyze_N_convergence(
    solver,
    ode,
    fourier_ref=None,
    Phi_T_ref=None,
    N_max=10,
    params_plot=None,
    ax_conv=None,
    csv_path=None,
    parameter=None,
    stability_method_class=KoopmanHillProjection,
) -> HBMSystem:
    """
    Analyze the convergence of the Koopman-Hill projection with increasing number of harmonics (N_HBM)
    in the context of a Hill-type problem.
    Parameters
    ----------
    f : callable
        The function defining the system's equations. Must have an equilibrium at zero.
    params : dict
        Dictionary of parameters required by the system.
    Phi_T_ref : ndarray, optional
        Reference monodromy matrix for comparison. If None, it is computed using shooting.
    N_max : int, optional
        Maximum number of harmonics to consider (default is 10).
    params_plot : dict, optional
        Dictionary of plotting parameters (e.g., color, marker style).
    ax : matplotlib.axes.Axes, optional
        Matplotlib axis to plot on. If None, a new axis is created.
    csv_path : str, optional
        Path to a CSV file where errors will be saved. If None, errors are not saved.
    key_param : str, optional
        Key of the parameter to track in the error output.
    Returns
    -------
    hbm: The HBM problem with N = N_max
    Side effects
    ------------
        The function plots the convergence of the monodromy matrix error and optionally saves errors to a CSV file.
    """

    # Setup
    T = 2 * np.pi / ode.omega

    if params_plot is None:
        params_plot = {"color": "k"}

    if fourier_ref is None:
        fourier_ref = Fourier(
            N_HBM=1, L_DFT=1024, n_dof=ode.n_dof, real_formulation=True
        )

    if Phi_T_ref is None:
        Phi_T_ref, sol_shoot = reference_monodromy_matrix(solver, ode)
        x_t = sol_shoot.equations[0].x_time(fourier_ref.time_samples(ode.omega))
    else:
        sol_shoot = None
        x_t = None

    # print("HACK!!!!")
    # Phi_T_ref = np.array(
    #     [
    #         [1.000022938650409, -0.005883294959915],
    #         [-0.007797981796352, 1.000022938650409],
    #     ]
    # )

    FMs_ref = np.linalg.eig(Phi_T_ref).eigenvalues

    # FMs_ref = np.array([1.006796255932213, 0.993249621368605])
    # FMs_ref = np.array([0, 0])

    ax_conv, axs, plot_FMs = setup_plot(ax_conv, FMs_ref)

    errors = initialize_errors_with_param(ode, parameter, csv_path)

    for N_HBM in range(1, N_max + 1):

        print(N_HBM)
        hbm_sys = setup_hbm_system(
            solver=solver,
            ode=ode,
            fourier_ref=fourier_ref,
            N_HBM=N_HBM,
            x_t=x_t,
            stability_method_class=stability_method_class,
        )

        FMs = hbm_sys.eigenvalues
        Phi_T = hbm_sys.equations[0].stability_method.fundamental_matrix(
            t_over_period=1, hbm=hbm_sys.equations[0]
        )

        # errors.append(np.linalg.norm(Phi_T - Phi_T_ref, ord=2))
        errors.append(
            np.minimum(
                np.linalg.norm(FMs - FMs_ref), np.linalg.norm(np.flip(FMs) - FMs_ref)
            )
        )
        ax_conv.semilogy(N_HBM, errors[-1], ".", **params_plot)

        if "label" in params_plot:
            del params_plot["label"]  # to ensure that only one dot appears in legend

        if plot_FMs:
            axs[1].plot(np.real(FMs), np.imag(FMs), ".")
        pass

    if csv_path:
        write_errors_to_csv(csv_path, errors)

    return hbm_sys


def reference_monodromy_matrix(solver, ode):
    """
    Computes the reference monodromy matrix for a given system with an equilibrium at zero using the shooting method.

    This function sets up and solves the shooting problem for a parametrically excited system of ODEs defined by `f`
    with the specified parameters and period `T`.
    It asserts that 0 is indeed a solution. The function returns the monodromy matrix.

    Args:
        f (callable): The system of ODEs to solve. Should accept state, time, and parameters.
        params (dict): Parameters to pass to the ODE system.
        T (float): The period over which to solve the shooting problem.
        variable (str): state variable of f.
        initial_guess(np.ndarray): Initial guess for point on periodic solution.
                                   If not passed, the function assumes that there exists an equilibrium at 0.
        autonomous(bool): Whether the considered system is autonomous.

    Returns:
        numpy.ndarray: The reference fundamental matrix (2x2 array).
        sol_shoot (ShootingProblem): The whole shooting problem

    Raises:
        AssertionError: If 0 is not an equilibrium and no initial guess was passed.
    """

    sys_shooting = ShootingSystem(ode, T=2 * np.pi / ode.omega, atol=1e-12, rtol=1e-12)

    solver.solve(sys_shooting)
    assert sys_shooting.solved

    bvp = sys_shooting.equations[0]

    monodromy_matrix = bvp.derivative(variable="x", update=True) + np.eye(bvp.ode.n_dof)

    return monodromy_matrix, sys_shooting


def setup_plot(ax, lambdas_ref):
    """
    Sets up a matplotlib plot for visualizing reference eigenvalues and optionally Floquet multipliers.
    If ax is None, a new figure with two subplots is created. Plot_FMs is set to True.
    Otherwise, ax is simply returned and plot_FMs is set to False.

    Parameters:
        ax (matplotlib.axes.Axes or None): The axis to plot on. If None, a new figure with two subplots is created.
        lambdas_ref (array-like): Reference eigenvalues to be plotted in the complex plane.

    Returns:
        tuple:
            - ax (matplotlib.axes.Axes): The axis to be used for further plotting.
            - axs (array-like or None): The array of axes if new subplots were created, otherwise None.
            - plot_FMs (bool): Flag indicating whether Floquet multipliers should be plotted.
    """
    if ax is None:
        _, axs = plt.subplots(ncols=2, nrows=1, gridspec_kw={"wspace": 0.4})
        phis = np.linspace(0, 2 * np.pi, 250)
        axs[1].plot(np.cos(phis), np.sin(phis), "--", color="gray")
        axs[1].plot(np.real(lambdas_ref), np.imag(lambdas_ref), "kx")
        axs[1].axis("equal")
        axs[1].set_xlabel("Re($\\lambda$)")
        axs[1].set_ylabel("Im($\\lambda$)")
        del phis
        plot_FMs = True
        ax = axs[0]
        ax.set_ylabel("E")
        ax.set_xlabel("N")
    else:
        axs = None
        plot_FMs = False
    return ax, axs, plot_FMs


def initialize_errors_with_param(ode, parameter, csv_path):
    """
    Initializes a list of errors.
    If a CSV path is provided, the first column of the CSV file corresponds to different parameter values. Then, the list is initialized with that value.
    Otherwise, the list is initialized empty.

    Args:
        params (dict): Dictionary containing configuration parameters. May include the key "smoothing".
        csv_path (str or None): Path to a CSV file. If provided (not None or empty), the smoothing parameter is used.

    Returns:
        list: A list containing the smoothing value from params if csv_path is provided; otherwise, an empty list.
    """
    if csv_path and parameter is not None:
        return [getattr(ode, parameter, 0)]
    else:
        return []


def setup_hbm_system(
    solver,
    ode,
    fourier_ref,
    N_HBM,
    x_t=None,
    stability_method_class=KoopmanHillProjection,
):
    """
    Sets up and solves a Harmonic Balance Method (HBM) problem for a given function and parameters. The function must have an equilibrium at 0.

    Parameters:
        f (callable): The function representing the system to be analyzed. Must have an equilibrium at 0.
        params (dict): Dictionary containing problem parameters, must include "omega".
        N_HBM (int): Number of harmonics to use in the HBM.

    Returns:
        HBMProblem: The solved HBMProblem instance.

    Raises:
        AssertionError: If the HBM problem does not converge or the solution is not 0
    """
    fourier = Fourier(
        N_HBM=N_HBM,
        L_DFT=fourier_ref.L_DFT,
        n_dof=fourier_ref.n_dof,
        real_formulation=fourier_ref.real_formulation,
    )
    if x_t is None:
        x_t = np.zeros((fourier.n_dof, fourier.L_DFT))

    X_init = fourier.DFT(x_t)

    hbm = HBMSystem(
        ode=ode,
        omega=ode.omega,
        fourier=fourier,
        initial_guess=X_init,
        stability_method=stability_method_class(fourier),
    )

    solver.solve(hbm)
    assert hbm.solved

    return hbm


def write_errors_to_csv(csv_path, errors):
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(errors)


def initialize_csv(csv_path, N_max, key_param=None):
    header = list(range(N_max))
    if key_param is not None:
        header = [key_param] + header

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(header)


if __name__ == "__main__":
    analyze_N_mathieu(N_max=10, csv_path="data_mathieu.csv")
    vals_smoothing = np.insert(np.logspace(-3, 0, 4, endpoint=True), 0, 0)
    analyze_smoothness_effects(
        N_max=70,
        csv_path="data_meissner_smoothed.csv",
        smoothing=vals_smoothing,
        stability_method_class=KoopmanHillProjection,
    )
    plt.show()
