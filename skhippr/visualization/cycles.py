"""

The :py:mod:`~skhippr.visualization.cycles` module provides standardized functions for visualizing limit cycles over multiple periods, their phase portraits as well as Floquet multipliers and exponents.

Supported equations are instances of :py:class:`~skhippr.cycles.hbm.HBMEquation` or :py:class:`~skhippr.cycles.hbm.HBMSystem` that contain such an equation.

It provides the functions :py:func:`~skhippr.visualization.cycles.plot_period` for plotting the time series of the equation solution,
:py:func:`~skhippr.visualization.cycles.plot_phase` for making phase portraits and :py:func:`~skhippr.visualization.cycles.plot_floquet_multipliers` as well as :py:func:`~skhippr.visualization.cycles.plot_floquet_exponents` for visualizing the floquet multipliers and exponents of a cycle.
"""

import numpy as np
import matplotlib.pyplot as plt

from skhippr.cycles.hbm import HBMEquation
from skhippr.cycles.hbm import HBMSystem
from collections.abc import Sequence


__all__ = [
    "plot_period",
    "plot_phase",
    "plot_floquet_multipliers",
    "plot_floquet_exponents",
]


def plot_period(
    hbm: HBMEquation | HBMSystem, ax=None, idx=0, n_periods: float = 1.0, **plot_kwargs
):
    """
    Plot the time series of a solved :py:class:`~skhippr.cycles.hbm.HBMEquation` over a given number of periods.

    Parameters
    ----------
    hbm : HBMEquation or HBMSystem
        The equation or equation system containing the solution. If it is of type :py:class:`~skhippr.cycles.hbm.HBMSystem`, the first valid :py:class:`~skhippr.cycles.hbm.HBMEquation` instance contained is used.
    ax : matplotlib.axes.Axes, optional
        The :py:class:`~matplotlib.axes.Axes` object on which to plot. If ``None``, an :py:class:`~matplotlib.axes.Axes` instance will be created.
    idx : int, optional
        The index of the state to be plotted.
    n_periods: float, optional
        The number of periods for which the time series is plotted. May be non-integer.
    **plot_kwargs
        Additional keyword arguments passed to ``ax.plot()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The :py:class:`~matplotlib.axes.Axes` object with the plotted period response.

    """
    generated_ax = False
    if ax is None:
        _, ax = plt.subplots(1, 1)
        generated_ax = True
    equation = _get_equation_helper(hbm)
    fourier = equation.fourier
    omega = equation.omega_solution
    x_time = equation.x_time()
    x_single_period = x_time[idx, :]
    t = fourier.time_samples(omega, n_periods)
    n = int(np.ceil(n_periods))
    x = np.tile(x_single_period, n)[: t.size]
    ax.plot(t, x, **plot_kwargs)
    if generated_ax:
        if n_periods == 1.0:
            ax.set_title("Time series over one period")
        else:
            ax.set_title(f"Time series over {n_periods} periods")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
    return ax


def plot_phase(
    hbm: HBMEquation | HBMSystem, ax=None, idx: Sequence[int] = [0, 1], **plot_kwargs
):
    """
    Plot the phase of a solved :py:class:`~skhippr.cycles.hbm.HBMEquation`.

    Parameters
    ----------
    hbm : HBMEquation or HBMSystem
        The equation or equation system containing the solution. If it is of type :py:class:`~skhippr.cycles.hbm.HBMSystem`, the first valid :py:class:`~skhippr.cycles.hbm.HBMEquation` instance contained is used.
    ax : matplotlib.axes.Axes, optional
        The :py:class:`~matplotlib.axes.Axes` object on which to plot. If None, a new :py:class:`~matplotlib.axes.Axes` instance will be created.
    idx : Sequence[int], optional
        Exactly two indices of the states to be considered, for example [0,1] for standard x-y phase plane
    **plot_kwargs
        Additional keyword arguments passed to ``ax.plot()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The :py:class:`~matplotlib.axes.Axes` object with the phase plot.
    """
    generated_ax = False
    if ax is None:
        _, ax = plt.subplots(1, 1)
        generated_ax = True
    equation = _get_equation_helper(hbm)
    x_time = equation.x_time()
    ax.plot(x_time[idx[0], :], x_time[idx[1], :], **plot_kwargs)
    if generated_ax:
        ax.set_title("Phase plot of solution")
        ax.set_xlabel("x_0")
        ax.set_ylabel("x_1")
    return ax


def plot_floquet_multipliers(hbm: HBMEquation | HBMSystem, ax=None, **plot_kwargs):
    """
    Plot the Floquet multipliers of the :py:class:`~skhippr.cycles.hbm.HBMEquation` solution.

    Parameters
    ----------
    hbm : HBMEquation or HBMSystem
        The equation or equation system containing the solution. If it is of type :py:class:`~skhippr.cycles.hbm.HBMSystem`, the first valid :py:class:`~skhippr.cycles.hbm.HBMEquation` instance contained is used.
    ax : matplotlib.axes.Axes, optional
        The :py:class:`~matplotlib.axes.Axes` object on which to plot. If ``None``, a new :py:class:`~matplotlib.axes.Axes` instance will be created.
    **plot_kwargs
        Additional keyword arguments passed to ``ax.plot()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The :py:class:`~matplotlib.axes.Axes` object with a scatter plot of the Floquet multipliers on the complex plane. A unit circle is also plotted, if no axes is given in the call.
    """
    generated_ax = False
    if ax is None:
        _, ax = plt.subplots(1, 1)
        generated_ax = True
    if "marker" not in plot_kwargs:
        plot_kwargs["marker"] = "x"

    equation = _get_equation_helper(hbm)
    floquet_multipliers = equation.eigenvalues
    fourier = equation.fourier
    ax.scatter(
        np.real(floquet_multipliers),
        np.imag(floquet_multipliers),
        label="$\\lambda$",
        **plot_kwargs,
    )
    if generated_ax:
        ax.set_title("Floquet multipliers")
        ax.set_xlabel("Re($\\lambda$)")
        ax.set_ylabel("Im($\\lambda$)")
        ax.set_aspect("equal")
        ax.legend(loc="best")
        ax.plot(
            np.cos(fourier.time_samples_normalized),
            np.sin(fourier.time_samples_normalized),
            "k",
        )
    return ax


def plot_floquet_exponents(hbm: HBMEquation | HBMSystem, ax=None, **plot_kwargs):
    """
    Calculate and plot the Floquet exponents of the :py:class:`~skhippr.cycles.hbm.HBMEquation` solution from the Floquet multipliers.

    Parameters
    ----------
    hbm : HBMEquation or HBMSystem
        The equation or equation system containing the solution. If it is of type :py:class:`~skhippr.cycles.hbm.HBMSystem`, the first valid :py:class:`~skhippr.cycles.hbm.HBMEquation` instance contained is used.
    ax : matplotlib.axes.Axes, optional
        The :py:class:`~matplotlib.axes.Axes` object on which to plot. If ``None``, a new :py:class:`~matplotlib.axes.Axes` instance will be created.
    **plot_kwargs
        Additional keyword arguments passed to ``ax.plot()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The :py:class:`~matplotlib.axes.Axes` object with a scatter plot of the Floquet exponents on the imaginary plane.
    """
    generated_ax = False
    if ax is None:
        _, ax = plt.subplots(1, 1)
        generated_ax = True
    if "marker" not in plot_kwargs:
        plot_kwargs["marker"] = "x"

    equation = _get_equation_helper(hbm)
    floquet_multipliers = equation.eigenvalues
    lambdas = np.asarray(floquet_multipliers)
    floquet_exponents = np.log(lambdas) / equation.T_solution
    ax.scatter(
        np.real(floquet_exponents),
        np.imag(floquet_exponents),
        label="$\\alpha$",
        **plot_kwargs,
    )
    if generated_ax:
        ax.set_title("Floquet exponents")
        ax.set_xlabel("Re($\\alpha$)")
        ax.set_ylabel("Im($\\alpha$)")
        ax.legend(loc="best")
        ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0)
    return ax


def _get_equation_helper(hbm: HBMEquation | HBMSystem):
    """
    Helper function that returns a :py:class:`~skhippr.cycles.hbm.HBMEquation`.
    Parameters
    ----------
    hbm : HBMEquation or HBMSystem
        A equation or system of equations.

    Returns
    -------
    equation : HBMEquation
        The first valid :py:class:`~skhippr.cycles.hbm.HBMEquation` found.

    Raises
    ------
    ValueError
        If 'hbm' does not contain any usable :py:class:`~skhippr.cycles.hbm.HBMEquation` instance.
    """
    if isinstance(hbm, HBMEquation):
        return hbm
    if isinstance(hbm, HBMSystem) and hasattr(hbm, "equations"):
        for equation in hbm.equations:
            if isinstance(equation, HBMEquation):
                return equation
    raise ValueError("hbm does not contain any usable HBMEquation instance")
