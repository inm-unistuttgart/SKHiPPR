"""

The :py:mod:`~skhippr.visualization.equilibria` module offers standardized functions for visualizing equilibria as well as equilibrium eigenvalues.
Supported equations are instances of classes implementing :py:class:`~skhippr.odes.AbstractODE.AbstractODE` as well as :py:class:`~skhippr.equations.EquationSystem.EquationSystem` instances that contain such an object.

It provides the function :py:func:`~skhippr.visualization.equilibria.plot_equilibrium` for plotting the equilibrium in a standard x-y plane
and :py:func:`~skhippr.visualization.equilibria.plot_eigenvalues` for making plots of the equation eigenvalues in the complex plane.
"""

import numpy as np
import matplotlib.pyplot as plt

from skhippr.odes.AbstractODE import AbstractODE
from skhippr.equations.EquationSystem import EquationSystem
from collections.abc import Sequence


def plot_equilibrium(
    ode: AbstractODE | EquationSystem,
    ax=None,
    idx: Sequence[int] = [0, 1],
    **plot_kwargs
):
    """
    Plot the equilibrium of an ordinary differential equation.

    Parameters
    ----------
    ode : AbstractODE or EquationSystem
        The ordinary differential equation or an equation system containing it.
    ax : matplotlib.axes.Axes, optional
        The :py:class:`~matplotlib.axes.Axes` object on which to plot. If ``None``, a new :py:class:`~matplotlib.axes.Axes` instance will be created.
    idx : Sequence[int], optional
        Exactly two indices of the states to be considered
    **plot_kwargs
        Additional keyword arguments passed to ``ax.plot()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The :py:class:`~matplotlib.axes.Axes` object with the equilibrium.
    """
    generated_ax = False
    if ax is None:
        _, ax = plt.subplots(1, 1)
        generated_ax = True
    if "marker" not in plot_kwargs:
        plot_kwargs["marker"] = "x"

    equation = _get_equation_helper(ode=ode)
    x = np.asarray(equation.x)
    ax.plot(x[idx[0]], x[idx[1]], **plot_kwargs)
    if generated_ax:
        ax.set_title("Equilibria")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
    return ax


def plot_eigenvalues(ode: AbstractODE | EquationSystem, ax=None, **plot_kwargs):
    """
    Plot the eigenvalues of an ordinary differential equation. If no :py:class:`matplotlib.axes.Axes` object is passed, the function wil also plot the imaginary axis.

    Parameters
    ----------
    ode : AbstractODE or EquationSystem
        The ordinary differential equation or an equation system containing it.
    ax : matplotlib.axes.Axes, optional
        The :py:class:`~matplotlib.axes.Axes` object on which to plot. If ``None``, a new :py:class:`~matplotlib.axes.Axes` instance will be created.
    **plot_kwargs
        Additional keyword arguments passed to ``ax.plot()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The :py:class:`~matplotlib.axes.Axes` object with the the plotted eigenvalues.
    """
    generated_ax = False
    if ax is None:
        _, ax = plt.subplots(1, 1)
        generated_ax = True
    if "marker" not in plot_kwargs:
        plot_kwargs["marker"] = "x"
    equation = _get_equation_helper(ode=ode)
    eigenvalues = np.asarray(equation.eigenvalues)
    ax.scatter(
        np.real(eigenvalues), np.imag(eigenvalues), label="eigenvalues", **plot_kwargs
    )
    if generated_ax:
        ax.set_title("Equilibria eigenvalues")
        ax.set_xlabel("Re($\\lambda$)")
        ax.set_ylabel("Im($\\lambda$)")
        ax.legend(loc="best")
        ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0)
    return ax


def _get_equation_helper(ode: AbstractODE | EquationSystem):
    """
    A helper function that returns an :py:class:`~skhippr.odes.AbstractODE.AbstractODE`.
    Parameters
    ----------
    ode : AbstractODE or EquationSystem
        An equation or system of equations.

    Returns
    -------
    equation : AbstractODE
        The first valid :py:class:`~skhippr.odes.AbstractODE.AbstractODE` found.

    Raises
    ------
    ValueError
        If ``ode`` does not contain any usable :py:class:`~skhippr.odes.AbstractODE.AbstractODE` instance.
    """
    if isinstance(ode, AbstractODE):
        return ode
    if isinstance(ode, EquationSystem) and hasattr(ode, "equations"):
        for equation in ode.equations:
            if isinstance(equation, AbstractODE):
                return equation
    raise ValueError("hbm does not contain any usable AbstractODE instance")
