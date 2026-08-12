"""

The :py:mod:`~skhippr.visualization.equilibria` module offers standardized functions for visualizing equilibria as well as equilibrium eigenvalues.
Supported equations are instances of classes implementing :py:class:`~skhippr.odes.AbstractODE.AbstractODE` as well as :py:class:`~skhippr.equations.EquationSystem.EquationSystem` instances that contain such an object.

It provides the functions:

* :py:func:`~skhippr.visualization.equilibria.plot_equilibrium` for plotting the equilibrium in a standard x-y plane
* :py:func:`~skhippr.visualization.equilibria.plot_eigenvalues` for making plots of the equation eigenvalues in the complex plane.

"""

import numpy as np
import matplotlib.pyplot as plt

from skhippr.odes.AbstractODE import AbstractODE
from skhippr.equations.EquationSystem import EquationSystem
from collections.abc import Sequence

from skhippr.visualization._helpers import (
    robust_plot,
    parse_and_generate_axis,
    extract_equation,
)


def plot_equilibrium(
    ode: AbstractODE | EquationSystem,
    ax=None,
    idx: Sequence[int] = [0, 1],
    **plot_kwargs,
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
    default_args = {
        "title": f"Equilibrium",
        "xlabel": f"x_{idx[0]}",
        "ylabel": f"x_{idx[1]}",
        "marker": "x",
    }

    kwargs = {**default_args, **plot_kwargs}

    # determine quantities to plot
    equation = extract_equation(ode, AbstractODE)

    # plot
    ax = parse_and_generate_axis(ax, **kwargs)
    robust_plot(ax.plot, equation.x[idx[0]], equation.x[idx[1]], **kwargs)

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
    default_args = {
        "title": "Floquet exponents",
        "xlabel": "Re($\\alpha$)",
        "ylabel": "Im($\\alpha$)",
        "marker": "x",
    }

    kwargs = {**default_args, **plot_kwargs}

    equation = extract_equation(ode, AbstractODE)
    eigenvalues = equation.eigenvalues

    new_axis = ax is None
    ax = parse_and_generate_axis(ax, **kwargs)

    # plot stability boundary
    if new_axis:
        ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0)

    robust_plot(
        ax.scatter,
        np.real(eigenvalues),
        np.imag(eigenvalues),
        **kwargs,
    )
    return ax
