"""

The :py:mod:`~skhippr.visualization.continuation` module provides a standardized function for visualizing :py:class:`collections.abc.Iterable`s of :py:class:`skhippr.solvers.continuation.BranchPoint` objects that can be generated with the method :py:class:`~skhippr.solvers.continuation.pseudo_arclength_continuator` and then collected.

It provides the method :py:func:`~skhippr.visualization.continuation.plot_continuation` which plots continuation data over a user given continuation parameter.
"""

from collections.abc import Iterable
from typing import Callable

from skhippr.solvers.continuation import BranchPoint

import matplotlib.pyplot as plt
import numpy as np


def plot_continuation(
    branch: Iterable[BranchPoint],
    plot_fun : Callable[[BranchPoint], np.ndarray | float] = lambda bp: bp.vector_of_unknowns[0],
    ax = None,
    **plot_kwargs
    ):
    """
    Plot a scalar measure of each :py:class:`~skhippr.solvers.continuation.BranchPoint` instance in a `branch` :py:class:`~collections.abc.Iterable` over a user given continuation parameter.
    If stability information is available for all :py:class:`~skhippr.solvers.continuation.BranchPoint` objects, stable intervals will be plotted in red and unstable intervals in blue.
    If any :py:class:`~skhippr.solvers.continuation.BranchPoint` does not have stability information, everything will be plotted in black.

    Parameters
    ----------
    branch : Iterable[BranchPoint]
        A :py:class:`collections.abc.Iterable` object containing :py:class:`~skhippr.solvers.continuation.BranchPoint` instances.
    plot_fun : Callable[[BranchPoint], float], optional
        Function that maps each :py:class:`~skhippr.solvers.continuation.BranchPoint` object to a scalar value to be plotted on the y-axis. The default is ``lambda bp: bp.unknowns[0][0]``, which is the value of the first unknown of the first equation.
    ax : matplotlib.axes.Axes, optional
        The :py:class:`~matplotlib.axes.Axes` object on which to plot. If ``None``, an :py:class:`~matplotlib.axes.Axes` instance will be created.
    **plot_kwargs
        Additional keyword arguments passed to ``ax.plot()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The :py:class:`~matplotlib.axes.Axes` object with the plotted period response.
    """
    branch_list = list(branch)

    stability_defined = all(
        getattr(bp, "equation_determining_stability", None) is not None
        for bp in branch_list
    )

    value_list, dim = _get_values_and_dimension(branch_list, plot_fun)

    if dim not in (1,2,3):
        raise ValueError("plot_fun must return a scalar or a numpy array of size 2 or 3.")

    generated_ax = False
    if ax is None:
        if dim in (1,2):
            _, ax = plt.subplots(1, 1)
            generated_ax = True
        else: 
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            
    parameter = branch_list[0].equations[-1].continuation_parameter
    
    if dim == 1:
        if parameter is None:
            raise ValueError("Continuation parameter cannot be None when plot_fun returns a scalar")
        else:
            param_values = np.array([np.squeeze(getattr(bp, parameter)) for bp in branch_list]) 
            measure_values = np.array([value[0] for value in value_list])
        
        if not stability_defined:
            ax.plot(param_values, measure_values, color="k", **plot_kwargs)
        else: 
            stable_flags = np.array([bp.stable for bp in branch_list])
            stable_parameters = np.where(stable_flags, param_values, np.nan)
            unstable_parameters = np.where(~stable_flags, param_values, np.nan)
            ax.plot(stable_parameters, measure_values, color="r", label ="stable",**plot_kwargs)
            ax.plot(unstable_parameters, measure_values, color="b", label="unstable", **plot_kwargs)

    if dim == 2:
        xs = np.array([value[0] for value in value_list])
        ys = np.array([value[1] for value in value_list])
        if not stability_defined:
            ax.plot(xs, ys, color="k", **plot_kwargs)
        else:
            stable_flags = np.array([bp.stable for bp in branch_list])
            xs_stable = np.where(stable_flags, xs, np.nan)
            xs_unstable = np.where(~stable_flags, xs, np.nan)
            ax.plot(xs_stable, ys, color="r", label ="stable",**plot_kwargs)
            ax.plot(xs_unstable, ys, color="b", label="unstable", **plot_kwargs)

    if dim == 3:
        xs = np.array([value[0] for value in value_list])
        ys = np.array([value[1] for value in value_list])
        zs = np.array([value[2] for value in value_list])
        if not stability_defined:
            ax.plot3D(xs, ys, zs, color="k", **plot_kwargs)
        else:
            stable_flags = np.array([bp.stable for bp in branch_list])
            xs_stable = np.where(stable_flags, xs, np.nan)
            xs_unstable = np.where(~stable_flags, xs, np.nan)
            ax.plot3D(xs_stable, ys, zs, color="r", label ="stable",**plot_kwargs)
            ax.plot3D(xs_unstable, ys, zs, color="b", label="unstable", **plot_kwargs)
    if generated_ax:
        ax.set_title("Continuation Plot")
        if dim == 1:
            ax.set_xlabel(str(parameter))
            ax.set_ylabel("measure")
        elif dim in (2,3):
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            if dim == 3:
                ax.set_zlabel("z")
        if stability_defined:
            ax.legend(loc="best")
    return ax

def _to_array(value):
    if np.isscalar(value):
        return np.array([value])
    value = np.asarray(value)
    return value

def _get_values_and_dimension(
    branch: Iterable[BranchPoint],
    plot_fun : Callable[[BranchPoint], np.ndarray | float]
    ):
    values = [plot_fun(bp) for bp in branch]
    value_list = [_to_array(value) for value in values]
    dim = value_list[0].size
    return value_list, dim

def _plot_from_dimension(
    value_list: list[np.ndarray],
    dim: int, 
    parameter: str
    ):
    pass