"""

The :py:mod:`~skhippr.visualization.continuation` module provides a standardized function for visualizing :py:class:`collections.abc.Iterable` instances of :py:class:`skhippr.solvers.continuation.BranchPoint` objects that can be generated with the method :py:class:`~skhippr.solvers.continuation.pseudo_arclength_continuator` and then collected.

It provides the method :py:func:`~skhippr.visualization.continuation.plot_continuation` for making 1D plots over a continuation parameter, as well as 2D and 3D continuation plots.
"""

from collections.abc import Iterable, Callable
import matplotlib.pyplot as plt
import numpy as np

from skhippr.solvers.continuation import BranchPoint


def plot_continuation(
    branch: Iterable[BranchPoint],
    plot_fun: Callable[
        [BranchPoint], Iterable[float] | float
    ] = lambda bp: bp.vector_of_unknowns[0],
    ax=None,
    **plot_kwargs
):
    """
    Plot the numerical continuation results stored in a branch of :py:class:`~skhippr.solvers.continuation.BranchPoint` objects.
    This function visualizes one-parameter continuation data by plotting a scalar or vector measure defined by ``plot_fun`` against the continuation parameter (or iteration index if no parameter is defined).
    If stability information is available for all :py:class:`~skhippr.solvers.continuation.BranchPoint` objects contained in ``branch``, stable segments will be drawn in red, unstable ones in blue. Otherwise, the entire branch is colored black.

    The plot automatically adapts to one-, two-, or three-dimensional measures returned by ``plot_fun``.

    Parameters
    ----------
    branch : Iterable[BranchPoint]
        A :py:class:`collections.abc.Iterable` object containing :py:class:`~skhippr.solvers.continuation.BranchPoint` instances.
    plot_fun : Callable[[BranchPoint], Iterable[float] | float], optional
        A function that maps each :py:class:`~skhippr.solvers.continuation.BranchPoint` object to a plot measure.
        Must return values accepted by :py:func:`~skhippr.visualization.continuation._to_array`. This means, ``plot_fun`` must return values that ``np.squeeze(element)`` squeezes to the same length for each :py:class:`~skhippr.solvers.continuation.BranchPoint`.
        The default is ``lambda bp: bp.unknowns[0][0]``, which is the value of the first unknown of the first equation.
    ax : matplotlib.axes.Axes | mpl_toolkits.mplot3d.axes3d.Axes3D, optional
        The :py:class:`~matplotlib.axes.Axes` or :py:class:`~mpl_toolkits.mplot3d.axes3d.Axes3D` object on which to plot. If ``plot_fun`` returns a vector of length 3, the given ``ax`` must be an instance of the class :py:class:`mpl_toolkits.mplot3d.axes3d.Axes3D`. If ``None``, a new instance will be created.
    **plot_kwargs
        Additional keyword arguments passed to ``ax.plot()`` or ``ax.plot3D``.
        Axes labels (``xlabel``, ``ylabel``, ``zlabel``), plot title (``title``) and the colors reflecting stability (``stable_color``, ``unstable_color``, ``color``) of the :py:class:`~skhippr.solvers.continuation.BranchPoint` can all be passed additionaly.


    Returns
    -------
    ax : matplotlib.axes.Axes
        The :py:class:`~matplotlib.axes.Axes` object containing the continuation diagram.

    Notes
    -----
    - For one-dimensional measures (``plot_fun`` returns scalars, 0D arrays with ``shape`` = ``()`` or single-element 1D arrays with ``shape`` = ``(1,)`` for each :py:class:`~skhippr.solvers.continuation.BranchPoint`), the continuation parameter (if available) defines the x-axis, otherwise the branch index is used
    - For two- or three-dimensional measures, the ``plot_fun`` output defines the plotting coordinates directly.
    - An example for a ``plot_fun`` with non-homogenous outputs that all squeeze to the same length can be found in :py:func:`examples.duffing_3d.plot_3D_frc`.
    """
    branch_list = list(branch)

    stability_defined = True
    for bp in branch_list:
        if (
            getattr(bp, "equation_determining_stability", None) is None
            or bp.equation_determining_stability.stability_method is None
        ):
            stability_defined = False
            break

    values, dim = _get_values_and_dimension(branch_list, plot_fun)
    if dim not in (1, 2, 3):
        raise ValueError(
            "plot_fun must return a scalar or a numpy array of size 2 or 3."
        )

    generated_ax = False
    if ax is None:
        if dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        else:
            _, ax = plt.subplots(1, 1)
        generated_ax = True

    parameter = branch_list[0].equations[-1].continuation_parameter
    if parameter is not None:
        parameter_values = np.array(
            [np.squeeze(getattr(bp, parameter)) for bp in branch_list]
        )
    else:
        parameter_values = np.arange(len(branch_list))

    if stability_defined:
        stable_flags = np.array([bp.stable for bp in branch_list])
    else:
        stable_flags = np.full(len(branch_list), True)

    stable_col = plot_kwargs.pop("stable_color", "r" if stability_defined else "k")
    stable_col = plot_kwargs.pop("color", stable_col)
    unstable_col = plot_kwargs.pop("unstbl_col", "b" if stability_defined else "k")

    title = plot_kwargs.pop("title", "Continuation Plot")
    xlabel = plot_kwargs.pop("xlabel", parameter if dim == 1 else "measure[0]")
    ylabel = plot_kwargs.pop("ylabel", "measure[1]")
    zlabel = plot_kwargs.pop("zlabel", "measure[2]")

    if dim == 1:
        xs, ys = parameter_values, values[:, 0]
    elif dim == 2:
        xs, ys = values[:, 0], values[:, 1]
    elif dim == 3:
        xs, ys, zs = values[:, 0], values[:, 1], values[:, 2]
    xs_stable = np.where(stable_flags, xs, np.nan)
    xs_unstable = np.where(~stable_flags, xs, np.nan)

    if dim == 3:
        ax.plot3D(xs_stable, ys, zs, color=stable_col, label="stable", **plot_kwargs)
        ax.plot3D(
            xs_unstable, ys, zs, color=unstable_col, label="unstable", **plot_kwargs
        )
    else:
        ax.plot(xs_stable, ys, color=stable_col, label="stable", **plot_kwargs)
        ax.plot(xs_unstable, ys, color=unstable_col, label="unstable", **plot_kwargs)

    if generated_ax:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if dim == 3:
            ax.set_zlabel(zlabel)
        if stability_defined:
            ax.legend(loc="best")

    return ax


def _to_array(value: float | np.ndarray | Iterable):
    """
    Convert different numeric inputs into a flat :py:class:`numpy.ndarray`.
    This function handles scalars, arrays, lists and tuples by squeezing each element individually when iterating over containers.
    It succeeds when all elements in ``value`` squeeze to the same length using ``np.squeeze(element)``.

    Parameters
    ----------
    value: float | np.ndarray | Iterable
        Input to be converted to 1D array. It accepts:

        - Scalars and arrays with dimension 0: ``1.0`` or ``1`` as well as ``np.array(1.0)``
        - Single-element containers: ``[1.0]``, ``(1.0,)``, ``np.array([1.0])``, ``np.array([[1.0]])``
        - Containers where all first level elements contained squeeze to the same length: ``[[1.0]]``, ``[(1.0,)]``, ``1.0``.

    Returns
    --------
    arr : numpy.ndarray
        1D :py:class:`numpy.ndarray` of shape ``(n,)`` where ``n`` is the common squeezed length of all input elements.

    Notes
    -----
    ``value`` may contain different combinations of each for each element, as long as `np.squeeze(element)` always returns the same shape.

    """
    if np.isscalar(value):
        return np.atleast_1d(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return np.atleast_1d(value.item())
        return np.ravel(value)
    return np.array([np.squeeze(v) for v in value])


def _get_values_and_dimension(
    branch: Iterable[BranchPoint],
    plot_fun: Callable[[BranchPoint], object],
):
    """
    Evaluate ``plot_fun`` across branch points and validate consistent dimensionality

    Parameters
    ----------
    branch : Iterable[BranchPoint]
        Continuation branch containing instances of class :py:class:`~skhippr.solvers.continuation.BranchPoint`.
    plot_fun : Callable[[BranchPoint], object]
        Measure function returning values accepted by :py:func:`~skhippr.visualization.continuation._to_array`.

    Returns
    -------
    values : numpy.ndarray
        Shape ``(n_points, dim)`` array of normalized measure values
    dim : int {1,2,3}
        Dimensionality of measures returned by ``plot_fun``.
    """
    branch = list(branch)
    value_list = [_to_array(plot_fun(bp)) for bp in branch]
    dim = value_list[0].size
    return np.array(value_list), dim
