"""

The :py:mod:`~skhippr.visualization.continuation` module provides a standardized function for visualizing :py:class:`collections.abc.Iterable` instances of :py:class:`skhippr.solvers.continuation.BranchPoint` objects that can be generated with the method :py:class:`~skhippr.solvers.continuation.pseudo_arclength_continuator` and then collected.

It provides the method

* :py:func:`~skhippr.visualization.continuation.plot_continuation`

for making 1D plots over a continuation parameter, as well as 2D and 3D continuation plots.

As well as the functions:

* :py:func:`~skhippr.visualization.continuation.plot_floquet_multiplier_continuation`
* :py:func:`~skhippr.visualization.continuation.plot_floquet_exponent_continuation`

for plotting the Floquet multipliers and exponents over the continuation parameter.

"""

from collections.abc import Iterable, Callable
import matplotlib.pyplot as plt
import numpy as np

from skhippr.solvers.continuation import BranchPoint

from skhippr.visualization._helpers import (
    parse_and_generate_axis,
    robust_plot,
    extract_equation,
)


def plot_continuation(
    branch: Iterable[BranchPoint],
    plot_fun: Callable[
        [BranchPoint], Iterable[float] | float
    ] = lambda bp: bp.vector_of_unknowns[0],
    ax=None,
    clean_legend=True,
    **plot_kwargs,
):
    """
    Plot the numerical continuation results stored in an :py:class:`~collections.abc.Iterable` of :py:class:`~skhippr.solvers.continuation.BranchPoint` objects.

    This function visualizes one-parameter continuation by applying ``plot_fun`` to each :py:class:`~skhippr.solvers.continuation.BranchPoint`.
    The dimensionality of the data returned by ``plot_fun`` determines how the results are plotted:

    #. If ``plot_fun`` returns one element per :py:class:`~skhippr.solvers.continuation.BranchPoint`, the values are plotted against the continuation parameter.
    #. If ``plot_fun`` returns two or three elements per :py:class:`~skhippr.solvers.continuation.BranchPoint`, the returned values are interpreted directly as plotting coordinates.

    For each :py:class:`~skhippr.solvers.continuation.BranchPoint`, ``plot_fun`` must return either a scalar or a :py:class:`~collections.abc.Sequence` of scalar-like values (e.g. ``return a``, ``return a, b`` or ``return a, b, c``) where each value is reducible to a single numerical value via ``np.squeeze``.

    If stability information is available for all :py:class:`~skhippr.solvers.continuation.BranchPoint` objects contained in ``branch``, stable segments will be drawn in red, unstable ones in blue. Otherwise, the entire branch is colored black.

    The plot automatically adapts to one-, two-, or three-dimensional measures returned by ``plot_fun``.

    Parameters
    ----------
    branch : Iterable[BranchPoint]
        A :py:class:`collections.abc.Iterable` object containing :py:class:`~skhippr.solvers.continuation.BranchPoint` instances.
    plot_fun : Callable[[BranchPoint], Iterable[float] | float], optional
        A function that maps each :py:class:`~skhippr.solvers.continuation.BranchPoint` object to a plot measure.
        Must return values accepted by :py:func:`~skhippr.visualization.continuation._to_array`. This means, ``plot_fun(:py:class:`~skhippr.solvers.continuation.BranchPoint`)`` must return values that ``np.squeeze(element)`` squeezes to the same length for each :py:class:`~skhippr.solvers.continuation.BranchPoint`.
        The default is ``lambda bp: bp.unknowns[0][0]``, which is the value of the first unknown of the first equation.
    ax : matplotlib.axes.Axes | mpl_toolkits.mplot3d.axes3d.Axes3D, optional
        The :py:class:`~matplotlib.axes.Axes` or :py:class:`~mpl_toolkits.mplot3d.axes3d.Axes3D` object on which to plot. If ``plot_fun`` returns a vector of length 3, the given ``ax`` must be an instance of the class :py:class:`mpl_toolkits.mplot3d.axes3d.Axes3D`. If ``None``, a new instance will be created.
    clean_legend: bool, optional
        Remove duplicate legend entries. Useful when plotting several continuation branches in one :py:class:`~matplotlib.axes.Axes` object.
    **plot_kwargs
        Additional keyword arguments passed to ``ax.plot()`` or ``ax.plot3D``.
        Axes labels (``xlabel``, ``ylabel``, ``zlabel``), legend labels (``label``), plot title (``title``) and the colors reflecting stability (``stable_color``, ``unstable_color``, ``color``) of the :py:class:`~skhippr.solvers.continuation.BranchPoint` can all be passed additionaly.


    Returns
    -------
    ax : matplotlib.axes.Axes
        The :py:class:`~matplotlib.axes.Axes` object containing the continuation diagram.

    Notes
    -----
    - When making plots without a continuation parameter, ``plot_fun`` must return the plotting coordinates explicitly.
    - An example for a ``plot_fun`` with outputs of different shapes that all squeeze to the same length can be found in :py:func:`examples.duffing_3d.plot_all_responses`.
    """

    default_kwargs = {
        "title": "Continuation Plot",
        "xlabel": "measure[0]",
        "ylabel": "measure[1]",
        "zlabel": "measure[2]",
        "stable_color": "r",
        "unstable_color": "b",
        "color": "k",
        "stable_label": "stable",
        "unstable_label": "unstable",
    }

    values, branch_list = eval_plot_fun(branch, plot_fun)

    # analyze dimension
    ndim = values.shape[1]
    if ndim not in (1, 2, 3):
        raise ValueError(
            "plot_fun must return a scalar or a numpy array of size 2 or 3."
        )

    if ndim == 1:
        # add parameter values as x axis
        parameter_values = np.array([bp.vector_of_unknowns[-1] for bp in branch_list])
        values = np.concatenate([parameter_values[:, np.newaxis], values], axis=1)
        default_kwargs["xlabel"] = branch_list[0].unknowns[-1]

    plot_kwargs = {**default_kwargs, **plot_kwargs}

    # handle stability

    bp = branch_list[0]
    stability_defined = not (
        getattr(bp, "equation_determining_stability", None) is None
        or bp.equation_determining_stability.stability_method is None
        or bp.stable is None
    )

    if stability_defined:
        stable_flags = np.array([bp.stable for bp in branch_list])
    else:
        stable_flags = np.full(len(branch_list), True)
        plot_kwargs["stable_color"] = plot_kwargs["color"]
        plot_kwargs["stable_label"] = ""

    vals_stable = np.where(stable_flags[:, np.newaxis], values, np.nan)
    vals_unstable = np.where(~stable_flags[:, np.newaxis], values, np.nan)

    ax = parse_and_generate_axis(ax, ndim=ndim, **plot_kwargs)
    if ndim == 3:
        plot_method = ax.plot3D
    else:
        plot_method = ax.plot

    plot_kwargs["color"] = plot_kwargs.pop("stable_color")
    plot_kwargs["label"] = plot_kwargs.pop("stable_label")
    robust_plot(plot_method, *vals_stable.T, **plot_kwargs)

    plot_kwargs["color"] = plot_kwargs.pop("unstable_color")
    plot_kwargs["label"] = plot_kwargs.pop("unstable_label")
    robust_plot(plot_method, *vals_unstable.T, **plot_kwargs)

    if stability_defined:
        ax.legend()
    if clean_legend:
        _deduplicate_legend(ax)
    return ax


def plot_floquet_multiplier_continuation(
    branch: Iterable[BranchPoint], ax=None, **plot_kwargs
):
    """
    Plot the magnitude of all Floquet multipliers over the continuation parameter.

    This function specifically handles Floquet stability analysis by:

    #. Extracting all Floquet multipliers from each :py:class:`~skhippr.solvers.continuation.BranchPoint`
    #. Plotting the absolute value of each multiplier against the continuation parameter
    #. Adding a reference line at ``magnitude = 1`` (stability boundary)
    #. Coloring stable/unstable segments based on individual multiplier magnitudes

    Parameters
    ----------
    branch : Iterable[BranchPoint]
        Collection of :py:class:`~skhippr.solvers.continuation.BranchPoint` objects from continuation analysis
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If ``None``, creates new subplot.
    **plot_kwargs
        Additional arguments passed to ax.plot()

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the Floquet multiplier continuation plot
    """
    branch_list = list(branch)

    if not branch_list:
        raise ValueError("Branch cannot be empty")

    parameter = branch_list[0].equations[-1].continuation_parameter
    if parameter is not None:
        parameter_values = np.array(
            [np.squeeze(getattr(bp, parameter)) for bp in branch_list]
        )
    else:
        parameter_values = np.arange(len(branch_list))

    all_magnitudes = []

    for bp in branch_list:
        multipliers = bp.equations[0].eigenvalues
        mag = np.abs(multipliers)
        all_magnitudes.append(mag)

    all_magnitudes = np.array(all_magnitudes)
    num_multipliers = len(all_magnitudes[-1])

    generated_ax = False
    if ax is None:
        _, ax = plt.subplots(1, 1)
        generated_ax = True

    stable_col = plot_kwargs.pop("stable_color", "r")
    unstable_col = plot_kwargs.pop("unstable_color", "b")
    title = plot_kwargs.pop("title", "Floquet Multiplier Continuation")
    xlabel = plot_kwargs.pop("xlabel", parameter if parameter else "Parameter")
    ylabel = plot_kwargs.pop("ylabel", "|$\\lambda$|")
    alpha = plot_kwargs.pop("alpha", 0.9)
    linestyle = plot_kwargs.pop("linestyle", "dotted")

    for i in range(num_multipliers):
        magnitudes_i = all_magnitudes[:, i]
        stability_i = magnitudes_i < 1.0

        xs_stable = np.where(stability_i, parameter_values, np.nan)
        xs_unstable = np.where(~stability_i, parameter_values, np.nan)
        ys_stable = np.where(stability_i, magnitudes_i, np.nan)
        ys_unstable = np.where(~stability_i, magnitudes_i, np.nan)

        ax.plot(
            xs_stable,
            ys_stable,
            color=stable_col,
            alpha=alpha,
            linewidth=1.5,
            **plot_kwargs,
        )
        ax.plot(
            xs_unstable,
            ys_unstable,
            color=unstable_col,
            alpha=alpha,
            linewidth=1.5,
            **plot_kwargs,
        )

    ax.axhline(
        y=1.0,
        color="k",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="Stability boundary (|λ| = 1)",
    )

    if generated_ax:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        y_min = max(0, all_magnitudes.min() * 0.9)
        y_max = all_magnitudes.max() * 1.1 if all_magnitudes.max() > 1.0 else 1.2
        ax.set_ylim(y_min, y_max)

    return ax


def plot_floquet_exponent_continuation(
    branch: Iterable[BranchPoint], ax=None, **plot_kwargs
):
    """
    Plot the real part of Floquet exponents over the continuation parameter.

    This function handles Floquet stability analysis by:

    #. Extracting Floquet multipliers from each :py:class:`~skhippr.solvers.continuation.BranchPoint`
    #. Computing Floquet exponents
    #. Plotting the real part of the exponents against the continuation parameter
    #. Adding a reference line at ``Re(alpha) = 0`` (stability boundary)
    #. Coloring stable/unstable segments based on individual exponent magnitudes

    Parameters
    ----------
    branch : Iterable[BranchPoint]
        Collection of :py:class:`~skhippr.solvers.continuation.BranchPoint` objects from continuation analysis
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If ``None``, creates new subplot.
    **plot_kwargs
        Additional arguments passed to ax.plot()

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the Floquet exponent continuation plot
    """
    branch_list = list(branch)

    if not branch_list:
        raise ValueError("Branch cannot be empty")

    parameter = branch_list[0].equations[-1].continuation_parameter
    if parameter is not None:
        parameter_values = np.array(
            [np.squeeze(getattr(bp, parameter)) for bp in branch_list]
        )
    else:
        parameter_values = np.arange(len(branch_list))

    all_exponents = []

    for bp in branch_list:

        multipliers = bp.equations[0].eigenvalues
        T = bp.equations[0].T_solution

        alphas = np.log(multipliers) / T
        all_exponents.append(alphas)

    num_exponents = len(all_exponents[-1])
    all_exponents = np.array(all_exponents)

    generated_ax = False
    if ax is None:
        _, ax = plt.subplots(1, 1)
        generated_ax = True

    stable_col = plot_kwargs.pop("stable_color", "r")
    unstable_col = plot_kwargs.pop("unstable_color", "b")
    title = plot_kwargs.pop("title", "Floquet Exponent Continuation")
    xlabel = plot_kwargs.pop("xlabel", parameter if parameter else "Parameter")
    ylabel = plot_kwargs.pop("ylabel", "Re($\\alpha$)")
    alpha = plot_kwargs.pop("alpha", 0.9)
    linestyle = plot_kwargs.pop("linestyle", "dotted")

    for i in range(num_exponents):
        exponents_i = all_exponents[:, i]
        real_part = np.real(exponents_i)
        stability_i = real_part < 0.0

        xs_stable = np.where(stability_i, parameter_values, np.nan)
        xs_unstable = np.where(~stability_i, parameter_values, np.nan)
        ys_stable = np.where(stability_i, real_part, np.nan)
        ys_unstable = np.where(~stability_i, real_part, np.nan)

        ax.plot(
            xs_stable,
            ys_stable,
            color=stable_col,
            alpha=alpha,
            linewidth=1.5,
            **plot_kwargs,
        )
        ax.plot(
            xs_unstable,
            ys_unstable,
            color=unstable_col,
            alpha=alpha,
            linewidth=1.5,
            **plot_kwargs,
        )

    ax.axhline(
        y=0.0,
        color="k",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="Stability boundary (Re($\\alpha$) = 0)",
    )

    if generated_ax:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        real_parts = np.real(all_exponents)
        y_min = (
            real_parts.min() * 1.1 if real_parts.min() < 0 else real_parts.min() * 0.9
        )
        y_max = (
            real_parts.max() * 1.1 if real_parts.max() > 0 else real_parts.max() * 0.9
        )
        y_min = min(y_min, -0.1)
        y_max = max(y_max, 0.1)
        ax.set_ylim(y_min, y_max)

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

        * Scalars and arrays with dimension 0: ``1.0`` or ``1`` as well as ``np.array(1.0)``
        * Single-element containers: ``[1.0]``, ``(1.0,)``, ``np.array([1.0])``, ``np.array([[1.0]])``
        * Containers where all first level elements contained squeeze to the same length: ``[[1.0]]``, ``[(1.0,)]``, ``1.0``.

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


def eval_plot_fun(
    branch: Iterable[BranchPoint],
    plot_fun: Callable[[BranchPoint], object],
):
    """
    Evaluate ``plot_fun`` across branch points

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

    value_list = []
    list_branch = []
    for bp in branch:
        value = plot_fun(bp)
        if np.isscalar(value):
            value_list.append(np.atleast_1d(value))
        else:
            value_list.append([np.squeeze(value[k]) for k in range(len(value))])

        list_branch.append(bp)
    return np.array(value_list), list_branch


def _get_handle_signature(handle):
    """
    Extract a hashable tuple representing the visual characteristics of a legend handle.
    The signature includes color, linestyle, marker, and linewidth properties.

    Parameters
    ----------
    handle : :py:class:`~matplotlib.lines.Line2D` | :py:class:`~matplotlib.artist.Artist`
        A matplotlib artist handle. Must support the methods ``get_color()``,
        ``get_linestyle()``, ``get_marker()``, and ``get_linewidth()``.

    Returns
    -------
    signature : :py:class:`tuple` of :py:class:`str`
        A 4-tuple containing string representations of the handle's visual properties in the order:
    """
    color = handle.get_color()
    linestyle = handle.get_linestyle()
    marker = handle.get_marker()
    linewidth = handle.get_linewidth()
    signature = (str(color), str(linestyle), str(marker), str(linewidth))
    return signature


def _deduplicate_legend(ax):
    """
    Remove duplicate legend entries based on visual signature and label.

    Parameters
    ----------
    ax : :py:class:`~matplotlib.axes.Axes`
        The matplotlib axes object containing the legend to be deduplicated.
        The legend must already exist on this axes.
    """
    legend = ax.get_legend()
    if legend is None:
        return
    handles = legend.get_lines()
    texts = legend.get_texts()

    if not handles:
        return
    seen = set()
    unique_handles = []
    unique_labels = []

    for handle, text in zip(handles, texts):
        sig = _get_handle_signature(handle)
        label = text.get_text()
        key = (sig, label)

        if key not in seen:
            seen.add(key)
            unique_handles.append(handle)
            unique_labels.append(label)

    ax.legend(unique_handles, unique_labels, loc="best")
