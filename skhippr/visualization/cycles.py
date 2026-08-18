"""

The :py:mod:`~skhippr.visualization.cycles` module provides standardized functions for visualizing limit cycles over multiple periods, their phase portraits as well as Floquet multipliers and exponents.

Supported arguments are an instance of :py:class:`~skhippr.cycles.hbm.HBMEquation` or an :py:class:`~skhippr.equations.EquationSystem.EquationSystem` like a :py:class:`~skhippr.cycles.hbm.HBMSystem` that contains such an equation.

It provides the functions:

* :py:func:`~skhippr.visualization.cycles.plot_period` for plotting the time series of the equation solution
* :py:func:`~skhippr.visualization.cycles.plot_phase` for making phase portraits
* :py:func:`~skhippr.visualization.cycles.plot_floquet_multipliers` as well as :py:func:`~skhippr.visualization.cycles.plot_floquet_exponents` for visualizing the Floquet multipliers and exponents of a cycle.

The corresponding animation functions:

* :py:func:`~skhippr.visualization.cycles.animate_period`
* :py:func:`~skhippr.visualization.cycles.animate_phase`
* :py:func:`~skhippr.visualization.cycles.animate_floquet_multipliers`
* :py:func:`~skhippr.visualization.cycles.animate_floquet_exponents`

create animations analog to the plotting functions but require :py:class:`collections.abc.Iterable` objects containing solved :py:class:`~skhippr.cycles.hbm.HBMEquation` instances instead.

This module also offers functions for visualizing matrices by their spectral norm. Namely:

* :py:func:`~skhippr.visualization.cycles.plot_matrix_block_norm` which subdivides a matrix into subblocks and creates a scatter plot colored by the 2-norm of each block
* :py:func:`~skhippr.visualization.cycles.plot_hill_matrix_blocks` which uses the former function to visualize the Hill matrix of a :py:class:`~skhippr.cycles.hbm.HBMEquation`.

"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation

from skhippr.cycles.hbm import HBMEquation
from skhippr.equations.EquationSystem import EquationSystem
from collections.abc import Sequence, Iterable

from skhippr.visualization._helpers import (
    robust_plot,
    parse_and_generate_axis,
    extract_equation,
    animate,
)


def plot_period(
    hbm: HBMEquation | EquationSystem,
    ax=None,
    idx=0,
    n_periods: float = 1.0,
    **plot_kwargs,
):
    """
    Plot the time series of a solved :py:class:`~skhippr.cycles.hbm.HBMEquation` over a given number of periods.

    Parameters
    ----------
    hbm : HBMEquation or EquationSystem
        The equation or equation system containing the solution. If it is of type :py:class:`~skhippr.equations.EquationSystem.EquationSystem`, the first valid :py:class:`~skhippr.cycles.hbm.HBMEquation` instance contained is used.
    ax : matplotlib.axes.Axes, optional
        The :py:class:`~matplotlib.axes.Axes` object on which to plot. If ``None``, an :py:class:`~matplotlib.axes.Axes` instance will be created.
    idx : int, optional
        The index of the state to be plotted.
    n_periods : float, optional
        The number of periods for which the time series is plotted. May be non-integer.
    **plot_kwargs
        Additional keyword arguments passed through to ``ax.plot()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The :py:class:`~matplotlib.axes.Axes` object with the plotted period response.

    """

    default_args = {
        "title": f"Time series - {n_periods} periods",
        "xlabel": "t",
        "ylabel": f"x_{idx}",
    }

    kwargs = {**default_args, **plot_kwargs}

    # determine quantities to plot
    equation = extract_equation(hbm, HBMEquation)
    omega = equation.omega_solution
    t = equation.fourier.time_samples(omega, n_periods)
    x = equation.x_time()[idx, :]

    # repeat the time series to cover the requested number of periods
    x = np.tile(x, int(np.ceil(n_periods)))[: t.size]

    # plot
    ax = parse_and_generate_axis(ax, **kwargs)
    robust_plot(ax.plot, t, x, **kwargs)

    return ax


def animate_period(
    hbm_set: Iterable[HBMEquation | EquationSystem],
    ax=None,
    idx: int = 0,
    n_periods: float = 1.0,
    interval: int = 30,
    repeat: bool = True,
    scaling=True,
    anim_title=None,
    scale_padding=1.05,
    **plot_kwargs,
):
    """
    Create an animated time series visualization for multiple solved :py:class:`~skhippr.cycles.hbm.HBMEquation` instances over a given number of periods.

    Parameters
    ----------
    hbm_set : Iterable[HBMEquation | EquationSystem]
        An iterable containing either :py:class:`~skhippr.cycles.hbm.HBMEquation` objects or :py:class:`~skhippr.equations.EquationSystem.EquationSystem` objects that contain a :py:class:`~skhippr.cycles.hbm.HBMEquation`.
        Each :py:class:`~skhippr.cycles.hbm.HBMEquation` instance will be animated sequentially.
    ax : matplotlib.axes.Axes, optional
        The :py:class:`~matplotlib.axes.Axes` object on which to plot. If ``None``, an
        :py:class:`~matplotlib.axes.Axes` instance will be created.
    idx : int, optional
        The index of the state to be plotted across all equations.
    n_periods : float, optional
        The number of periods for which each time series is animated. May be non-integer.
    interval : int, optional
        The delay between frames in milliseconds. Default is 30 ms.
    repeat : bool, optional
        Whether to repeat the animation. Default is ``True``.
    **plot_kwargs
        Additional keyword arguments passed to ``ax.plot()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The :py:class:`~matplotlib.axes.Axes` object with the animated period response.
    animation : matplotlib.animation.FuncAnimation
        The animation object that controls the visualization.

    Notes
    -----
    The returned :py:class:`~matplotlib.animation.FuncAnimation` object must be kept in a variable and not discarded to prevent Python's
    garbage collector from deleting it, causing the animation to stop.
    """

    default_args = {
        "xlabel": "t",
        "ylabel": f"x_{idx}",
    }

    plot_kwargs = {**default_args, **plot_kwargs}

    if ax is None and anim_title is None:
        anim_title = f"time history x[{idx}]"

    times = []
    signals = []

    for hbm in hbm_set:
        equation = extract_equation(hbm, HBMEquation)
        fourier = equation.fourier
        omega = equation.omega_solution
        x_singleperiod = equation.x_time()[idx, :]

        t = fourier.time_samples(omega, n_periods)
        n_rep = int(np.ceil(n_periods))
        x = np.tile(x_singleperiod, n_rep)[: t.size]
        times.append(t)
        signals.append(x)

    ax, animation = animate(
        times,
        signals,
        ax,
        anim_title,
        interval,
        repeat,
        scaling,
        scale_padding,
        **plot_kwargs,
    )
    return ax, animation


def plot_phase(
    hbm: HBMEquation | EquationSystem,
    ax=None,
    idx: Sequence[int] = [0, 1],
    **plot_kwargs,
):
    """
    Plot the phase of a solved :py:class:`~skhippr.cycles.hbm.HBMEquation`.

    Parameters
    ----------
    hbm : HBMEquation or EquationSystem
        The equation or equation system containing the solution. If it is of type :py:class:`~skhippr.equations.EquationSystem.EquationSystem`, the first valid :py:class:`~skhippr.cycles.hbm.HBMEquation` instance contained is used.
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

    default_args = {
        "title": f"Phase plot - x_{idx[0]} vs x_{idx[1]}",
        "xlabel": f"x_{idx[0]}",
        "ylabel": f"x_{idx[1]}",
    }

    kwargs = {**default_args, **plot_kwargs}

    # determine quantities to plot
    equation = extract_equation(hbm, HBMEquation)
    x = equation.x_time()

    # plot
    ax = parse_and_generate_axis(ax, **kwargs)
    robust_plot(
        ax.plot, x[idx[0], :], x[idx[1], :], default_args=default_args, **kwargs
    )

    return ax


def animate_phase(
    hbm_set: Iterable[HBMEquation | EquationSystem],
    ax=None,
    idx=(0, 1),
    interval: int = 30,
    repeat: bool = True,
    scaling=True,
    anim_title=None,
    scale_padding=1.05,
    **plot_kwargs,
):
    """
    Create an animated phase plane visualization for multiple solved :py:class:`~skhippr.cycles.hbm.HBMEquation` instances.

    This function generates a sequence of phase plots, where each frame displays the trajectory of a different equation from the input set.

    Parameters
    ----------
    hbm_set : iterable of HBMEquation or EquationSystem
        An iterable containing either :py:class:`~skhippr.cycles.hbm.HBMEquation` objects or :py:class:`~skhippr.equations.EquationSystem.EquationSystem` objects that contain a :py:class:`~skhippr.cycles.hbm.HBMEquation`.
        Each :py:class:`~skhippr.cycles.hbm.HBMEquation` instance will be animated sequentially.
    ax : matplotlib.axes.Axes, optional
        The :py:class:`~matplotlib.axes.Axes` object on which to plot. If ``None``,
        a new :py:class:`~matplotlib.axes.Axes` instance will be created.
    idx : Sequence[int], optional
        Exactly two indices of the states to be considered for the phase plane
        (e.g., ``(0, 1)`` for an x-y phase plot). Default is ``(0, 1)``.
    interval : int, optional
        The delay between frames in milliseconds. Default is 30 ms.
    repeat : bool, optional
        Whether to repeat the animation loop. Default is ``True``.
    **plot_kwargs
        Additional keyword arguments passed to ``ax.plot()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The :py:class:`~matplotlib.axes.Axes` object with the animated phase plot.
    animation : matplotlib.animation.FuncAnimation
        The animation object controlling the visualization loop.

    Notes
    -----
    The returned :py:class:`~matplotlib.animation.FuncAnimation` object must be kept in a variable and not discarded to prevent Python's
    garbage collector from deleting it, causing the animation to stop.
    """
    default_args = {
        "xlabel": f"x_{idx[0]}",
        "ylabel": f"x_{idx[1]}",
    }

    plot_kwargs = {**default_args, **plot_kwargs}

    if ax is None and anim_title is None:
        anim_title = f"phase plot x[{idx[0]}], x[{idx[1]}]"

    xdata = []
    ydata = []

    for hbm in hbm_set:
        equation = extract_equation(hbm, HBMEquation)
        x = equation.x_time()
        xdata.append(x[idx[0], :])
        ydata.append(x[idx[1], :])

    ax, animation = animate(
        xdata,
        ydata,
        ax,
        anim_title,
        interval,
        repeat,
        scaling,
        scale_padding,
        **plot_kwargs,
    )
    return ax, animation


def plot_floquet_multipliers(hbm: HBMEquation | EquationSystem, ax=None, **plot_kwargs):
    """
    Plot the Floquet multipliers of the :py:class:`~skhippr.cycles.hbm.HBMEquation` solution.

    Parameters
    ----------
    hbm : HBMEquation or EquationSystem
        The equation or equation system containing the solution. If it is of type :py:class:`~skhippr.equations.EquationSystem.EquationSystem`, the first valid :py:class:`~skhippr.cycles.hbm.HBMEquation` instance contained is used.
    ax : matplotlib.axes.Axes, optional
        The :py:class:`~matplotlib.axes.Axes` object on which to plot. If ``None``, a new :py:class:`~matplotlib.axes.Axes` instance will be created.
    **plot_kwargs
        Additional keyword arguments passed to ``ax.plot()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The :py:class:`~matplotlib.axes.Axes` object with a scatter plot of the Floquet multipliers on the complex plane. A unit circle is also plotted, if no axes is given in the call.
    """

    default_args = {
        "title": "Floquet multipliers",
        "xlabel": "Re($\\lambda$)",
        "ylabel": "Im($\\lambda$)",
        "marker": "x",
    }

    kwargs = {**default_args, **plot_kwargs}

    equation = extract_equation(hbm, HBMEquation)
    floquet_multipliers = equation.eigenvalues

    new_axis = ax is None
    ax = parse_and_generate_axis(ax, **kwargs)

    # plot unit circle
    if new_axis:
        ax.plot(
            np.cos(equation.fourier.time_samples_normalized),
            np.sin(equation.fourier.time_samples_normalized),
            "k",
        )
        ax.set_aspect("equal", adjustable="datalim")

    robust_plot(
        ax.scatter,
        np.real(floquet_multipliers),
        np.imag(floquet_multipliers),
        **kwargs,
    )
    return ax


def animate_floquet_multipliers(
    hbm_set: Iterable[HBMEquation | EquationSystem],
    ax=None,
    interval: int = 30,
    repeat: bool = True,
    scaling=True,
    show_unit_circle=True,
    anim_title=None,
    scale_padding=1.05,
    **plot_kwargs,
):
    """
    Create an animated visualization of Floquet multipliers for multiple solved :py:class:`~skhippr.cycles.hbm.HBMEquation` instances on the complex plane.
    Each frame displays the multipliers for one equation.
    If no :py:class:`~matplotlib.axes.Axes` object is passed, the function will create one as well as a unit circle to visualize stability.

    Parameters
    ----------
    hbm_set : iterable of HBMEquation or EquationSystem
        An iterable containing either :py:class:`~skhippr.cycles.hbm.HBMEquation` objects or :py:class:`~skhippr.equations.EquationSystem.EquationSystem` objects that contain a :py:class:`~skhippr.cycles.hbm.HBMEquation`.
        The Floquet multipliers of each :py:class:`~skhippr.cycles.hbm.HBMEquation` instance will be animated sequentially.
    ax : matplotlib.axes.Axes, optional
        The :py:class:`~matplotlib.axes.Axes` object on which to plot. If ``None``,
        a new :py:class:`~matplotlib.axes.Axes` instance will be created.
    show_full_range : bool, optional
        If ``True``, the axes limits are fixed from the start to the range spanning all Floquet
        multipliers across every frame (padded so the unit circle stays visible). If ``False``
        (default), the axes limits are fixed to ``[-1.2, 1.2]`` on both axes.
    interval : int, optional
        The delay between frames in milliseconds. Default is 30 ms.
    repeat : bool, optional
        Whether to repeat the animation loop. Default is ``True``.
    **plot_kwargs
        Additional keyword arguments passed to ``ax.scatter()``. Note that the
        default marker is set to 'x' unless overridden.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The :py:class:`~matplotlib.axes.Axes` object with the animated plot.
    animation : matplotlib.animation.FuncAnimation
        The animation object controlling the visualization loop.

    Notes
    -----
    The returned :py:class:`~matplotlib.animation.FuncAnimation` object must be kept in a variable and not discarded to prevent Python's
    garbage collector from deleting it, causing the animation to stop.
    """

    default_args = {
        "xlabel": f"Re(\lambda)",
        "ylabel": f"Im(\lambda)",
        "marker": "x",
        "linestyle": "none",
    }

    plot_kwargs = {**default_args, **plot_kwargs}

    if ax is None and anim_title is None:
        anim_title = f"Floquet multipliers"

    if show_unit_circle:
        ax = parse_and_generate_axis(ax)
        theta = np.linspace(0, 2 * np.pi, 400)
        unit_x = np.cos(theta)
        unit_y = np.sin(theta)
        ax.plot(unit_x, unit_y, "k-")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)

    xdata = []
    ydata = []

    for hbm in hbm_set:
        equation = extract_equation(hbm, HBMEquation)
        FMs = equation.eigenvalues
        xdata.append(np.real(FMs))
        ydata.append(np.imag(FMs))

    ax, animation = animate(
        xdata,
        ydata,
        ax,
        anim_title,
        interval,
        repeat,
        scaling,
        scale_padding,
        **plot_kwargs,
    )

    return ax, animation


def plot_floquet_exponents(hbm: HBMEquation | EquationSystem, ax=None, **plot_kwargs):
    """
    Calculate and plot the Floquet exponents of the :py:class:`~skhippr.cycles.hbm.HBMEquation` solution from the Floquet multipliers.

    Parameters
    ----------
    hbm : HBMEquation or EquationSystem
        The equation or equation system containing the solution. If it is of type :py:class:`~skhippr.equations.EquationSystem.EquationSystem`, the first valid :py:class:`~skhippr.cycles.hbm.HBMEquation` instance contained is used.
    ax : matplotlib.axes.Axes, optional
        The :py:class:`~matplotlib.axes.Axes` object on which to plot. If ``None``, a new :py:class:`~matplotlib.axes.Axes` instance will be created.
    **plot_kwargs
        Additional keyword arguments passed to ``ax.plot()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The :py:class:`~matplotlib.axes.Axes` object with a scatter plot of the Floquet exponents on the imaginary plane.
    """

    default_args = {
        "title": "Floquet exponents",
        "xlabel": "Re($\\alpha$)",
        "ylabel": "Im($\\alpha$)",
        "marker": "x",
    }

    kwargs = {**default_args, **plot_kwargs}

    equation = extract_equation(hbm, HBMEquation)
    floquet_multipliers = equation.eigenvalues
    floquet_exponents = np.log(floquet_multipliers) / equation.T_solution

    new_axis = ax is None
    ax = parse_and_generate_axis(ax, **kwargs)

    # plot stability boundary
    if new_axis:
        ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0)

    robust_plot(
        ax.scatter,
        np.real(floquet_exponents),
        np.imag(floquet_exponents),
        **kwargs,
    )
    return ax


def animate_floquet_exponents(
    hbm_set: Iterable[HBMEquation | EquationSystem],
    ax=None,
    interval: int = 30,
    repeat: bool = True,
    scaling=True,
    anim_title=None,
    scale_padding=1.05,
    **plot_kwargs,
):
    """
    Create an animated visualization of Floquet exponents for multiple solved :py:class:`~skhippr.cycles.hbm.HBMEquation` instances in the complex plane.
    Each frame displays the exponents for one equation.
    If no :py:class:`~matplotlib.axes.Axes` object is passed, the function will create one as well as the imaginary axis to visualize stability.

    Parameters
    ----------
    hbm_set : iterable of HBMEquation or EquationSystem
       An iterable containing either :py:class:`~skhippr.cycles.hbm.HBMEquation` objects or :py:class:`~skhippr.equations.EquationSystem.EquationSystem` objects that contain a :py:class:`~skhippr.cycles.hbm.HBMEquation`.
       The Floquet exponents of each :py:class:`~skhippr.cycles.hbm.HBMEquation` instance will be animated sequentially.
    ax : matplotlib.axes.Axes, optional
        The :py:class:`~matplotlib.axes.Axes` object on which to plot. If ``None``, a new :py:class:`~matplotlib.axes.Axes` instance will be created.
    show_full_range: bool, optional
        Whether to show the full data range. If ``False`` the axis limits are set to show the data range of the first :py:class:`~skhippr.cycles.hbm.HBMEquation`.
    interval : int, optional
        The delay between frames in milliseconds. Default is 30 ms.
    repeat : bool, optional
        Whether to repeat the animation loop. Default is ``True``.
    **plot_kwargs
        Additional keyword arguments passed to ``ax.scatter()``. Note that the
        default marker is set to 'x' unless overridden.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The :py:class:`~matplotlib.axes.Axes` object with the animated plot.
    animation : matplotlib.animation.FuncAnimation
        The animation object controlling the visualization loop.

    Notes
    -----
    The returned :py:class:`~matplotlib.animation.FuncAnimation` object must be kept in a variable and not discarded to prevent Python's
    garbage collector from deleting it, causing the animation to stop.

    """
    default_args = {
        "xlabel": f"Re(\alpha)",
        "ylabel": f"Im(\alpha)",
        "marker": "x",
        "linestyle": "none",
    }

    plot_kwargs = {**default_args, **plot_kwargs}

    if ax is None and anim_title is None:
        anim_title = f"Floquet exponents"

    xdata = []
    ydata = []

    for hbm in hbm_set:
        equation = extract_equation(hbm, HBMEquation)
        FMs = equation.eigenvalues
        FEs = np.log(FMs) / equation.T_solution
        xdata.append(np.real(FEs))
        ydata.append(np.imag(FEs))

    ax, animation = animate(
        xdata,
        ydata,
        ax,
        anim_title,
        interval,
        repeat,
        scaling,
        scale_padding,
        **plot_kwargs,
    )

    return ax, animation


def plot_hill_matrix_blocks(
    hbm: HBMEquation | EquationSystem,
    real_formulation=None,
    ax=None,
    logscale=False,
    vmax=None,
    vmin=None,
    **plot_kwargs,
):
    """
    Plot the Hill matrix as a grid of blocks, colored by the magnitude of each block.
    This function computes the Hill matrix of a solved :py:class:`~skhippr.cycles.hbm.HBMEquation`,
    partitions it into ``n_dof`` x ``n_dof`` sub-blocks and creates a scatter plot where each
    block is visualized as a dot using the :py:func:`~skhippr.visualization.cycles.plot_matrix_block_norm` function.
    The color represents the block`s 2-norm (spectral norm) and the scatter point size is automatically
    adjusted by default.

    Parameters
    ----------
    hbm : HBMEquation or EquationSystem
        The equation or equation system containing the solution. If it is of type :py:class:`~skhippr.equations.EquationSystem.EquationSystem`, the first valid :py:class:`~skhippr.cycles.hbm.HBMEquation` instance contained is used.
    ax : matplotlib.axes.Axes, optional
        The :py:class:`~matplotlib.axes.Axes` object on which to plot. If ``None``, a new :py:class:`~matplotlib.axes.Axes` instance will be created.
    real_formulation : bool, optional
        If ``True``, the Hill matrix is built using the real-valued form of the HBM equations. The plot axes ticks display ``0``, ``1c``, ``...``, ``Nc``, ``1s``, ``...``, ``Ns`` when ``True`` and ``-N``, ``...``, ``N`` if ``False``.
        When no ``real_formulation`` is given, the Hill matrix is built using the native formulation contained in the :py:class:`~skhippr.cycles.hbm.HBMEquation`.
    logscale : bool, optional
        If ``True``, uses logarithmic color scaling via :py:class:`matplotlib.colors.LogNorm`.
    vmax, vmin : float, optional
        Colorbar value limits passed to :py:class:`~matplotlib.colors.LogNorm` when ``logscale=True``.
        If ``None``, limits are determined automatically.
    **plot_kwargs
        Additional keyword arguments passed to :py:func:`matplotlib.axes.Axes.scatter`. The automatic scatter point size can be overridden by passing it here as ``s`` explicitly.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The :py:class:`~matplotlib.axes.Axes` object with the plotted Hill matrix blocks.
    """
    hbm = _get_equation_helper(hbm=hbm)
    H = hbm.hill_matrix(real_formulation=real_formulation, update=True)
    if real_formulation is None:
        real_formulation = hbm.fourier.real_formulation
    n_dof = hbm.fourier.n_dof

    if real_formulation:
        index = range(2 * hbm.fourier.N_HBM + 1)
    else:
        index = range(-hbm.fourier.N_HBM, hbm.fourier.N_HBM + 1)

    ax, _ = plot_matrix_block_norm(
        H,
        n_dof,
        ax=ax,
        index=index,
        logscale=logscale,
        vmax=vmax,
        vmin=vmin,
        **plot_kwargs,
    )
    N = hbm.fourier.N_HBM
    step = max(1, len(index) // 15)
    tick_locs = index[::step]
    if real_formulation:
        labels = (
            ["0"]
            + [f"{i}c" for i in range(1, N + 1)]
            + [f"{i}s" for i in range(1, N + 1)]
        )
        tick_labels = labels[::step]
    else:
        tick_labels = [str(loc) for loc in tick_locs]
    ax.set_xticks(tick_locs, labels=tick_labels, fontsize=10)
    ax.set_yticks(tick_locs, labels=tick_labels, fontsize=10)

    return ax


def plot_matrix_block_norm(
    matrix: np.ndarray,
    block_size: int,
    ax=None,
    index=None,
    logscale=False,
    vmax=None,
    vmin=None,
    **plot_kwargs,
):
    """
    Plot a square matrix as a grid of color-coded blocks, where each block's color corresponds to
    its 2-norm and its position reflects its placement within the matrix.

    Parameters
    ----------
    matrix : np.ndarray
        The square matrix to be visualized.
    block_size : int
        The dimension of each sub-block.
    ax : matplotlib.axes.Axes, optional
        The :py:class:`~matplotlib.axes.Axes` object on which to plot.
        If ``None``, a new :py:class:`~matplotlib.axes.Axes` instance will be created.
    index : array-like, optional
        Custom tick labels or coordinate indices.
        If ``None``, uses ``range(matrix.shape[0] // block_size)``.
    logscale : bool, optional
        If ``True``, use logarithmic color scaling via :py:class:`~matplotlib.colors.LogNorm`.
    vmax, vmin : float, optional
        Colorbar value limits passed to :py:class:`~matplotlib.colors.LogNorm`
        when ``logscale=True``. If ``None``, limits are determined automatically.
    **plot_kwargs
        Additional keyword arguments passed to ``ax.scatter()``.
        The plot title can also be passed here via the dictionary key ``title``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The :py:class:`~matplotlib.axes.Axes` object with the plotted matrix blocks.
    sc : matplotlib.collections.PathCollection
        The scatter plot collection object (for accessing colorbar, etc.).

    Notes
    -----
    Each matrix block is represented as a dot located by its (row, column) indices, colored by the 2-norm of that block.
    If no explicit point size (``s``) is specified, the scatter points are automatically sized based on the total number of blocks.
    This ensures proportional visualization of each dot in the scatter plot depending on matrix dimension.
    The scatter point size is bounded to guarantee clarity for both small and large block grids.
    """
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Input matrix must be 2-D and square but has shape {matrix.shape}"
        )

    if index is None:
        num_blocks = matrix.shape[0] // block_size
        index = range(num_blocks)
    else:
        num_blocks = len(index)

    if num_blocks * block_size != matrix.shape[0]:
        raise ValueError(
            f"Matrix size ({matrix.shape[0]}) not given by block size "
            f"({block_size}) times number of blocks ({num_blocks})"
        )

    x_positions = []
    y_positions = []
    norm_values = []

    for i in range(num_blocks):
        for j in range(num_blocks):
            block = matrix[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size,
            ]
            norm = np.linalg.norm(block, ord=2)
            x_positions.append(index[j])
            y_positions.append(index[i])
            norm_values.append(norm)

    if isinstance(ax, str):
        title = ax
    else:
        title = ""
    if ax is None or isinstance(ax, str):
        _, ax = plt.subplots()
        ax.set_xlabel("Column index")
        ax.set_ylabel("Row index")
        ax.set_title(title)
        ax.invert_yaxis()

    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)
    norm_values = np.array(norm_values)

    title = plot_kwargs.pop("title", None)

    if "s" not in plot_kwargs:
        base_size = 13000.0
        s_min, s_max = 1.0, 200.0
        auto_s = base_size / max(num_blocks**2, 1)
        auto_s = max(s_min, min(s_max, auto_s))
        plot_kwargs["s"] = auto_s
        plot_kwargs["s"] = auto_s

    scatter_defaults = {"cmap": "viridis", "alpha": 0.8}
    if logscale:
        scatter_defaults["norm"] = LogNorm(vmax=vmax, vmin=vmin, clip=False)
    scatter_defaults.update(plot_kwargs)

    sc = robust_plot(
        ax.scatter,
        x_positions,
        y_positions,
        c=norm_values,
        **scatter_defaults,
    )
    cbar = plt.colorbar(sc, ax=ax)
    if logscale:
        cbar.formatter = plt.matplotlib.ticker.LogFormatterMathtext(base=10)
        cbar.update_ticks()
    cbar.set_label("2-norm of block")

    if title is None:
        ax.set_title("Block Matrix Norms")
    else:
        ax.set_title(title)

    return ax, sc


def _get_equation_helper(hbm: HBMEquation | EquationSystem):
    """
    Helper function that returns a :py:class:`~skhippr.cycles.hbm.HBMEquation`.
    Parameters
    ----------
    hbm : HBMEquation or EquationSystem
        An equation or system of equations.

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
    if isinstance(hbm, EquationSystem):
        for equation in hbm.equations:
            if isinstance(equation, HBMEquation):
                return equation
    raise ValueError("hbm does not contain any usable HBMEquation instance")


def _get_padded_limits(x, pad_multiplier=0.05):
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    x_range = x_max - x_min
    x_pad = pad_multiplier * x_range
    return [x_min - x_pad, x_max + x_pad]
