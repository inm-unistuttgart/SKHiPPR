"""

The :py:mod:`~skhippr.visualization.cycles` module provides standardized functions for visualizing limit cycles over multiple periods, their phase portraits as well as Floquet multipliers and exponents.

Supported arguments are an instance of :py:class:`~skhippr.cycles.hbm.HBMEquation` or an :py:class:`~skhippr.equations.EquationSystem.EquationSystem` like a :py:class:`~skhippr.cycles.hbm.HBMSystem` that contains such an equation.

It provides the functions :py:func:`~skhippr.visualization.cycles.plot_period` for plotting the time series of the equation solution,
:py:func:`~skhippr.visualization.cycles.plot_phase` for making phase portraits and :py:func:`~skhippr.visualization.cycles.plot_floquet_multipliers` as well as :py:func:`~skhippr.visualization.cycles.plot_floquet_exponents` for visualizing the Floquet multipliers and exponents of a cycle.

This module also offers functions for visualizing matrices by their spectral norm. Namely :py:func:`~skhippr.visualization.cycles.plot_matrix_block_norm` which subdivides a matrix into subblocks and creates a scatter plot colored by the 2-norm of each block,
as well as :py:func:`~skhippr.visualization.cycles.plot_hill_matrix_blocks` which uses the prior function to visualize the Hill matrix of a :py:class:`~skhippr.cycles.hbm.HBMEquation`.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from skhippr.cycles.hbm import HBMEquation
from skhippr.equations.EquationSystem import EquationSystem
from collections.abc import Sequence


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
    generated_ax = False
    if ax is None:
        _, ax = plt.subplots(1, 1)
        generated_ax = True

    kwargs = {"marker": "x"}
    kwargs.update(plot_kwargs)

    equation = _get_equation_helper(hbm)
    floquet_multipliers = equation.eigenvalues
    fourier = equation.fourier
    ax.scatter(
        np.real(floquet_multipliers),
        np.imag(floquet_multipliers),
        **kwargs,
    )
    if generated_ax:
        ax.set_title("Floquet multipliers")
        ax.set_xlabel("Re($\\lambda$)")
        ax.set_ylabel("Im($\\lambda$)")
        ax.set_aspect("equal")
        ax.plot(
            np.cos(fourier.time_samples_normalized),
            np.sin(fourier.time_samples_normalized),
            "k",
        )
    return ax


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
    generated_ax = False
    if ax is None:
        _, ax = plt.subplots(1, 1)
        generated_ax = True

    kwargs = {"marker": "x"}
    kwargs.update(plot_kwargs)

    equation = _get_equation_helper(hbm)
    floquet_multipliers = equation.eigenvalues
    lambdas = np.asarray(floquet_multipliers)
    floquet_exponents = np.log(lambdas) / equation.T_solution
    ax.scatter(
        np.real(floquet_exponents),
        np.imag(floquet_exponents),
        **kwargs,
    )
    if generated_ax:
        ax.set_title("Floquet exponents")
        ax.set_xlabel("Re($\\alpha$)")
        ax.set_ylabel("Im($\\alpha$)")
        ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0)
    return ax


def plot_hill_matrix_blocks(
    hbm: HBMEquation | EquationSystem,
    real_formulation=False,
    ax=None,
    logscale=False,
    vmax=None,
    vmin=None,
    **plot_kwargs,
):
    """
    Plot the Hill matix as a grid of blocks, colored and sized by its properties.
    This function computes the Hill matrix of a solved :py:class:`~skhippr.cycles.hbm.HBMEquation`,
    partitions it into ``n_dof`` x ``n_dof`` sub-blocks and creates a scatter plot where each
    block is visualized as a dot. 
    The color represents the block`s 2-norm (spectral norm) and the scatter point size is automatically 
    adjusted to the number of blocks, ensuring visually balanced scaling across different matrix and problem sizes.

    Parameters
    ----------
    hbm : HBMEquation or EquationSystem
        The equation or equation system containing the solution. If it is of type :py:class:`~skhippr.equations.EquationSystem.EquationSystem`, the first valid :py:class:`~skhippr.cycles.hbm.HBMEquation` instance contained is used.
    ax : matplotlib.axes.Axes, optional
        The :py:class:`~matplotlib.axes.Axes` object on which to plot. If ``None``, a new :py:class:`~matplotlib.axes.Axes` instance will be created.
    real_formulation : bool, optional
        If ``True``, the Hill matrix is built using the real-valued form of the HBM equations. The plot axes ticks display ``0``, ``1c``, ``...``, ``Nc``, ``1s``, ``...``, ``Ns`` when ``True`` and ``-N``, ``...``, ``N`` otherwise.
    logscale : bool, optional
        If ``True``, uses logarithmic color scaling via :py:class:`matplotlib.colors.LogNorm`.
    vmax, vmin : float, optional
        Colorbar value limits passed to :py:class:`~matplotlib.colors.LogNorm` when ``logscale=True``.
        If ``None``, limits are determined automatically.
    **plot_kwargs
        Additional keyword arguments passed to :py:func:`matplotlib.axes.Axes.scatter`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The :py:class:`~matplotlib.axes.Axes` object with the plotted Hill matrix blocks.
    """
    hbm = _get_equation_helper(hbm=hbm)
    H = hbm.hill_matrix(real_formulation=real_formulation, update=True)
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
        **plot_kwargs
    )
    N = hbm.fourier.N_HBM
    step = max(1, len(index) // 15)
    tick_locs = index[::step]
    if real_formulation:
        labels = ['0'] + [f'{i}c' for i in range(1, N+1)] + [f'{i}s' for i in range(1, N+1)]
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
    its 2-norm and its poistion reflects its placement within the matrix.

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

    sc = ax.scatter(
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
