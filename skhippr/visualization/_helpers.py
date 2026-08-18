import warnings
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation

from skhippr.equations.AbstractEquation import AbstractEquation
from skhippr.cycles.hbm import HBMEquation, HBMSystem
from skhippr.equations.EquationSystem import EquationSystem
from collections.abc import Sequence, Iterable, Callable


def robust_plot(plot_method: Callable[..., Any], *args, **kwargs) -> Any:
    """
    Plotting wrapper:
    Call ``plot_method(*args, **kwargs)``, where ``plot_method`` is typically
    ``ax.plot`` or ``ax.scatter`` while handling keyword arguments robustly.

    * Keyword arguments that are valid properties of the resulting artist are
    passed through and take effect normally.
    * A keyword argument that is not a valid property is ignored.

    Parameters
    ----------
    plot_method : Callable[..., Any]
        The plotting method to call, e.g. ``ax.plot`` or ``ax.scatter``.
    *args
        Positional arguments passed through to ``plot_method``.
    **kwargs
        Keyword arguments passed through to ``plot_method``. Any keyword argument that
        ``plot_method`` rejects with an ``AttributeError`` (i.e. it is not a valid
        property of the resulting artist) is silently dropped and the call is retried.

    Returns
    -------
    result : Any
        Whatever ``plot_method`` returns, e.g. a list of :py:class:`~matplotlib.lines.Line2D` for ``ax.plot``, or a :py:class:`~matplotlib.collections.PathCollection` for ``ax.scatter``.
    """

    while True:
        try:
            return plot_method(*args, **kwargs)
        except AttributeError as error:
            bad_kwarg = getattr(error, "name", None)
            if bad_kwarg is None or bad_kwarg not in kwargs:
                # Not a case of an unrecognized keyword argument: re-raise.
                raise error
            # warnings.warn(
            #     f"Ignoring keyword argument '{bad_kwarg}={kwargs[bad_kwarg]}': "
            #     "not a valid property of the plotted artist.",
            #     UserWarning,
            # )
            del kwargs[bad_kwarg]


def animate(
    xdata: Sequence,
    ydata: Sequence,
    ax: plt.Axes | None = None,
    anim_title: str | None = None,
    interval: int = 30,
    repeat: bool = True,
    scaling: bool = True,
    scale_padding: float = 1.05,
    **kwargs,
) -> tuple[plt.Axes, FuncAnimation]:
    """
    Animate a sequence of 2D line plots by cycling through frames of ``xdata``/``ydata``.

    Each frame corresponds to one element of ``xdata`` and ``ydata`` and is drawn as a
    single line via :py:func:`~skhippr.visualization._helpers.robust_plot`. This is the
    shared animation backend used by
    :py:func:`~skhippr.visualization.cycles.animate_period`,
    :py:func:`~skhippr.visualization.cycles.animate_phase`,
    :py:func:`~skhippr.visualization.cycles.animate_floquet_multipliers` and
    :py:func:`~skhippr.visualization.cycles.animate_floquet_exponents`.

    Parameters
    ----------
    xdata : Sequence
        Sequence of x-data arrays, one per frame.
    ydata : Sequence
        Sequence of y-data arrays, one per frame. Must have the same length as ``xdata``.
    ax : matplotlib.axes.Axes, optional
        The :py:class:`~matplotlib.axes.Axes` object on which to plot. If ``None``, a new :py:class:`~matplotlib.axes.Axes` instance will be created.
    anim_title : str, optional
        If given, used as a title prefix for every frame; the current frame index is appended automatically (e.g. ``"{anim_title} (frame 3/10)"``). If ``None``, the axis title is left unchanged.
    interval : int, optional
        The delay between frames in milliseconds. Default is 30 ms.
    repeat : bool, optional
        Whether to repeat the animation loop. Default is ``True``.
    scaling : bool, optional
        If ``True`` (default), the axes limits are rescaled to the data range of each frame as the animation plays. If ``False``, the axes limits are instead fixed once, at creation time, to the range spanning all frames (only takes effect when a new axis is created, i.e. ``ax`` was ``None``).
    scale_padding : float, optional
        Multiplicative padding factor applied to the axes limits computed from the data range. Default is 1.05.
    **kwargs
        Additional keyword arguments passed to ``ax.plot()`` for the initial line, and to :py:func:`~skhippr.visualization._helpers.parse_and_generate_axis` if a new axis is created.

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

    ax_was_none = ax is None
    ax = parse_and_generate_axis(ax, **kwargs)
    (line,) = robust_plot(ax.plot, xdata[0], ydata[0], **kwargs)

    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    if ax_was_none and not scaling:
        ax.set_xlim(scale_padding * np.nanmin(xdata), scale_padding * np.nanmax(xdata))
        ax.set_ylim(scale_padding * np.nanmin(ydata), scale_padding * np.nanmax(ydata))

    def _update(frame):
        line.set_data(xdata[frame], ydata[frame])
        if scaling:
            try:
                ax.set_xlim(
                    scale_padding * np.min(xdata[frame]),
                    scale_padding * np.max(xdata[frame]),
                )
                ax.set_ylim(
                    scale_padding * np.min(ydata[frame]),
                    scale_padding * np.max(ydata[frame]),
                )
            except ValueError:
                pass
        if anim_title is not None:
            ax.set_title(f"{anim_title} (frame {frame + 1}/{len(xdata)})")
        return (line,)

    anim = FuncAnimation(
        ax.figure,
        _update,
        frames=len(xdata),
        interval=interval,
        repeat=repeat,
    )

    return ax, anim


def parse_and_generate_axis(ax: plt.Axes | None, ndim: int = 2, **kwargs) -> plt.Axes:
    """
    Parse the input axis and generate a new one if necessary.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        The axis to parse. If None, a new axis will be generated.
    ndim : int, optional
        The dimensionality of the axis to create if ``ax`` is ``None``. Use ``2`` (default) for a standard :py:class:`~matplotlib.axes.Axes`, or ``3`` for a 3D :py:class:`~mpl_toolkits.mplot3d.axes3d.Axes3D`. Ignored if ``ax`` is not ``None``.
    **kwargs
        Keyword arguments set as properties on a newly created axis via ``ax.set(**kwargs)``, e.g. ``title``, ``xlabel``, ``ylabel``. Keys that are not valid axis properties are silently ignored. Ignored if ``ax`` is not ``None``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The parsed or newly generated axis.
    """
    if ax is None:

        if ndim <= 2:
            _, ax = plt.subplots()
        elif ndim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        # Try to set arguments passed as keyword arguments to the axis.
        # If a keyword argument is not a valid property of the axis, it will be ignored.
        for key, value in kwargs.items():
            try:
                ax.set(**{key: value})
            except AttributeError as error:
                pass

    return ax


def extract_equation(
    eq: AbstractEquation | EquationSystem,
    usable_class: type[AbstractEquation] = AbstractEquation,
) -> AbstractEquation:
    """
    Extract a usable equation instance from ``eq``.

    Parameters
    ----------
    eq : AbstractEquation or EquationSystem
        Either an instance of ``usable_class`` directly, or an :py:class:`~skhippr.equations.EquationSystem.EquationSystem` that contains one.
    usable_class : type, optional
        The (sub)class of :py:class:`~skhippr.equations.AbstractEquation.AbstractEquation` to look for. Default is :py:class:`~skhippr.equations.AbstractEquation.AbstractEquation` itself.

    Returns
    -------
    equation : AbstractEquation
        ``eq`` itself if it is already an instance of ``usable_class``, or the first equation contained in ``eq.equations`` that is an instance of ``usable_class``.

    Raises
    ------
    ValueError
        If ``eq`` is neither an instance of ``usable_class`` nor an :py:class:`~skhippr.equations.EquationSystem.EquationSystem`, or if it is an :py:class:`~skhippr.equations.EquationSystem.EquationSystem` that does not contain any instance of ``usable_class``.
    """
    if isinstance(eq, usable_class):
        return eq
    elif isinstance(eq, EquationSystem):
        for equation in eq.equations:
            if isinstance(equation, usable_class):
                return equation
        raise ValueError(
            f"EquationSystem does not contain any usable {usable_class} instance"
        )

    raise ValueError(
        f"eq is not an instance of {usable_class} or an EquationSystem containing one"
    )
