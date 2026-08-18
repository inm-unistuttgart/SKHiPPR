import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation

from skhippr.equations.AbstractEquation import AbstractEquation
from skhippr.cycles.hbm import HBMEquation, HBMSystem
from skhippr.equations.EquationSystem import EquationSystem
from collections.abc import Sequence, Iterable


def robust_plot(plot_method, *args, **kwargs):
    """
    Plotting wrapper:
    Call ``plot_method(*args, **kwargs)``, where ``plot_method`` is typically
    ``ax.plot`` or ``ax.scatter`` while handling keyword arguments robustly.

    * Keyword arguments that are valid properties of the resulting artist are
    passed through and take effect normally.
    * A keyword argument that is not a valid property is ignored.
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
    xdata,
    ydata,
    ax=None,
    anim_title=None,
    interval=30,
    repeat=True,
    scaling=True,
    scale_padding=1.05,
    **kwargs,
):

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


def parse_and_generate_axis(ax: plt.Axes | None, ndim=2, **kwargs):
    """
    Parse the input axis and generate a new one if necessary.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        The axis to parse. If None, a new axis will be generated.

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


def extract_equation(eq, usable_class=AbstractEquation):
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
