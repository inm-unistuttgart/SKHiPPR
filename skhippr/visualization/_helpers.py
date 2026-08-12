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


def parse_and_generate_axis(ax: plt.Axes | None, **kwargs) -> tuple[bool, plt.Axes]:
    """
    Parse the input axis and generate a new one if necessary.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        The axis to parse. If None, a new axis will be generated.

    Returns
    -------
    generated_ax : bool
        True if a new axis was generated, False if the input axis was used.
    ax : matplotlib.axes.Axes
        The parsed or newly generated axis.
    """
    if ax is None:
        _, ax = plt.subplots()

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
            if isinstance(equation, HBMEquation):
                return equation
        raise ValueError(
            f"EquationSystem does not contain any usable {usable_class} instance"
        )

    raise ValueError(
        f"eq is not an instance of {usable_class} or an EquationSystem containing one"
    )
