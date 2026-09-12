"""Van der Pol oscillator: continuation w.r.t. nu and animation of the resulting phase portrait."""

import matplotlib.pyplot as plt

from examples.vanderpol_minimal import compute_frc

# Visualization
from skhippr.visualization.cycles import (
    plot_phase,
    animate_period,
    animate_floquet_multipliers,
    animate_phase,
)
from skhippr.visualization.continuation import plot_continuation


def main():
    """Demonstration of the continuation of the Van der Pol oscillator w.r.t. nu and animation of the resulting phase portrait.
    This function performs the following steps:

    #. Continuation of the :py:class:`~skhippr.cycles.hbm.HBMSystem` for the ``Vanderpol`` ODE (defined in :py:mod:`examples.vanderpol_minimal`) w.r.t. ``nu``, via :py:func:`examples.vanderpol_minimal.compute_frc`, collecting the branch points.
    #. Animation of the phase portrait, time series and Floquet multipliers along the branch using :py:func:`~skhippr.visualization.cycles.animate_phase`, :py:func:`~skhippr.visualization.cycles.animate_period` and :py:func:`~skhippr.visualization.cycles.animate_floquet_multipliers`.
    #. A plot of ``omega`` over the continuation parameter ``nu`` using :py:func:`~skhippr.visualization.continuation.plot_continuation`.
    #. A phase portrait overlaying every branch point, colored along the branch.

    Returns
    -------
    animation0, animation1, animation2 : matplotlib.animation.FuncAnimation
        The phase, period and Floquet multiplier animations. Each must be kept referenced to keep playing (see the respective ``animate_*`` docstrings).
    """

    hbm, branch = compute_frc(nu_range=(0, 10), max_stepsize=0.5)

    # --- Create animations from the HBMEquations in the continuation branch ---
    _, animation0 = animate_phase(branch, scaling=False)
    _, animation1 = animate_period(branch, scaling=True)
    _, animation2 = animate_floquet_multipliers(
        branch, show_unit_circle=True, scaling=False
    )
    plot_continuation(
        branch,
        plot_fun=lambda point: point.omega,
        xlabel="nu",
        ylabel="omega",
        title="Van der Pol continuation",
    )

    # --- Export animations ---
    # Animations can be saved as a .gif and as video files such as .mp4.
    # Video formats require the user to have FFmpeg installed.
    # save_animation(animation0, "plots/vanderpol_animations/phase_animation.gif")

    # And, finally, a pretty phase plot
    ax = None
    for k, bp in enumerate(branch):
        ax = plot_phase(bp, ax=ax, label=None, color=plt.cm.viridis(k / len(branch)))

    return animation0, animation1, animation2


if __name__ == "__main__":
    animations = main()
    plt.show()
