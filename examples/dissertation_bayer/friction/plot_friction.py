import numpy as np

from skhippr.visualization.cycles import (
    plot_period,
    plot_phase,
    plot_floquet_multipliers,
)

import tikzplotlib


def plot_hbm_result(hbm, description, path="plots/"):
    path = "plots/"
    r = np.linalg.norm(hbm.residual(update=False))

    # position, velocity, lambda in three plots
    ax_pos = plot_and_save_solution(hbm, f"position_{description}", (0, 1), path)
    ax_vel = plot_and_save_solution(hbm, f"velocity_{description}", (0, 1), path)
    ax_force = plot_and_save_solution(hbm, f"lambda_{description}", (0, 1), path)

    # force law
    ax_forcelaw = plot_phase(hbm, idx=(3, 4))
    ax_forcelaw.set_title(f"force_law_{description}")

    ax_forcelaw.set_xlabel("x3")
    ax_forcelaw.set_ylabel("lambda")
    ax_forcelaw.set_title(f"force law {description} r = {r}")
    ax_forcelaw.legend()
    tikzplotlib.save(f"{path}forcelaw_{description}.tikz")

    return ax_pos, ax_vel, ax_force, ax_forcelaw

    # np.savetxt(f"X_{description}.csv", hbm.X, delimiter=";")


def plot_and_save_solution(hbm, description, idx=(0, 1), path_export=None):
    ax = None
    for i in idx:
        ax = plot_period(hbm, idx=i, ax=ax, label=f"x_{i}")
    ax.set_title(f"{description}")
    ax.legend()
    if path_export is not None:
        tikzplotlib.save(f"{path_export}{description}.tikz")
