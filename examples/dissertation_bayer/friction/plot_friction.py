import numpy as np
import matplotlib.pyplot as plt

from skhippr.visualization.cycles import (
    plot_period,
    plot_phase,
    plot_floquet_multipliers,
)

import tikzplotlib

from floquet import sort_FMs


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


def plot_FM_convergence(hbms, description, path="plots/"):
    fig, ax = plt.subplots(1, 1)

    # unit circle
    phis = np.linspace(0, 2 * np.pi, 250)
    ax.plot(np.cos(phis), np.sin(phis))

    # Sort FMs into one numpy array
    FMs_all = np.zeros((hbms[0].fourier.n_dof, len(hbms)), dtype=complex)
    for k, hbm in enumerate(hbms):
        FMs = sort_FMs(FMs=hbm.eigenvalues)
        FMs_all[:, k] = FMs

    # Plot sorted FMs state by state
    for l in range(FMs_all.shape[0]):
        ax.plot(np.real(FMs_all[l, :]), np.imag(FMs_all[l, :]), "-x", label=f"FM {l}")

    # Plot highest-N FMs as reference
    ax.plot(np.real(FMs), np.imag(FMs), "o", label=f"ref(N={hbm.fourier.N_HBM})")
    ax.set_aspect("equal")
    ax.set_title(description)
    ax.legend(loc="upper left")
    if path is not None:
        tikzplotlib.save(f"{path}FMs_{description}.tikz")
    return ax
