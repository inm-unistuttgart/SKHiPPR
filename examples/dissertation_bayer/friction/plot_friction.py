import numpy as np
import matplotlib.pyplot as plt

from skhippr.visualization.cycles import (
    plot_period,
    plot_phase,
    plot_floquet_multipliers,
)

import tikzplotlib

from floquet import sort_FMs
from friction_error import hbm_error_freq, hbm_error_time, FM_error


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
    if path is not None:
        tikzplotlib.save(f"{path}forcelaw_{description}.tikz")

    # phase diagram
    ax_phasediagram = None
    for i in range(2):
        ax_phasediagram = plot_phase(hbm, idx=(i, i + 2), label=f"mass {i}")
        ax_phasediagram.set_title(f"phase_diagram_{description}")
    if path is not None:
        tikzplotlib.save(f"{path}phase_diagram_{description}.tikz")

    return ax_pos, ax_vel, ax_force, ax_forcelaw, ax_phasediagram


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


def plot_FM_error(hbms, FM_ref, description, path="plots/"):
    fig, ax = plt.subplots(1, 1)

    err = FM_error(hbms, FM_ref)
    for l in range(err.shape[0]):
        ax.plot([hbm.fourier.N_HBM for hbm in hbms], err[l, :], "-", label=f"FM {l}")

    ax.set_yscale("log")
    ax.set_title(description)
    ax.legend(loc="upper right")
    if path is not None:
        tikzplotlib.save(f"{path}FM_error_{description}.tikz")
    return ax


def plot_hbm_convergence(hbms, description, path="plots/"):
    _, ax_time = plt.subplots(1, 1)
    _, ax_freq = plt.subplots(1, 1)

    err_freq = hbm_error_freq(hbms, hbms[-1])
    err_time = hbm_error_time(hbms, hbms[-1].x_time())

    for ax, err, label in zip(
        (ax_freq, ax_time), (err_freq, err_time), ("FCs", "time")
    ):

        for l in range(err.shape[0]):
            ax.plot(
                [hbm.fourier.N_HBM for hbm in hbms],
                err[l, :],
                "-",
                label=f"label x{l}",
            )

        ax.set_yscale("log")
        ax.set_title(description)
        ax.legend(loc="upper right")
        ax.set_xlabel("N")
        ax.set_ylabel(f"error")
        ax.set_title(f"HBM {label} convergence {description}")
        if path is not None:
            tikzplotlib.save(f"{path}HBM_error_{label}_{description}.tikz")
        return ax
