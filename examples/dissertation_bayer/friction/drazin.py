import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

from scipy.linalg import lu_factor, lu_solve

from skhippr.stability.KoopmanHillProjection import drazin
from skhippr.cycles.hbm import HBMEquationDAE


def compute_drazin_ratio(
    hbm: HBMEquationDAE,
    ax=None,
    tol_cond=1e6,
    tol_drazin=1e-7,
) -> float:

    hill_matrix = hbm.hill_matrix(update=True)
    mass_matrix = hbm.M()

    # Copy&pasted from KoopmanHillDAE.generalized_exponential()
    a_vals = [1.0, 10.0, 0.1, 100, 0.01, 1000, 0.001]
    success = False
    for a in a_vals:
        pencil = a * mass_matrix - hill_matrix
        if np.linalg.cond(pencil) < tol_cond:
            success = True
            # print(f"N = {N}: a = {a}")
            break
    if not success:
        raise RuntimeError(
            f"Could not find suitable scaling factor 'a' for Drazin inverse with condition < {tol_cond}."
        )

    pencil_lu = lu_factor(a * mass_matrix - hill_matrix)
    pencil_M = lu_solve(pencil_lu, mass_matrix)
    _, ratio = drazin(pencil_M, tol_drazin, ax_plot=ax)
    # print(f"Ratio of Drazin inverse: {ratio}")
    return ratio


def plot_drazin_and_ratio(
    hbms,
    tol_drazin,
    description,
    ratio_limits=(),
    ax_drazin=None,
    ax_ratio=None,
    path_drazin=None,
    path_ratio=None,
):

    drazin_ratios = []
    Ns_HBM = []

    # Drazin inverse analysis + plotting for all HBMs
    for hbm in hbms:
        drazin_ratios.append(
            compute_drazin_ratio(hbm, ax_drazin, tol_drazin=tol_drazin)
        )
        Ns_HBM.append(hbm.fourier.N_HBM)

    if ax_drazin is not None:

        ax_drazin.axhline(tol_drazin, linestyle="--")
        ax_drazin.set_xlabel("n*(2*N+1)")
        ax_drazin.set_ylabel("magnitude of eigenvalue")
        ax_drazin.set_title(f"Drazin eigenvalues {description}")

        if path_drazin is not None:
            tikzplotlib.save(f"{path_drazin}_drazin_{description}.tikz")

    if ax_ratio is not None:
        ax_ratio.plot(Ns_HBM, drazin_ratios, "-x")
        for limit in ratio_limits:
            ax_ratio.axhline(limit, linestyle="--")
        ax_ratio.set_title(f"Drazin ratio {description}")

        if path_ratio is not None:
            tikzplotlib.save(f"{path_ratio}_drazinratio_{description}.tikz")
