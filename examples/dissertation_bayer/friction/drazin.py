import numpy as np

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
