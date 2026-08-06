from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from skhippr.Fourier import round_to_significant_digits

from skhippr.cycles.hbm import HBMEquationDAE
from skhippr.stability.KoopmanHillProjection import KoopmanHillDAE

from drazin import plot_drazin_and_ratio


def sort_FMs(FMs, significant_digits=2):

    FMs_rounded = np.zeros_like(FMs)
    for k, FM in enumerate(FMs):
        FMs_rounded[k] = round_to_significant_digits(
            FM, significant_digits=significant_digits
        )
    idx_sort = np.lexsort((np.angle(FMs_rounded), np.abs(FMs_rounded)))
    FMs = FMs[idx_sort]
    FMs_rounded = FMs_rounded[idx_sort]
    # Separate complex and real eigenvalues
    FMs = np.hstack((FMs[np.imag(FMs_rounded) == 0], FMs[np.imag(FMs_rounded) != 0]))
    return FMs


def KH_other_N(hbm, N_other, description=""):
    fourier_other = hbm.fourier.__replace__(N_HBM=N_other)
    X_other = fourier_other.resize_coefficients(hbm.X)
    hbm_other = copy(hbm)
    hbm_other = HBMEquationDAE(
        hbm_other.ode,
        hbm_other.omega,
        fourier_other,
        initial_guess=X_other,
        stability_method=KoopmanHillDAE(
            fourier_other,
            hbm.stability_method.tol,
            autonomous=False,
            tol_drazin=hbm.stability_method.tol_drazin,
        ),
    )
    _ = hbm_other.hill_matrix(update=True)
    _, ax = plt.subplots(1, 1)
    plot_drazin_and_ratio(
        [hbm_other], 1e-7, f"Drazin EVs N_Hill = {N_other} {description}", ax_drazin=ax
    )
