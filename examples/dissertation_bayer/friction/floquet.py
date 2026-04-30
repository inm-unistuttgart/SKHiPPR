import numpy as np
from skhippr.Fourier import round_to_significant_digits


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
