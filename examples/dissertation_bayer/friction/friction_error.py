import numpy as np
from floquet import sort_FMs


def hbm_error_freq(hbms, hbm_ref):
    """Compute errors of HBM solutions in Fourier domain against X_ref.

    Parameters
    ----------
    hbms: list of HBMEquationDAE
        List of HBM solutions to compare, ordered by increasing N_HBM.
     X_ref: numpy array (n_dof x (2*N_ref+1))
        Reference Fourier coefficients to compare against.

    Returns
    -------
    err: numpy array (n_dof x len(hbms))
        Errors of HBM solutions in Fourier domain (norm of Fourier coefficients) for each state variable

    """
    err = np.zeros((hbm_ref.fourier.n_dof, len(hbms)))

    for k, hbm in enumerate(hbms):

        X_comp = hbm_ref.fourier.resize_coefficients(hbm.X)
        X_comp[np.abs(X_comp) <= 1e-17] = 0
        e_comp = np.reshape(X_comp - hbm_ref.X, (hbm.fourier.n_dof, -1), order="F")

        err[:, k] = np.linalg.norm(e_comp, axis=1)


def hbm_error_time(hbms, x_time_ref):
    """Compute errors of HBM solutions in time and Fourier domain against x_time_ref.

    Parameters
    ----------
    hbms: list of HBMEquationDAE
        List of HBM solutions to compare, ordered by increasing N_HBM.
     x_time_ref: numpy array (n_dof x L_DFT)
        Reference time-domain solution to compare against.
    time_domain: bool, optional
        Whether to compute errors in the time domain (max norm) or in the Fourier domain (L2 norm). Default is False.

    Returns
    -------
    err: numpy array (n_dof x len(hbms))
        Errors of HBM solutions in time domain (max norm) for each state variable or in the freq domain (norm of Fourier coefficients) for each state variable

    """

    err = np.zeros((hbms[0].fourier.n_dof, len(hbms)))
    for k, hbm in enumerate(hbms):
        x_time = hbm.x_time()
        err[:, k] = np.max(np.abs(x_time - x_time_ref), axis=1)


def FM_error(hbms, FM_ref):
    """Compute errors of Floquet multipliers of HBM solutions against FM_ref.

    Parameters
    ----------
    hbms: list of HBMEquationDAE+
        List of HBM solutions to compare, ordered by increasing N_HBM.
    FM_ref: numpy array (n_dof,)
        Reference Floquet multipliers to compare against.

    Returns
    -------
    err: numpy array (n_dof x len(hbms))
        Errors of (sorted) Floquet multipliers for each state variable and each HBM solution.
    """
    FM_ref = sort_FMs(FM_ref)
    err = np.zeros((hbms[0].fourier.n_dof, len(hbms)))
    for k, hbm in enumerate(hbms):

        FMs = sort_FMs(hbm.eigenvalues)
        err[:, k] = np.abs(FMs - FM_ref)
