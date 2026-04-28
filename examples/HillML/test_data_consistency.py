import numpy as np
from scipy.linalg import expm

from parse_csv import parse_stability_data

from skhippr.Fourier import Fourier


def main():
    max_error = 0
    max_error_J = 0
    for data_point in parse_stability_data(
        filename="examples/HillML/Duffing_alpha_1_beta_0.5_F_0.3_delta_0.1_N_26_L_256_solvertol_1e-10.csv"
    ):
        FMs = data_point.FMs
        hill_mat = data_point.hill_matrix
        n_dof = data_point.J_coeffs.shape[0]
        N_HBM = (data_point.J_coeffs.shape[-1] - 1) // 4

        fourier = Fourier(N_HBM=N_HBM, n_dof=n_dof, L_DFT=2**9)
        omega = data_point.parameter

        FMs_direct = koopman_hill_direct(hill_mat, n_dof, N_HBM, omega)

        error_FMs = np.linalg.norm(FMs - FMs_direct)

        if error_FMs > max_error:
            max_error = error_FMs

        if error_FMs > 1e-4:
            # raise ValueError(f"Error in Floquet multipliers: {error_FMs}")
            pass

        J_coeffs = data_point.J_coeffs
        J_time = fourier.matrix_inv_DFT(hill_mat)
        J_coeffs_from_time = fourier.DFT(J_time)
        # reshape and order J_coeffs

        errors_Jc = [
            np.linalg.norm(
                J_coeffs[:, :, k] - J_coeffs_from_time[k * n_dof : (k + 1) * n_dof, :]
            )
            for k in range(N_HBM + 1)
        ]
        errors_Js = [
            np.linalg.norm(
                J_coeffs[:, :, 2 * N_HBM + 1 + k]
                - J_coeffs_from_time[
                    (N_HBM + 1 + k) * n_dof : (N_HBM + 1 + k + 1) * n_dof, :
                ]
            )
            for k in range(N_HBM)
        ]
        print(
            f"Error in J_coeffs: {max(errors_Jc)}, {max(errors_Js)}, error in FMs: {error_FMs}"
        )
        max_error_J = max(max_error_J, max(errors_Jc + errors_Js))

    print(f"max error J: {max_error_J}, max error FMs: {max_error}")


def koopman_hill_direct(hill_mat, n_dof, N_HBM, omega):
    """Compute the Floquet multipliers from the Hill matrix using direct Koopman-Hill method,
       which is less accurate than the subharmonic method used to generate the reference data.

    See Eqs. (46), (47) in:
    Bayer et al., "Koopman-Hill stability computation of periodic orbits in
    polynomial dynamical systems using a real-valued quadratic harmonic balance
    formulation", International Journal of Non-Linear Mechanics, 2024.
    DOI: https://doi.org/10.1016/j.ijnonlinmec.2024.104894
    """

    C_small = np.zeros((1, 2 * N_HBM + 1))
    C_small[0, 0] = 1
    C = np.kron(C_small, np.eye(n_dof))

    W_small = np.zeros((2 * N_HBM + 1, 1))
    W_small[0, 0] = 1
    W_small[1 : N_HBM + 1, 0] = 2
    W = np.kron(W_small, np.eye(n_dof))

    T = 2 * np.pi / omega

    # Koopman-Hill formula
    monodromy = C @ expm(hill_mat * T) @ W
    FMs = np.linalg.eigvals(monodromy)

    # Sort the eigenvalues
    if np.imag(FMs[0]) > 1e-10:
        idx = np.argsort(np.imag(FMs))
        FMs = FMs[idx]
    else:
        idx = np.argsort(np.real(FMs))
        FMs = FMs[idx]

    return FMs


if __name__ == "__main__":
    main()
