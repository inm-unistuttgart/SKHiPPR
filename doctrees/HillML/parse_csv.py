from collections import namedtuple
from collections.abc import Iterable

import csv
import numpy as np

DataPoint = namedtuple(
    "DataPoint",
    ["parameter", "arclength", "X", "FMs", "FEs", "J_coeffs", "hill_matrix"],
)


def parse_stability_data(filename, omega=None) -> Iterable[DataPoint]:
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=";")
        header = next(reader)

        # number of states and harmonics
        n_dof = sum(entry.startswith("FM") for entry in header[1:])
        N_harmo = (
            (sum(entry.startswith("X") for entry in header[1:]) // n_dof) - 1
        ) // 2

        for row in reader:
            # parameter, arclength
            parameter = float(row[0])
            arclength = float(row[1])

            if omega is None:
                omega = parameter

            # Floquet multipliers and Floquet exponents
            floquet_multipliers = np.array([float(x) for x in row[2 : 2 + n_dof]])

            T = 2 * np.pi / omega
            floquet_exponents = np.log(floquet_multipliers) / T

            # Fourier coefficients of periodic solution
            X = np.array(
                [float(x) for x in row[2 + n_dof : 2 + n_dof * (2 * N_harmo + 2)]]
            )

            # Fourier coefficients of Jacobian
            J_coeffs = np.array(
                [[float(x) for x in row[2 + n_dof * (2 * N_harmo + 2) : -1]]]
            )
            J_coeffs = J_coeffs.reshape((n_dof, n_dof, -1), order="F")

            # Construct block-Toeplitz Hill matrix from Fourier coefficients of Jacobian
            hill_matrix = construct_hill_matrix(J_coeffs, omega)

            yield DataPoint(
                parameter,
                arclength,
                X,
                floquet_multipliers,
                floquet_exponents,
                J_coeffs,
                hill_matrix,
            )


def construct_hill_matrix(J_coeffs, n_dof, N_HBM, omega):
    """Construct the real-valued block-Toeplitz Hill matrix from the Fourier coefficients of the Jacobian.

    Follows the notation of Equations (83) -- (85) in:
    Bayer et al., "Koopman-Hill stability computation of periodic orbits in polynomial
    dynamical systems using a real-valued quadratic harmonic balance formulation",
    International Journal of Non-Linear Mechanics, 2024.
    DOI https://doi.org/10.1016/j.ijnonlinmec.2024.104894:
    """

    # Components
    J0 = J_coeffs[:, :, 0]
    Jc = J_coeffs[:, :, 1 : N_HBM + 1]
    Js = J_coeffs[:, :, N_HBM + 1 :]

    Jcs = np.reshape(
        J_coeffs[:, :, 1:], shape=(n_dof, 2 * N_HBM * n_dof, -1), order="F"
    )

    # block-wise transposition
    Jcs_tr = np.transpose(J_coeffs[:, :, 1], (0, 2, 1))
    Jcs_tr = np.reshape(Jcs_tr, shape=(2 * N_HBM * n_dof, n_dof, -1), order="F")

    # Construct block-Hankel component matrices block by block
    Kc = np.zeros((n_dof * N_HBM, n_dof * N_HBM))
    Ks = np.zeros((n_dof * N_HBM, n_dof * N_HBM))

    for K, J in zip([Kc, Ks], [Jc, Js]):
        for i in range(N_HBM):
            for j in range(N_HBM):
                block = 0.5 * J[:, :, (i + 1) + (j + 1) - 1]
                K[i * n_dof : (i + 1) * n_dof, j * n_dof : (j + 1) * n_dof] = block

    # construct block-Toeplitz components block by block
    Tc = np.zeros((n_dof * N_HBM, n_dof * N_HBM))
    Ts = np.zeros((n_dof * N_HBM, n_dof * N_HBM))

    for i in range(N_HBM):
        for j in range(N_HBM):

            if i == j:
                block_c = J0
                block_s = np.zeros((n_dof, n_dof))
            else:
                block_c = 0.5 * Jc[:, :, abs(i - j) - 1]
                block_s = 0.5 * np.sign(j - i) * Js[:, :, abs(i - j) - 1]

            Tc[i * n_dof : (i + 1) * n_dof, j * n_dof : (j + 1) * n_dof] = block_c
            Ts[i * n_dof : (i + 1) * n_dof, j * n_dof : (j + 1) * n_dof] = block_s

    Kc = np.zeros((n_dof * N_HBM, n_dof * N_HBM))
    Ks = np.zeros((n_dof * N_HBM, n_dof * N_HBM))

    # Derivative matrix
    D = np.zeros((2 * N_HBM + 1, 2 * N_HBM + 1))
    D[2 : (N_HBM + 1), (N_HBM + 1) :] = np.diag(np.arange(1, N_HBM + 1) * omega)
    D[(N_HBM + 1) :, 2 : (N_HBM + 1)] = -np.diag(np.arange(1, N_HBM + 1) * omega)

    D = np.kron(D, np.eye(n_dof))

    # assemble the Hill matrix
    # D is only nonzero in the mid-right and bottom-mid blocks,
    # so '+=' is not necessary everywhere else
    hill_matrix = -D
    hill_matrix[:n_dof, :n_dof] = J0
    hill_matrix[n_dof:, :n_dof] = Jcs_tr
    hill_matrix[n_dof:, n_dof:] = 0.5 * Jcs

    hill_matrix[n_dof : n_dof * (N_HBM + 1), n_dof : n_dof * (N_HBM + 1)] = Kc + Tc
    hill_matrix[n_dof : n_dof * (N_HBM + 1), n_dof * (N_HBM + 1) :] += Ks - Ts
    hill_matrix[n_dof * (N_HBM + 1) :, n_dof : n_dof * (N_HBM + 1)] += Ks + Ts
    hill_matrix[n_dof * (N_HBM + 1) :, n_dof * (N_HBM + 1) :] = Tc - Kc
