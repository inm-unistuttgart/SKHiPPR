from typing import override
import numpy as np
import matplotlib.pyplot as plt

from skhippr.Fourier import Fourier
from skhippr.cycles.hbm import HBMEquation
from skhippr.stability.AbstractStabilityHBM import AbstractStabilityHBM


class ClassicalHill(AbstractStabilityHBM):
    """
    Stability analysis for periodic solutions by solving the Hill eigenvalue problem for the Floquet exponents with subsequent sorting.

    Caution
    -------

    In accordance with the other methods, the Floquet multipliers (not the Floquet exponents) are returned by :py:func:`~skhippr.stability.ClassicalHill.ClassicalHill.determine_eigenvalues`.

    """

    def __init__(
        self,
        fourier: Fourier,
        sorting_method: str,
        tol: float = 0,
        autonomous: bool = False,
    ):
        super().__init__(
            f"Hill {sorting_method} sorting", fourier, tol, autonomous=autonomous
        )
        if sorting_method == "imaginary":
            self.sorting_criterion = self._imaginary_part_criterion
        elif sorting_method == "symmetry":
            self.sorting_criterion = self._weighted_mean
        else:
            raise ValueError(
                f"Unknown sorting method {sorting_method}. Allowed values: 'imaginary', 'symmetry'."
            )

    def fundamental_matrix(self, t_over_period: float, hbm: HBMEquation):
        raise NotImplementedError("Not implemented yet for classical Hill")

    def hill_EVP(
        self, hbm: HBMEquation, visualize: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solves the eigenvalue problem for the Hill matrix and performs sorting to identify the Floquet exponents.

        Computes the eigenvalues of :py:func:`hbm.hill_matrix <skhippr.cycles.hbm.HBMEquation.hill_matrix>` and then chooses those that minimize ``self.sorting_criterion``.

        The Floquet exponents can optionally be visualized in the complex plane.

        Parameters
        ----------

        problem : HBMProblem
            encodes the solution whose stability is sought.
        visualize : bool, optional
            If ``True``, plots the real and imaginary parts of all eigenvalues and the selected Floquet exponents (default is ``False``).

        Returns
        -------

        floquet_exponents : np.ndarray
            Array of computed Floquet exponents (eigenvalues).

        eigenvectors : np.ndarray
            Array of eigenvectors corresponding to the selected Floquet exponents.

        Caution
        -------

        This function returns the Floquet exponents, not the Floquet multipliers. The conversion to Floquet multipliers happens in :py:func:`~skhippr.stability.ClassicalHill.ClassicalHill.determine_eigenvalues`.

        Notes
        -----

        All sorting methods can exhibit numerical issues for negative real Floquet multipliers, which require extra care.
        The function currently does not explicitly handle the case for negative real Floquet multipliers specially, being error-prone in this case.

        """

        hill_matrix = hbm.hill_matrix()
        FE_all, eigenvectors_all = np.linalg.eig(hill_matrix)
        indices = np.argsort(
            [
                self.sorting_criterion((FE_all[i], eigenvectors_all[:, i]))
                for i in range(len(FE_all))
            ]
        )
        floquet_exponents = FE_all[indices[: self.fourier.n_dof]]
        eigenvectors = eigenvectors_all[:, indices[: self.fourier.n_dof]]
        # TODO handle case for negative real floquet multipliers
        if visualize:
            eig_all = np.linalg.eig(hill_matrix)[0]
            plt.plot(np.real(eig_all), np.imag(eig_all), "kx")
            plt.plot(
                np.real(floquet_exponents),
                np.imag(floquet_exponents),
                "o",
                mfc="none",
                mec="r",
            )
            plt.xlabel("Real part")
            plt.ylabel("Imaginary part")
            # plt.show()
        return floquet_exponents, eigenvectors

    @override
    def determine_eigenvalues(self, hbm: HBMEquation) -> np.ndarray:
        """
        Determine the eigenvalues (Floquet multipliers) for the given periodic solution.

        This method computes the Floquet exponents using the Hill eigenvalue problem
        and converts them to Floquet multipliers.

        Parameters
        ----------

        problem : HBMProblem
            The periodic solution to be investigated

        Returns
        -------

        np.ndarray
            Array of Floquet multipliers corresponding to the computed Floquet multipliers, converted from Floquet exponents.
        """

        floquet_exponents, _ = self.hill_EVP(hbm, visualize=False)
        floquet_mult = np.exp(floquet_exponents * 2 * np.pi / hbm.omega)
        return floquet_mult

    def _imaginary_part_criterion(self, eigenpair: tuple[float, np.ndarray]) -> float:
        """
        Returns the absolute value of the imaginary part of the eigenvalue.

        Parameters
        ----------

        eigenpair : tuple[float, np.ndarray]
            A tuple containing an eigenvalue and its corresponding eigenvector.

        Returns
        -------

        float
            The absolute value of the imaginary part of the eigenvalue.
        """
        return abs(np.imag(eigenpair[0]))

    def _weighted_mean(self, eigenpair: tuple[float, np.ndarray]) -> float:
        """
        Compute the weighted mean of the eigenvector as defined in Guillot2020, Eq. (26).

        This method calculates a weighted mean value for the provided eigenvector, where the norms of the eigenvector blocks are weighted by the harmonic index.

        Parameters
        ----------

        eigenpair : tuple[float, np.ndarray]
            A tuple containing the eigenvalue and the corresponding eigenvector.

        Returns
        -------

        float
            The absolute value of the weighted mean of the eigenvector.

        Caution
        -------

        This function performs significantly worse than expected and probably has a bug. It is recommended to use the ``"imaginary"`` setting instead.

        References
        ----------

        Guillot et al. (2020), https://doi.org/10.1016/j.jcp.2020.109477
        """

        eigvec = eigenpair[1]
        if self.fourier.real_formulation:
            eigvec = self.fourier.T_to_cplx_from_real * eigvec

        eigvec = np.reshape(eigenpair[1], (self.fourier.n_dof, -1), order="F")
        weighted_sum = 0
        unweighted_sum = 0
        for k in range(-self.fourier.N_HBM, self.fourier.N_HBM + 1):
            norm = np.linalg.norm(eigvec[:, k + self.fourier.N_HBM])
            weighted_sum += k * norm
            unweighted_sum += norm

        return np.abs(weighted_sum / unweighted_sum)
