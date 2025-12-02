from typing import override
import numpy as np
from scipy import sparse
from scipy.linalg import expm

from skhippr.Fourier import Fourier
from skhippr.cycles.hbm import HBMEquation
from skhippr.stability.AbstractStabilityHBM import AbstractStabilityHBM


class KoopmanHillProjection(AbstractStabilityHBM):
    """
    Direct Koopman Hill projection method for stability analysis of periodic solutions.

    This subclass of :py:class:`~skhippr.stability.AbstractStabilityHBM.AbstractStabilityHBM` implements the abstract method :py:func:`~skhippr.stability.AbstractStabilityHBM.AbstractStabilityHBM.fundamental_matrix` of its parent class using the direct Koopman-Hill projection formula: ::

        fundamental_matrix = C @ D_time(t) @ np.expm(hill_matrix*t) @ W

    Upon initialization, the projection matrices ``self.C`` and ``self.W`` are constructed once-and-for-all depending on the formulation (real or complex), and stored as attributes.

    Notes
    -----
    It is possible to overwrite the direct projection matrix ``self.C`` manually with a nontrivial choice (which must respect the normalization constraint, see Bayer & Leine 2023) after instantiation of the object. However, all such nontrivial choices show reduced convergence (see Bayer & Leine 2025).

    Parameters
    ----------
    fourier : Fourier
        The Fourier object containing harmonic balance settings and transformation matrices.
    tol : float, optional
        Tolerance for stability computations (default is 0).
    autonomous : bool, optional
        Whether the system is autonomous (default is False).


    Attributes:
    -----------
    C : np.ndarray
        Direct projection matrix for the Koopman Hill method. Is zero almost everywhere. In the real formulation, an identity matrix is in the first block. In the complex formulation, an identity matrix is in the central block.
    W : np.ndarray
        Initial condition matrix for the Koopman Hill method. In the complex formulation, it is a stack of identity matrices. In the real formulation, it consists of an identity matrix, then a stack of identity matrices multiplied by 2, then a stack of zeros.


    References
    ----------

    * Complex formulation: Bayer and Leine (2023): *Sorting-free Hill-based stability analysis of periodic solutions through Koopman analysis*. Nonlinear Dyn 111, 8439–8466, https://doi.org/10.1007/s11071-023-08247-7.
    * Real formulation: Bayer et al. (2024): *Koopman-Hill Stability Computation of Periodic Orbits in Polynomial Dynamical Systems Using a Real-Valued Quadratic Harmonic Balance Formulation*. International Journal of Non-Linear Mechanics, 167, 104894, https://doi.org/10.1016/j.ijnonlinmec.2024.104894.
    * Convergence guarantee: Bayer and Leine (2025, preprint): *Explicit error bounds and guaranteed convergence of the Koopman-Hill projection stability method for linear time-periodic dynamics*, https://arxiv.org/abs/2503.21318
    """

    def __init__(self, fourier: Fourier, tol: float = 0, autonomous=False):
        super().__init__("Koopman-Hill projection", fourier, tol, autonomous=autonomous)
        eye = np.eye(self.fourier.n_dof)

        """ Construct projection matrices. 
        First in complex-valued formulation, 
        transform to real-valued if necessary"""

        C0 = np.zeros((1, 2 * self.fourier.N_HBM + 1))
        C0[0, self.fourier.N_HBM] = 1
        W0 = np.ones((2 * self.fourier.N_HBM + 1, 1))

        self.C = np.kron(C0, eye)
        self.W = np.kron(W0, eye)

        if self.fourier.real_formulation:
            self.W = self.fourier.T_to_real_from_cplx @ self.W
            self.C = self.C @ self.fourier.T_to_cplx_from_real

    def fundamental_matrix(self, t_over_period: float, hbm: HBMEquation) -> np.ndarray:
        """
        Compute the fundamental solution matrix for the given periodic solution and normalized time using direct Koopman-Hill projection.

        The fundamental matrix is calculated as: ::

            self.C @ D_time(t) @ np.expm(hill_matrix * t) @ self.W

        where:

            * ``C`` and ``W`` are projection matrices that were computed once-and-for-all at initialization.
            * :py:func:`~skhippr.cycles.hbm.hbmProblem.hill_matrix` is obtained from the Jacobian matrix of the :py:class:`~skhippr.cycles.hbm.hbmProblem`.
            * :py:func:`~skhippr.stability.KoopmanHillProjection.KoopmanHillProjection.D_time` scales the projection matrix -- Only relevant if ``t_over_period`` is non-integer and ``self.C`` has been manually modified to be nontrivial.

        Parameters
        ----------

        t_over_period : float
            The time normalized over the period (i.e., t/T, where T is the period).
        problem : HBMProblem
            A solved :py:class:`~skhippr.cycles.hbm.hbmProblem`, encoding the periodic solution.

        Returns
        -------

        np.ndarray
            The computed fundamental matrix as a NumPy array.

        """

        C = self.C_time(t_over_period)
        hill_matrix = hbm.hill_matrix()
        t = t_over_period * 2 * np.pi / hbm.omega

        funda_mat = C @ expm(hill_matrix * t) @ self.W

        return funda_mat

    def C_time(self, t_over_period: float) -> np.ndarray:
        """
        Compute the time-dependent, scaled projection matrix C at a given, nontrivial normalized time.

        * If ``t_over_period`` is integer, this function always returns ``self.C``.
        * If ``self.C`` has not been modified manually after initialization, this function has no effect and returns ``self.C``.

        Parameters
        ----------

        t_over_period : float
            The normalized time over the period (e.g., t / T), where T is the period.

        Returns
        -------

        np.ndarray
            The scaled projection matrix C at the specified normalized time.

        """

        if t_over_period.is_integer():
            # shortcut
            return self.C
        else:
            return self.C @ self.D_time(t_over_period)

    def D_time(self, t_over_period: float) -> np.ndarray:
        """
        Constructs the time-dependent transformation matrix D(t) for the projection matrix at arbitrary (non-integer) times t.

        For the complex-valued formulation, the ``k``-th block column of ``self.C`` is scaled by ``np.exp(k * omega * t)`` and the resulting matrix is diagonal (see Bayer&Leine, 2023). The real-valued formulation follows by multiplication with the transformation matrix :py:attr:`HBMProblem.T_to_cplx_from_real <skhippr.cycles.hbm.hbmProblem.T_to_cplx_from_real>`.

        Parameters
        ----------
        t_over_period : float
            The normalized time, expressed as a fraction of the period (t/T).

        Returns
        -------
        np.ndarray
            The time-dependent transformation matrix D(t), with shape
            (n_dof * n_coeff, n_dof * n_coeff), where n_coeff depends on the Fourier formulation.

        References
        ----------
        Bayer and Leine (2023): *Sorting-free Hill-based stability analysis of periodic solutions through Koopman analysis*. Nonlinear Dyn 111, 8439–8466, https://doi.org/10.1007/s11071-023-08247-7.

        """
        if self.fourier.real_formulation:
            omega_ts = 2 * np.pi * t_over_period * np.arange(1, self.fourier.N_HBM + 1)
            coss = np.diag(np.cos(omega_ts))
            sins = np.diag(np.sin(omega_ts))
            D = np.vstack(
                (
                    np.hstack(([[1]], np.zeros((1, 2 * self.fourier.N_HBM)))),
                    np.hstack((np.zeros((self.fourier.N_HBM, 1)), coss, sins)),
                    np.hstack((np.zeros((self.fourier.N_HBM, 1)), -sins, coss)),
                )
            )
        else:
            D = np.diag(
                np.exp(
                    2j
                    * np.pi
                    * t_over_period
                    * np.arange(-self.fourier.N_HBM, self.fourier.N_HBM + 1)
                )
            )
        return np.kron(D, np.eye(self.fourier.n_dof))

    @override
    def error_bound(self, t, a, b):
        """
        Compute the theoretical error bound for the fundamental solution matrix based on the exponential decay of Fourier coefficient matrices as returned by :py:class:`~skhippr.cycles.hbm.hbmProblem.exponential_decay_parameters`.

        According to (Bayer & Leine, 2025), the error bound is given by: ::

            || E(t) || < (2*exp(-b))**N_HBM * exp(4*a*t)

        Parameters
        ----------

        t : float
            The time at which the error bound is evaluated.
        a : float
            Factor in front of the exponential decay
        b : float
            Exponential decay

        Returns
        -------

        float
            The computed error bound for the fundamental solution matrix.

        """
        return (2 * np.exp(-b)) ** self.fourier.N_HBM * (np.exp(4 * a * np.abs(t)) - 1)


class KoopmanHillSubharmonic(KoopmanHillProjection):
    """
    Subharmonic Koopman Hill projection method for stability analysis of periodic solutions.

    This class modifies its parent class :py:class:`~skhippr.stability.KoopmanHillProjection.KoopmanHillProjection` to implement the subharmonic Koopman-Hill projection formula: ::

        fundamental_matrix = C @ D_time(t) @ np.expm(hill_matrix*t) @ W + C_subh @ D_sub(t) @ np.expm(hill_sub*t) @ W_subh

    The subharmonic formulation is more accurate (error bound decays twice as fast) at the cost of approximately twice the computation time of the direct method.

    Upon initialization, the projection matrices ``self.C`` and ``self.W`` as well as ``self.C_subh`` and ``self.W_subh`` are constructed once-and-for-all depending on the formulation (real or complex), and stored as attributes.

    Parameters
    ----------
    fourier : Fourier
        The Fourier object containing harmonic balance settings and transformation matrices.
    tol : float, optional
        Tolerance for stability computations (default is 0).
    autonomous : bool, optional
        Whether the system is autonomous (default is False).


    Attributes:
    -----------
    C : np.ndarray
        Projection matrix for the non-subharmonic components of the subharmonic Koopman Hill method.
    C_subh : ndarray
        Projection matrix for the subharmonic components of the subharmonic Koopman Hill method. Is the negative of ``self.C``, with the block corresponding to the 0-th frequency removed.
    W : np.ndarray
        Initial condition matrix for the non-subharmonic components of the subharmonic Koopman Hill method. Is the same as in :py:class:`~skhippr.stability.KoopmanHillProjection.KoopmanHillProjection`.
    W_subh: np.ndarray
        Initial condition matrix for the subharmonic components of the subharmonic Koopman Hill method. Is equal to ``self.W``, with the block corresponding to the 0-th frequency removed.


    References
    ----------

    * Bayer and Leine (2025, preprint): *Explicit error bounds and guaranteed convergence of the Koopman-Hill projection stability method for linear time-periodic dynamics*, https://arxiv.org/abs/2503.21318
    """

    """Subharmonic Koopman-Hill projection"""

    def __init__(self, fourier: Fourier, tol: float = 0, autonomous=False):
        super().__init__(fourier=fourier, tol=tol, autonomous=autonomous)
        self.label = "Subharmonic " + self.label
        eye = np.eye(self.fourier.n_dof)
        """Construct subharmonic projection matrices"""
        if self.fourier.real_formulation:
            C0 = np.hstack(
                (
                    np.ones((1, self.fourier.N_HBM + 1)),
                    np.zeros((1, self.fourier.N_HBM)),
                )
            )

            W0 = C0.T.copy()
            W0[1 : self.fourier.N_HBM + 1, :] = 2

        else:
            W0 = np.ones((2 * self.fourier.N_HBM + 1, 1))
            C0 = W0.T

        self.W = np.kron(W0, eye)
        self.C = np.kron(C0, eye)

        self.W_subh = np.kron(W0[1:, :], eye)
        self.C_subh = -np.kron(C0[:, 1:], eye)

    def fundamental_matrix(self, t_over_period: float, hbm: HBMEquation) -> np.ndarray:
        """
        Compute the fundamental solution matrix for the given periodic solution and normalized time using subharmonic Koopman-Hill projection.

        The fundamental matrix is calculated as: ::

            self.C @ D_time(t) @ np.expm(hill_matrix * t) @ self.W + self.C_subh_time(t) @ np.expm(hill_subh * t) @ self.W_subh

        where:

            * ``C``, ``W``, ``W_subh`` are projection matrices that were computed once-and-for-all at initialization.
            * :py:func:`~skhippr.stability.KoopmanHillProjection.KoopmanHillSubharmonic.C_subh_time` is computed by scaling ``C_subh`` for non-integer times.
            * :py:func:`~skhippr.cycles.hbm.hbmProblem.hill_matrix` is obtained from the Jacobian matrix of the :py:class:`~skhippr.cycles.hbm.hbmProblem`.
            * :py:func:`~skhippr.stability.KoopmanHillProjection.KoopmanHillSubharmonic.hill_subh` is obtained from the Hill matrix by eliminating the constant row and column and shifting the frequencies by 0.5.

        Parameters
        ----------

        t_over_period : float
            The time normalized over the period (i.e., t/T, where T is the period).
        problem : HBMProblem
            A solved :py:class:`~skhippr.cycles.hbm.hbmProblem`, encoding the periodic solution.

        Returns
        -------

        np.ndarray
            The computed fundamental matrix as a NumPy array.

        """
        # print(f"omega={omega}]")
        C = self.C_time(t_over_period)
        C_subh = self.C_subh_time(t_over_period=t_over_period)

        Phi_t = super().fundamental_matrix(t_over_period=t_over_period, hbm=hbm)

        hill_mat_subh = self.hill_subh(equ=hbm)
        t = 2 * np.pi / hbm.omega * t_over_period
        Phi_t += C_subh @ expm(hill_mat_subh * t) @ self.W_subh

        return Phi_t

    def C_subh_time(self, t_over_period: float) -> np.ndarray:
        """
        Computes the scaled subharmonic projection matrix at a given normalized time.

        Caution
        -------
        In contrast to :py:func:`~skhippr.stability.KoopmanHillProjection.KoopmanHillProjection.C_time`, the scaling must be considered even for integer ``t_over_period``.

        Parameters
        ----------
        t_over_period : float
            The normalized time over the period.

        Returns
        -------

        np.ndarray
            The scaled subharmonic projection matrix evaluated at the specified normalized time.

        """

        if t_over_period.is_integer():
            # shortcut
            return ((-1) ** (t_over_period + 1)) * self.C[:, self.fourier.n_dof :]
        elif self.fourier.real_formulation:
            omega_t = (
                2 * np.pi * t_over_period * (np.arange(1, self.fourier.N_HBM + 1) - 0.5)
            )
            C_subh = -np.hstack((np.cos(omega_t), np.sin(omega_t)))
        else:
            C_subh = -np.exp(
                2j
                * np.pi
                * t_over_period
                * (np.arange(-self.fourier.N_HBM, self.fourier.N_HBM) + 0.5)
            )
        return np.kron(C_subh, np.eye(self.fourier.n_dof))

        if C is None:
            C = self.C_time(t_over_period)

    def hill_subh(self, equ: HBMEquation) -> np.ndarray:
        """
        Constructs the subharmonic Hill matrix for the given HBM problem.

        As described in (Bayer and Leine, 2025), the subharmonic Hill matrix is given by the even row and column blocks of the Hill matrix evaluated with the halved frequency.

        However, practically, the subharmonic Hill matrix is constructed immediately from the :py:func:`~skhippr.cycles.hbm.hbmProblem.hill_matrix` by removing the 0-frequency row and column and shifting all frequency terms by ``omega/2``.

        Parameters
        ----------

        problem : HBMProblem
            The harmonic balance method (HBM) problem instance containing the periodic solution and the Hill matrix.

        Returns
        -------

        np.ndarray
            The subharmonic Hill matrix as a NumPy array. It has ``n_dof`` fewer rows / columns than the Hill matrix itself.

        References
        ----------
        * Bayer and Leine, 2025: Subharmonic formulation
        * Bayer et al., 2024, Appendix: Details on the block structure real-valued formulation.
        """

        hill_mat = equ.hill_matrix()
        if self.fourier.real_formulation:
            # Split the Hill matrix into blocks for const, cos, sin
            blocks = []
            idx_split = [
                0,
                self.fourier.n_dof,
                self.fourier.n_dof * (self.fourier.N_HBM + 1),
                hill_mat.shape[0],
            ]
            for k in range(len(idx_split) - 1):
                blocks.append(
                    [
                        hill_mat[
                            idx_split[k] : idx_split[k + 1],
                            idx_split[l] : idx_split[l + 1],
                        ]
                        for l in range(len(idx_split) - 1)
                    ]
                )

            # and then identify 0.5*J_c, 0.5*J_s, K_c, K_s, T_c, T_s (cf. Bayer2024, Appendix)
            Jc = blocks[0][1]
            Js = blocks[0][2]
            Tc = 0.5 * (blocks[1][1] + blocks[2][2])
            Ts = 0.5 * (blocks[1][2] - blocks[2][1])
            Kc = 0.5 * (blocks[1][1] - blocks[2][2])
            Ks = 0.5 * (blocks[1][2] + blocks[2][1])

            # Construct their subharmonic pendants
            # Tc = Tc
            Ts += 0.5 * equ.omega * np.eye(self.fourier.n_dof * self.fourier.N_HBM)

            Kc = np.vstack((Jc, Kc[: -self.fourier.n_dof, :]))
            Ks = np.vstack((Js, Ks[: -self.fourier.n_dof, :]))

            # Reconstruct the subharmonic Hill matrix
            return np.block([[Kc + Tc, Ks + Ts], [Ks - Ts, Tc - Kc]])

        else:
            Hill_subh = hill_mat[self.fourier.n_dof :, self.fourier.n_dof :]
            Hill_subh = Hill_subh + 0.5j * equ.omega * np.eye(
                self.fourier.n_dof * 2 * self.fourier.N_HBM
            )
        return Hill_subh

    def error_bound(self, t, a, b):
        """
        Compute the theoretical error bound for the fundamental solution matrix based on the exponential decay of Fourier coefficient matrices as returned by :py:class:`~skhippr.cycles.hbm.hbmProblem.exponential_decay_parameters`.

        According to (Bayer & Leine, 2025), the error bound in the subharmonic formulation is given by: ::

            || E(t) || < (2*exp(-b))**(2*N_HBM) * exp(4*a*t)

        Parameters
        ----------

        t : float
            The time at which the error bound is evaluated.
        a : float
            Factor in front of the exponential decay
        b : float
            Exponential decay

        Returns
        -------

        float
            The computed error bound for the fundamental solution matrix.

        """
        return (2 * np.exp(-b)) ** (2 * self.fourier.N_HBM) * (
            np.exp(4 * a * np.abs(t)) - 1
        )
