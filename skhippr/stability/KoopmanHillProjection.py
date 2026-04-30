from typing import override
import numpy as np
from scipy.linalg import (
    expm,
    schur,
    solve_triangular,
    lu_factor,
    lu_solve,
    solve_sylvester,
)
import warnings

from skhippr.Fourier import Fourier, round_to_significant_digits
from skhippr.cycles.hbm import HBMEquation, HBMEquationDAE
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

    def fundamental_matrix(
        self, t_over_period: float, hbm: HBMEquation | np.ndarray, omega=None
    ) -> np.ndarray:
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
        if isinstance(hbm, np.ndarray):
            hill_matrix = hbm
        else:
            hill_matrix = hbm.hill_matrix()
            omega = hbm.omega
        t = t_over_period * 2 * np.pi / omega

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

    def fundamental_matrix(
        self, t_over_period: float, hbm: HBMEquation | np.ndarray, omega=None
    ) -> np.ndarray:
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
        if omega is None:
            omega = hbm.omega
        # print(f"omega={omega}]")
        C = self.C_time(t_over_period)
        C_subh = self.C_subh_time(t_over_period=t_over_period)

        Phi_t = super().fundamental_matrix(
            t_over_period=t_over_period, hbm=hbm, omega=omega
        )

        hill_mat_subh = self.hill_subh(equ=hbm, omega=omega)
        t = 2 * np.pi / omega * t_over_period
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

    def hill_subh(self, equ: HBMEquation | np.ndarray, omega=None) -> np.ndarray:
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

        if isinstance(equ, np.ndarray):
            hill_mat = equ
        else:
            hill_mat = equ.hill_matrix(update=False)
            omega = equ.omega
        if self.fourier.real_formulation:
            # Split the Hill matrix into blocks
            Jc, Js, Tc, Ts, Kc, Ks = self.determine_toeplitz_hankel_blocks(hill_mat)
            # Construct their subharmonic pendants
            # Tc = Tc
            Ts += 0.5 * omega * np.eye(self.fourier.n_dof * self.fourier.N_HBM)

            Kc = np.vstack((Jc, Kc[: -self.fourier.n_dof, :]))
            Ks = np.vstack((Js, Ks[: -self.fourier.n_dof, :]))

            # Reconstruct the subharmonic Hill matrix
            return np.block([[Kc + Tc, Ks + Ts], [Ks - Ts, Tc - Kc]])

        else:
            Hill_subh = hill_mat[self.fourier.n_dof :, self.fourier.n_dof :]
            Hill_subh = Hill_subh + 0.5j * omega * np.eye(
                self.fourier.n_dof * 2 * self.fourier.N_HBM
            )
        return Hill_subh

    def determine_toeplitz_hankel_blocks(self, hill_mat_real):
        # if not self.fourier.real_formulation:
        #     raise ValueError(
        #         "This method is only applicable for real-valued formulation."
        #     )

        blocks = []
        idx_split = [
            0,
            self.fourier.n_dof,
            self.fourier.n_dof * (self.fourier.N_HBM + 1),
            hill_mat_real.shape[0],
        ]
        for k in range(len(idx_split) - 1):
            blocks.append(
                [
                    hill_mat_real[
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

        return Jc, Js, Tc, Ts, Kc, Ks

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


class KoopmanHillDAE(KoopmanHillProjection):
    def __init__(self, fourier, tol=0, autonomous=False, tol_drazin=1e-6):
        super().__init__(fourier, tol, autonomous)
        self.tol_drazin = tol_drazin

    @override
    def fundamental_matrix(self, t_over_period, hbm: HBMEquationDAE, omega=None):
        C = self.C_time(t_over_period)
        hill_matrix = hbm.hill_matrix()

        if omega is None:
            omega = hbm.omega
        else:
            pass
        t = t_over_period * 2 * np.pi / hbm.omega

        if hbm.ode.invertible:
            hill_matrix_inv = np.linalg.solve(hbm.M(), hill_matrix)
            funda_mat = C @ expm(hill_matrix_inv * t) @ self.W
        else:
            funda_mat = (
                C
                @ generalized_exponential(hbm.M(), hill_matrix, t, self.tol_drazin)[0]
                @ self.W
            )

        return funda_mat

    @override
    def error_bound(self, t, a, b):
        raise NotImplementedError("Error bound not applicable for DAEs.")


class KoopmanHillDAESubharmonic(KoopmanHillSubharmonic):
    def __init__(self, fourier: Fourier, tol=0, autonomous=False):
        super().__init__(
            fourier=fourier.__replace__(real_formulation=False),
            tol=tol,
            autonomous=autonomous,
        )

    @override
    def fundamental_matrix(self, t_over_period, hbm, omega=None):
        if not hbm.ode.invertible:
            raise ValueError(
                "Subharmonic Koopman-Hill for DAEs with singular mass matrix is not implemented."
            )

        hill_matrix = hbm.hill_matrix(real_formulation=False)

        M = hbm.M()

        if hbm.fourier.real_formulation:
            M = hbm.fourier.T_to_cplx_from_real @ M @ hbm.fourier.T_to_real_from_cplx

        # ## Other way around:
        hill_matrix_inv = np.linalg.solve(M, hill_matrix)

        # hill_subh_inv = hill_matrix_inv[self.fourier.n_dof :, self.fourier.n_dof :]
        # hill_subh_inv = hill_subh_inv + 0.5j * hbm.omega * np.eye(
        #     hill_subh_inv.shape[0]
        # )

        M_subh = M[self.fourier.n_dof :, self.fourier.n_dof :]

        if omega is None:
            omega = hbm.omega
        else:
            pass

        hill_subh = hill_matrix[self.fourier.n_dof :, self.fourier.n_dof :]
        hill_subh = hill_subh + 0.5j * omega * M_subh
        hill_subh_inv = np.linalg.solve(M_subh, hill_subh)

        t = t_over_period * 2 * np.pi / omega

        C = self.C_time(t_over_period)
        C_subh = self.C_subh_time(t_over_period=t_over_period)

        funda_mat = C @ expm(hill_matrix_inv * t) @ self.W

        funda_mat += C_subh @ expm(hill_subh_inv * t) @ self.W_subh

        if np.any(np.abs(np.imag(funda_mat)) > 1e-7):
            raise RuntimeError(
                "KoopmanHillDAESubharmonic: Significant imaginary part in fundamental matrix."
            )
        return np.real(funda_mat)


def drazin(A, tol=0, ax_plot=None):
    """Compute the Drazin inverse of a matrix A.

    Parameters
    ----------
    A : np.ndarray
        The input square matrix.
    tol : float, optional
        Tolerance for determining the rank (default is 0).

    Returns
    -------
    np.ndarray
        The Drazin inverse of the matrix A.
    """
    n = A.shape[0]

    T, Z, n_cutoff = schur(A, output="complex", sort=lambda x: abs(x) > tol)

    if ax_plot is not None:
        eigenvalues = np.diag(T)
        eigenvalues_plot = np.zeros_like(eigenvalues)
        for k, eigenvalue in enumerate(eigenvalues):
            eigenvalues_plot[k] = round_to_significant_digits(eigenvalue, 2)

        _, idx_unique = np.unique(eigenvalues_plot, return_index=True)
        ax_plot.semilogy(
            n * np.ones_like(eigenvalues[idx_unique]),
            np.abs(eigenvalues[idx_unique]),
            "x",
        )

    R = T[:n_cutoff, :n_cutoff]
    N = T[n_cutoff:, n_cutoff:]
    C = T[:n_cutoff, n_cutoff:]

    W = np.eye(n, dtype=complex)

    if np.linalg.norm(C, np.inf) > tol:
        # warnings.warn(
        #     "Drazin inverse computation: Non-zero coupling block detected. Results may be inaccurate."
        # )

        W_nz = solve_sylvester(R, -N, -C)
        W[:n_cutoff, n_cutoff:] = W_nz

    # if np.max(np.abs(np.linalg.eig(N)[0])) > tol:
    #     warnings.warn(
    #         "Drazin inverse computation: Non-nilpotent block detected. Results may be inaccurate."
    #     )

    # if np.linalg.norm(Z @ Z.T.conj() - np.eye(n), np.inf) > tol:
    #     warnings.warn(
    #         "Drazin inverse computation: Schur vectors are not unitary. Results may be inaccurate."
    #     )

    drazin_schur = np.zeros_like(T)
    drazin_schur[:n_cutoff, :n_cutoff] = solve_triangular(
        R, np.eye(n_cutoff), lower=False
    )

    return Z @ W @ drazin_schur @ solve_triangular(W, Z.T.conj()), n_cutoff / n


def generalized_exponential(M, hill_matrix, t, tol_drazin=1e-6, tol_cond=1e6):
    """Compute the generalized matrix exponential for DAEs based on the Drazin inverse.
    This yields the fundamental solution matrix for the LTI DAE

    M*z_dot = hill_matrix * z

    Parameters
    ----------
    hill_matrix : np.ndarray
        The Hill matrix of the DAE system.
    t : float
        The time at which to evaluate the exponential.
    """

    a_vals = [1.0, 10.0, 0.1, 100, 0.01, 1000, 0.001]
    success = False
    for a in a_vals:
        pencil = a * M - hill_matrix
        if np.linalg.cond(pencil) < tol_cond:
            success = True
            break
    if not success:
        raise RuntimeError(
            f"Could not find suitable scaling factor 'a' for Drazin inverse with condition < {tol_cond}."
        )

    pencil_lu = lu_factor(a * M - hill_matrix)
    pencil_H = lu_solve(pencil_lu, hill_matrix)
    pencil_M = lu_solve(pencil_lu, M)
    pencil_drazin, ratio = drazin(pencil_M, tol_drazin)

    P_0 = pencil_drazin @ pencil_M
    exp = expm((pencil_drazin @ pencil_H) * t)

    return exp @ P_0, P_0, a
