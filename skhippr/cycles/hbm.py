from typing import override, TYPE_CHECKING
import warnings
import numpy as np

from skhippr.equations.AbstractEquation import AbstractEquation
from skhippr.cycles.AbstractCycleEquation import AbstractCycleEquation
from skhippr.odes.AbstractODE import AbstractODE
from skhippr.Fourier import Fourier
from skhippr.equations.EquationSystem import EquationSystem

# Imports only needed for type hinting
if TYPE_CHECKING:
    from skhippr.stability.AbstractStabilityHBM import AbstractStabilityHBM


class HBMEquation(AbstractCycleEquation):
    """
    :py:class:`~ skhippr.cycles.hbm.HBMEquation` is a subclass of :py:class:`~ skhippr.equations.AbstractEquation.AbstractEquation` which encodes the harmonic balance equations to find a periodic solution of a non-autonomous :py:class:`~skhippr.odes.AbstractODE.AbstractODE` with periodic excitation.

    This class computes a nonlinear harmonic balance residual in the frequency domain using a :py:class:`~skhippr.Fourier.Fourier` object. The resulting nonlinear harmonic balance equations in the frequency domain are then solved using the methods of the parent :py:class:`~skhippr.cycles.newton.NewtonProblem` class.

    As the :py:class:`~ skhippr.cycles.hbm.HBMEquation` is a subclass of :py:class:`~skhippr.cycles.AbstractCycleEquation.AbstractCycleEquation`, attribute calls and updates are delegated to the underlying :py:class:`~skhippr.odes.AbstractODE.AbstractODE` whenever applicable.

    Attributes:

    * :py:attr:`~skhippr.hbm.HBMEquation.ode`: :py:class:`~skhippr.odes.AbstractODE.AbstractODE` object representing the non-autonomous ODE of which a periodic solution is sought.
    * :py:attr:`~skhippr.hbm.HBMEquation.omega`: Frequency of the system forcing.
    * :py:attr:`~skhippr.hbm.HBMEquation.fourier: :py:class:`~skhippr.Fourier.Fourier` object used to perform the FFT.
    * :py:attr:`~skhippr.hbm.HBMEquation.X`: Vector of Fourier coefficients of the periodic solution.
    * :py:attr:`~skhippr.hbm.HBMEquation.period_k`: The period time of the sought-after periodic solution is ``period_k`` times the forcing period given by ``omega``.
    * :py:attr:`~skhippr.hbm.HBMEquation.stability_method`: The stability method.
    """

    def __init__(
        self,
        ode: AbstractODE,
        omega: float,
        fourier: Fourier,
        initial_guess: np.ndarray = None,
        period_k: float = 1,
        stability_method: "AbstractStabilityHBM" = None,
    ):
        """
        Initialize the HBM equations.
        """
        super().__init__(
            ode=ode, omega=omega, period_k=period_k, stability_method=stability_method
        )
        self.fourier = fourier

        # DFT initial guess if required
        if initial_guess is None:
            initial_guess = self.fourier.DFT(ode.x)

        self.X = initial_guess

    def x_time(self) -> np.ndarray:
        """
        Return the time series of the solution at the FFT sample points as a ``self.ode.n_dof`` x ``self.fourier.L_DFT`` array.
        """

        return self.fourier.inv_DFT(self.X)

    def aft(self, X=None) -> np.ndarray:
        """
        Compute the HBM residual and its derivatives using the AFT (Alternating Frequency/Time) method.

        Parameters
        ----------
        X : np.ndarray, optional
            State vector in the frequency domain. If None, uses self.X.

        Returns
        -------
        R : np.ndarray
            Residual vector in the frequency domain.
        """

        if X is None:
            X = self.X

        x_samp = self.fourier.inv_DFT(X)
        ts = self.fourier.time_samples(self.omega_solution)

        """ Determine the ODE at the time samples."""
        # vectorized formulation
        try:
            fs_samp = self.ode.dynamics(t=ts, x=x_samp)
        except:
            # vectorization impossible, iterate through time samples
            fs_samp = np.zeros_like(x_samp)
            for k in range(x_samp.shape[1]):
                fs_samp[:, k, ...] = self.ode.dynamics(ts[k], x_samp[:, k, ...])

        """Construct residual"""
        R = self.fourier.DFT(fs_samp) - self.fourier.derivative_coeffs(
            X, self.omega_solution
        )

        return R

    @override
    def residual_function(self):
        """Returns ``self.aft(self.X)`` as the residual function of the HBM equations."""
        return self.aft(self.X)

    def closed_form_derivative(self, variable):
        """Return the closed-form derivative of the residual function w.r.t. the variable. by Fourier transforming the corresponding derivatives in the time domain,evaluated at the samples."""
        if variable == "X":
            return self.dR_dX(self.X)
        elif variable == "omega":
            return self.dR_domega(self.X)
        else:
            return self.dR_dvar(variable, self.X)

    def dR_dX(self, X=None):

        # Sample the derivatives of the ode
        if X is None:
            X = self.X

        x_samp = self.fourier.inv_DFT(X)
        ts = self.fourier.time_samples(self.omega_solution)

        try:
            Js = self.ode.closed_form_derivative(variable="x", t=ts, x=x_samp)
        except NotImplementedError:
            # use finite differences
            self.ode.t = ts
            self.ode.x = x_samp
            Js = self.ode.derivative(variable="x")
        except:
            # Vectorization not working, determine sample by sample
            Js = np.zeros((x_samp.shape[0], *x_samp.shape))
            for k, t in enumerate(ts):
                Js[:, :, k, ...] = self.ode.closed_form_derivative(
                    "x", t, np.squeeze(x_samp[:, k])
                )

        derivative = self.fourier.matrix_DFT(Js)
        derivative -= self.omega_solution * self.fourier.derivative_matrix

        return derivative

    def dR_domega(self, X=None):
        if X is None:
            X = self.X
        dR_dom = (
            -self.fourier.factors_derivative
            / self.period_k
            * X[self.fourier.idx_derivative]
        )
        return dR_dom[:, np.newaxis]

    def dR_dvar(self, variable, X=None):

        # Sample the derivatives of the ode
        if X is None:
            X = self.X

        x_samp = self.fourier.inv_DFT(X)
        ts = self.fourier.time_samples(self.omega_solution)

        try:
            derivatives_time = self.ode.closed_form_derivative(
                variable=variable, t=ts, x=x_samp
            )
        except NotImplementedError:
            # use finite differences
            self.ode.t = ts
            self.ode.x = x_samp
            derivatives_time = self.ode.derivative(variable=variable, update=True)
        except:
            # Vectorization not working, determine sample by sample
            derivatives_time = np.zeros_like(x_samp)
            for k, t in enumerate(ts):
                derivatives_time[:, k, ...] = self.ode.closed_form_derivative(
                    variable, t, np.squeeze(x_samp[:, k])
                )

        return self.fourier.DFT(derivatives_time)

    def hill_matrix(self, real_formulation: bool = None) -> np.ndarray:
        """Return the Hill matrix, which is the derivative of the HBM equations w.r.t. ``X``.

        Parameters
        ----------
        real_formulation : bool, optional
            If True, returns the Hill matrix in real formulation, otherwise in complex formulation.
            If None, uses the value of ``self.fourier.real_formulation``.

        """

        H = self.derivative("X", update=False)

        # Transform between real and complex formulation
        if real_formulation is not None:

            if self.real_formulation and not real_formulation:
                # return complex-valued Hill matrix
                H = self.T_to_cplx_from_real @ H @ self.T_to_real_from_cplx

            elif real_formulation and not self.real_formulation:
                # return real-valued Hill matrix
                H = self.T_to_real_from_cplx @ H @ self.T_to_cplx_from_real

        return H

    def ode_coeffs(self) -> np.ndarray:
        """
        Extract and return the Fourier coefficients of the Jacobian matrix df/dx(x(t)) from the Hill matrix.

        Note
        ----

        * In the real formulation, coefficients are computed according to Bayer et al. 2024, Eq.s (82)-(83) (see references), with the first N cos / sin coefficients appearing in the first row block.
        * Otherwise, the most centered coefficients are taken from the central row block .

        Returns
        -------
        np.ndarray
            A 3D array of shape (n_dof, n_dof, 2*N_HBM+1) containing the Fourier coefficients of df/dx(x(t))

        References
        ----------

        Bayer et al. 2024, https://doi.org/10.1016/j.ijnonlinmec.2024.104894
        """

        Hill_mat = self.hill_matrix()

        if self.fourier.real_formulation:
            # Cf. Bayer et al. 2024, Eq.s (82) - (83)
            # First N coefficients are in first row block
            J_coeffs = 2 * Hill_mat[: self.fourier.n_dof, :]
            J_coeffs[:, : self.fourier.n_dof] *= 0.5

        else:
            # Most centered coeffs are in central row block (in reversed order)
            # Order is flipped after the reshaping.
            J_coeffs = Hill_mat[
                (self.fourier.N_HBM * self.fourier.n_dof) : (
                    (self.fourier.N_HBM + 1) * self.fourier.n_dof
                ),
                :,
            ]

        J_coeffs = np.reshape(
            J_coeffs, shape=(self.fourier.n_dof, self.fourier.n_dof, -1), order="F"
        )
        if not self.fourier.real_formulation:
            J_coeffs = np.flip(J_coeffs, axis=2)

        return J_coeffs

    def ode_samples(self, fourier=None) -> np.ndarray:
        """
        Generate samples of the Jacobian matrix J(t) in time from the Hill matrix.

        Parameters
        ----------

        fourier : Optional
            An object providing the `matrix_inv_DFT` method. If None, uses `self.fourier`.

        Returns
        -------

        np.ndarray
            The df/dx(x(t)) at the FFT samples.
        """

        if fourier is None:
            fourier = self.fourier
        return fourier.matrix_inv_DFT(self.hill_matrix())

    @override
    def stability_criterion(self, eigenvalues) -> bool:
        """
        The stability criterion for the HBM equations is that all Floquet multipliers must have a modulus less than 1.
        If the ode is autonomous, the Floquet multiplier corresponding to the phase freedom is removed from the stability criterion.
        """

        if self.stability_method is None:
            raise ValueError("No stability method available!")
        else:
            floquet_multipliers = eigenvalues
            if self.ode.autonomous:
                idx_freedom_of_phase = np.argmin(abs(floquet_multipliers - 1))
                floquet_multipliers = np.delete(
                    floquet_multipliers, idx_freedom_of_phase
                )

        return np.all(np.abs(floquet_multipliers) < 1 + self.stability_method.tol)

    def exponential_decay_parameters(self, threshold=5e-15):
        """
        Estimate the exponential decay of the Fourier coefficients of the Jacobian matrix J(t).

        This method computes parameters ``a`` and ``b`` such that the norm of the ``k``-th Fourier coefficient
        of df/dx(x(t)) is bounded by ``max(threshold, a * exp(-b * |k|))``.

        Note
        ----

        These decay parameters play a central role in the error bound of Bayer and Leine, 2025 (https://arxiv.org/pdf/2503.21318).

        Parameters
        ----------

        threshold : float, optional
            The minimum threshold for considering the norm of the coefficients.
            Defaults to 5e-15.

        Returns
        -------

        np.ndarray
            an array of shape (``m``, 2) where each row contains
            a pair ``[a, b]`` for an exponential envelope which is exact at two Fourier coefficients and upperbounds the others.

        Warns:
            UserWarning: If the norm of the coefficients does not decay below the threshold within the
                available harmonic range (``N_HBM``), a warning is issued suggesting to increase ``N_HBM``.
        """

        Js = self.ode_coeffs()
        norms = []
        ks = []
        for k in range(self.fourier.N_HBM + 1):
            # Determine J_k (complex-valued)
            if self.fourier.real_formulation:
                if k > 0:
                    J_cplx = 0.5 * Js[:, :, k] + 0.5j * Js[:, :, self.fourier.N_HBM + k]
                else:
                    J_cplx = Js[:, :, 0]
            else:
                # consider the positive component
                J_cplx = Js[:, :, self.fourier.N_HBM + k]

            norm_J = np.linalg.norm(J_cplx, ord=2)
            if norm_J > threshold:
                norms.append(norm_J)
                ks.append(k)

        # Ensure that ||J_k|| decays below threshold within available N_HBM range
        if norm_J > threshold:
            warnings.warn(
                f"||J_k|| did not decay below threshold {threshold} within N_HBM={self.fourier.N_HBM}. Final norm is {norm_J}. Consider increasing N_HBM for accurate determination of exponential decay."
            )

        # Find enveloping lines: a' + b'*k >= log(norms)
        lines = find_linear_envelopes(ks, np.log(norms), 0.1 * threshold)

        if lines.size == 0:
            raise ValueError("No exponential decay could be fitted.")

        if len(lines) > 0:
            lines[:, 0] = np.exp(lines[:, 0])  # a = exp(a')
            lines[:, 1] = -lines[:, 1]  # b = -b'

        return lines

    def error_bound_fundamental_matrix(
        self, t: float | np.ndarray = None, _as=None, bs=None
    ):
        """

        Attempts to compute an error bound for the fundamental solution matrix at specified time(s) ``t``.
        For each time instant, the lowest bound produced by a combination of ``(a, b)`` is returned.

        Notes
        -----

        * If both ``_as`` and ``bs`` are provided, exponential decay parameters are not computed from the Hill matrix and the given parameters are used directly.
        * If only one of them is provided, all applicable exponential
        decay parameter combinations are computed and the ones closest to the provided values are
        used.
        * If neither is provided, all possible combinations of decay parameters are considered.
        At each time instant, the combination of ``(a, b)`` that produces the lowest bound is used.

        Parameters
        ----------

        t : float or np.ndarray, optional
            Time or array of times at which to evaluate the error bound. If None, uses the
            solution's period.
        ``_as``, ``bs``: array_like, optional
            (Target) parameters for the exponential decay.

        Returns
        -------

        E_bound : np.ndarray
            Array of error bounds evaluated at each time in ``t``.

        """
        if t is None:
            # monodromy matrix: evaluate after a period
            t = 2 * np.pi / (self.factor_k * self.omega)

        if _as is None or bs is None:
            params_decay = self.exponential_decay_parameters()

            if _as is not None:
                idxs = np.unique(
                    [np.argmin(np.abs(params_decay[:, 0] - a)) for a in _as]
                )
                params_decay = params_decay[idxs, :]
            elif bs is not None:
                idxs = np.unique(
                    [np.argmin(np.abs(params_decay[:, 1] - b)) for b in bs]
                )
                params_decay = params_decay[idxs, :]

            _as = params_decay[:, 0]
            bs = params_decay[:, 1]

        # At every time instant in t, use the combination of (a, b) that produces the lowest bound.
        E_bound = np.inf * np.ones_like(t)
        for k, a in enumerate(_as):
            E_bound_next = self.stability_method.error_bound(t, a, bs[k])
            E_bound = np.minimum(E_bound, E_bound_next)

        return E_bound


class HBMSystem(EquationSystem):
    """This subclass of :py:class:`~skhippr.equations.EquationSystem.EquationSystem` instantiates a :py:class:`~skhippr.cycles.hbm.HBMEquation` and considers it as the first equation. The Fourier coefficient vector ``X`` is the first unknown.

    If the underlying ODE is autonomous, the frequency ``omega`` of the periodic solution is not known in advance and is appended to the unknowns. Correspondingly, a :py:class:`~skhippr.cycles.hbm.HBMPhaseAnchor` equation is appended to the equations.
    """

    def __init__(
        self,
        ode,
        omega,
        fourier,
        initial_guess: np.ndarray = None,
        period_k: float = 1,
        stability_method: "AbstractStabilityHBM" = None,
        harmo_anchor: int = 1,
        dof_anchor: int = 0,
    ):
        hbm = HBMEquation(
            ode,
            omega,
            fourier,
            initial_guess,
            period_k,
            stability_method=stability_method,
        )

        equations = [hbm]
        unknowns = ["X"]

        if ode.autonomous:
            unknowns.append("omega")
            anchor_equation = HBMPhaseAnchor(
                fourier=hbm.fourier, X=hbm.X, harmo=harmo_anchor, dof=dof_anchor
            )
            equations.append(anchor_equation)

        super().__init__(
            equations=equations, unknowns=unknowns, equation_determining_stability=hbm
        )


class HBMPhaseAnchor(AbstractEquation):
    """This class implements an anchor equation for the harmonic balance method (HBM) in autonomous systems to ensure that the phase of a specified degree of freedom and harmonic of the periodic solution does not change during the HBM solution procedure.

    * Complex formulation:
        exp(i*phi) = X+/X- = const
    * Real formulation:
        -tan(phi) = c_k/s_k = const

    Hereby, ``harmo`` and ``dof``   specify the harmonic and degree of freedom for which the phase is anchored.

    """

    def __init__(self, fourier, X, harmo, dof):
        super().__init__(None)
        self.X = X
        self.idx_anchor = self._determine_anchor(fourier, harmo, dof)
        self.anchor = np.zeros((1, X.size), dtype=X.dtype)
        self.anchor[0, self.idx_anchor[0]] = -1
        # self.phase_required = self.X[self.idx_anchor[0]] / self.X[self.idx_anchor[1]]

    def residual_function(self):
        """Always returns zero."""
        # anchor equation (phase may not change):
        # delta X[anchor[0]] = X[anchor[0]]/X[anchor[1]] * delta X[anchor[1]]

        # phase = self.X[self.idx_anchor[0]] / self.X[self.idx_anchor[1]]
        # return phase - self.phase_required
        return np.atleast_1d(0)

    def closed_form_derivative(self, variable):
        """Return the anchor as derivative w.r.t ``X`` and zero otherwise."""
        if variable == "X":
            self.anchor[0, self.idx_anchor[1]] = (
                self.X[self.idx_anchor[0]] / self.X[self.idx_anchor[1]]
            )
            return self.anchor
        else:
            return np.atleast_2d(0)

    def _determine_anchor(self, fourier, harmo: int = 1, dof: int = 0) -> np.ndarray:
        """Determine the index of the anchor equation.
        The anchor equation ensures that the phase of the  harmo-th harmonic
        and the dof-th degree of freedom does not change during HBM solution for autonomous systems.
        """
        if fourier.real_formulation:
            # -tan(phi) = c_k/s_k = const -->  delta c = c_k/s_k * delta s
            idx_anchor = [
                harmo * fourier.n_dof + dof,
                (harmo + fourier.N_HBM) * fourier.n_dof + dof,
            ]
        else:
            # exp(i*phi) = X+/X- = const -->  delta X+ = X+/X- * delta X-
            idx_anchor = [
                (fourier.N_HBM + harmo) * fourier.n_dof + dof,
                (fourier.N_HBM - harmo) * fourier.n_dof + dof,
            ]

        # Avoid division by zero
        if abs(self.X[idx_anchor[1]]) < (1e-4 * abs(self.X[idx_anchor[0]])):
            idx_anchor.reverse()

        return np.array(idx_anchor)


def find_linear_envelopes(x_vals, y_vals, tolerance: float = 0) -> np.array:
    """
    Find all linear upper envelope segments for a set of points.

    Given arrays of x and y values, this function computes all pairs (a, b) such that the line
    y(x) = a + b*x passes through two of the given points and lies above (or on) all other points,
    within a specified tolerance.

    Parameters
    ----------
    x_vals : array-like
        1D array or list of x-coordinates of the points.
    y_vals : array-like
        1D array or list of y-coordinates of the points.
    tolerance : float, optional
        Allowed tolerance for a point to be considered below the line. Default is 0.

    Returns
    -------
    np.ndarray
        A numpy array with ``a`` in the first column and ``b`` in the second column,
        where each pair defines an upperbounding line y(x) = a + b*x.
    """

    results = []
    y_vals = np.atleast_1d(y_vals)
    x_vals = np.atleast_1d(x_vals)

    #  Ensure correct inputs - y_vals and x_vals must be of same length >= 2
    if x_vals.shape != y_vals.shape or x_vals.shape[0] < 2:
        raise ValueError(
            f"x_vals (shape: {x_vals.shape}) and y_vals (shape: {y_vals.shape}) are expected to be equal and contain at least 2 data pairs"
        )

    for l, (x_1, y_1) in enumerate(zip(x_vals, y_vals)):
        for x_2, y_2 in zip(x_vals[l + 1 :], y_vals[l + 1 :]):

            # Determine line through (x_1, y_1) and (x_2, y_2)
            b = (y_2 - y_1) / (x_2 - x_1)
            a = y_1 - b * x_1

            y_bound = a + b * x_vals
            if np.any(y_bound + tolerance < y_vals):
                continue
            results.append([a, b])

    return np.atleast_2d(results)
