from typing import Any
import numpy as np
from fractions import Fraction
from math import lcm
from copy import replace
from scipy import fft

from skhippr.cycles.hbm import HBMEquation
from skhippr.stability.AbstractStabilityHBM import AbstractStabilityHBM
from skhippr.Fourier import Fourier


class SinglePassRK(AbstractStabilityHBM):
    """
    SinglePassRK implements single-pass fixed-step explicit Runge-Kutta methods for stability analysis in the context of Harmonic Balance Methods (HBM).

    Parameters:
    -----------
    fourier : Fourier
        Fourier object containing discretization and transformation information.
    A : np.ndarray
        Butcher tableau A (stage weights). Must be strictly lower triangular, i.e., explicit Runge Kutta method.
    b : np.ndarray
        Butcher tableau b (quadrature weights).
    c : np.ndarray
        Butcher tableau c (normalized evaluation points), must be an array of Fractions with a low common denominator.
    stepsize : float, optional
        Desired step size for the integration. If not provided, it is chosen such that no additional FFTs are required.
    tol : float, optional
        Tolerance for the stability computation (default is 0).

    """

    def __init__(
        self,
        fourier: Fourier,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        label: str,
        stepsize: float = None,
        tol: float = 0,
    ):
        """c must be provided as an array of Fractions."""
        if not c.dtype == Fraction:
            c = np.array([Fraction(c[k]).limit_denominator() for k in range(len(c))])
        if any(c > 1) or any(c < 0):
            raise ValueError("Butcher tableau c values must lie between 0 and 1")

        self.samples_per_step = lcm(*[frac.denominator for frac in c])
        # Count the number of evaluations over a period needed to (approximately) fulfill stepsize
        if stepsize is not None:
            self.steps_per_period: int = fft.next_fast_len(int(2 * np.pi / stepsize))
            # L = fft.next_fast_len(self.samples_per_step * self.steps_per_period)
            L = self.samples_per_step * self.steps_per_period
            if L != fourier.L_DFT:
                fourier = fourier.__replace__(L_DFT=L)

        else:
            self.steps_per_period = fourier.L_DFT / self.samples_per_step  # type:ignore
            if not self.steps_per_period.is_integer():
                raise ValueError(
                    f"{fourier} clashes with single pass method {label}, which needs L_DFT divisible by {self.samples_per_step}"
                )
            else:
                self.steps_per_period = int(self.steps_per_period)

        super().__init__(f"single pass {label}", fourier, tol)

        # Parse Butcher Tableau
        self.A = A
        self.b = b
        self.c = (np.array(c) * self.samples_per_step).astype(int, casting="unsafe")

        self.h0 = (
            self.samples_per_step / fourier.L_DFT
        )  # normalized step size: one period is at tau=1

    def fundamental_matrix(
        self, t_over_period: float, hbm: HBMEquation | np.ndarray, omega=None
    ) -> np.ndarray:
        """
        Computes the fundamental matrix for a given normalized time.

        This method fixed-step-integrates the variational equation up to ``t``.

        Parameters
        ----------

        t_over_period : float
            The time, expressed as a multiple of the period over which to integrate.
        problem : HBMProblem
            The problem instance containing system parameters, Fourier information, and Jacobian samples.

        Returns
        -------

        np.ndarray
            The fundamental matrix (state transition matrix) after integrating over the specified time.

        """

        if isinstance(hbm, np.ndarray):
            J_samples = self.fourier.matrix_inv_DFT(hbm)
        else:
            J_samples = hbm.ode_samples(self.fourier)
            omega = hbm.omega
        T = 2 * np.pi / omega
        dt = self.h0 * T
        A = self.A * dt
        b = self.b * dt

        desired_full_periods = np.floor(t_over_period)
        period = 0
        t_end = np.mod(t_over_period, 1) * T

        Phi = np.eye(self.fourier.n_dof, dtype=self.fourier.iDFT_small.dtype)
        increments: list[Any] = [None] * len(self.c)
        finished = False

        while not finished:
            idx_n = 0
            for t in np.linspace(
                start=0, stop=T, num=self.steps_per_period, endpoint=False
            ):
                if period == desired_full_periods and t >= t_end:
                    finished = True
                    break

                for k in range(len(self.c)):
                    increments[k] = J_samples[
                        :, :, np.mod(idx_n + self.c[k], self.fourier.L_DFT)
                    ] @ (Phi + sum([A[k - 1, s] * increments[s] for s in range(k)]))
                Phi += sum([b[k] * increments[k] for k in range(len(self.c))])

                t += dt
                idx_n += self.samples_per_step

            period += 1
        return Phi


class SinglePassRK4(SinglePassRK):
    """
    Explicit Single-Pass method using the RK4 tableau.

    This method only needs one intermediate sample between each integration step, allowing for a relatively low ``L_DFT`` at small time step.

    References
    ----------

    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Examples (accessed on 06/17/2025)
    """

    def __init__(self, fourier: Fourier, stepsize: float = None, tol: float = 0):
        A = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
        b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
        c = np.array([0, 0.5, 0.5, 1])
        super().__init__(fourier, A, b, c, "RK4", stepsize, tol)


class SinglePassRK38(SinglePassRK):
    """
    Explicit Single-Pass method using the 3/8-rule.

    This method needs two intermediate samples per time step.

    References
    ----------

    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Examples (accessed on 06/17/2025)
    """

    def __init__(self, fourier: Fourier, stepsize: float = None, tol: float = 0):
        A = np.array([[1 / 3, 0, 0], [-1 / 3, 1, 0], [1, -1, 1]])
        b = np.array([1 / 8, 3 / 8, 3 / 8, 1 / 8])
        c = np.array([0, 1 / 3, 2 / 3, 1])
        super().__init__(fourier, A, b, c, "RK3/8", stepsize, tol)
