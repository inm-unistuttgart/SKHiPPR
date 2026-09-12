"""Algebraic equation system. Encodes a system of algebraic equations. Base class."""

from abc import ABC, abstractmethod

from typing import override
from collections.abc import Callable
import numpy as np
from copy import copy
import warnings

from skhippr.stability.AbstractStabilityMethod import StabilityEquilibrium
from skhippr.equations.AbstractEquation import AbstractEquation


# still an abstract class
class AbstractODE(AbstractEquation):
    """Abstract base class for first-order differential equations. The equilibrium problem can immediately solved by passing the ODE into the :py:class:`~skhippr.solvers.newton.NewtonSolver`. If no stability method is provided during instantiation, the default :py:class:`~skhippr.stability.StabilityEquilibrium` is used for the equilibrium problem.

    Attributes:
    -----------

    autonomous : bool
        Whether the ODE is autonomous (does not depend on time).
    n_dof : int
        Number of degrees of freedom of the ODE.
    x : np.ndarray
        State vector.
    t : float
        Time variable.
    """

    def __init__(self, autonomous: bool, n_dof: int, stability_method=None):
        """The constructor must set the number of degrees of freedom as well as all required parameter values (including the initial state) as properties."""
        if stability_method is None:
            stability_method = StabilityEquilibrium(n_dof)
        super().__init__(stability_method=stability_method)
        self.autonomous = autonomous
        self.n_dof = n_dof
        self.has_nontrivial_omega_derivative = False

    @abstractmethod
    def dynamics(self, t=None, x=None) -> np.ndarray:
        """Return the right-hand side of the first-order ode x_dot = f(t, x) as a 1-D numpy array of size ``self.n_dof``. This method can be passed into ``scipy.integrate.solve_ivp``.

        Subclass implementations are expected to check the correct dimensions of ``x`` and ``t`` using :py:func:`~skhippr.odes.AbstractODE.AbstractODE.check_dimensions`.

        Parameters
        ----------
        t : float | np.ndarray, optional
            Time variable. If ``None``, ``self.t`` is used.
        x : np.ndarray, optional
            State vector. If ``None``, ``self.x`` is used.
        """
        if x is None:
            x = self.x
        if t is None:
            t = self.t
        self.check_dimensions(t, x)
        f = ...
        return f

    def check_dimensions(self, t=None, x=None):
        """Check the dimensions of the time variable and state vector.
        If ``t`` is a scalar, ``x`` must be 1-D or have exactly one column. If ``t`` is a vector, ``x`` must have the same number of columns as ``t``.

        Raises
        ------

        ValueError
            If the dimensions of ``t`` and ``x`` are incompatible or if ``x`` does not have the expected number of rows.
        """
        if x is not None and x.shape[0] != self.n_dof:
            raise ValueError(f"{x} must have {self.n_dof} rows but has {x.shape[0]}")
        if t is not None and np.isscalar(t) and len(x.shape) > 1 and x.shape[1] > 1:
            raise ValueError(
                f"{x} must be 1-D or have exactly one column if t is scalar"
            )
        if t is not None and np.squeeze(t).size > 1 and x.shape[1] != len(t):
            raise ValueError(
                f"t and x have incompatible sizes: {t.shape} vs. {x.shape}"
            )

    @override
    def residual_function(self) -> np.ndarray:
        """Evaluates ``self.dynamics()`` at ``self.t`` and ``self.x``. The residual is zero at an equilibrium."""
        return self.dynamics()

    @override
    def stability_criterion(self, eigenvalues) -> bool:
        """Checks if all eigenvalues have a non-positive real part."""
        return np.all(np.real(eigenvalues) < self.stability_method.tol)

    @override
    def derivative(
        self, variable, update=False, h_fd=1e-4, t=None, x=None
    ) -> np.ndarray:
        """Provides an interface for optional arguments ``t`` and ``x`` into the derivative method."""
        if t is not None or x is not None:
            update = True

        ########## DEBUGGING always use finite differences
        # if True:  # variable in ("X", "omega"):
        #     warnings.warn("Override closed form derivative in FirstOrderODE")
        #     print(f"Override closed form derivative w.rt. {variable} in FirstOrderODE")
        #     if t is not None:
        #         t_old = self.t
        #         self.t = t

        #     if x is not None:
        #         x_old = self.x
        #         self.x = x

        #     derivative = self.finite_difference_derivative(variable, h_step=h_fd)

        #     if t is not None:
        #         self.t = t_old

        #     if x is not None:
        #         self.x = x_old

        #     # Check sizes
        #     cols_expected = np.atleast_1d(getattr(self, variable)).shape[0]
        #     rows_expected = self.residual(update=False).shape[0]
        #     others_expected = self.residual(update=False).shape[1:]
        #     if derivative.shape != (rows_expected, cols_expected, *others_expected):
        #         raise ValueError(
        #             f"Size mismatch in derivative w.r.t. '{variable}': Expected {(rows_expected, cols_expected, *others_expected)}, got {derivative.shape[:2]}"
        #         )

        #     self._derivative_dict[variable] = derivative

        #     return derivative
        ##########

        if not update:
            return super().derivative(variable, update, h_fd)
        # provide an interface which offers t and x
        try:
            derivative = self.closed_form_derivative(variable, t, x)
        except NotImplementedError:
            t_old = self.t
            x_old = self.x

            if t is not None:
                self.t = t
            else:
                t = self.t
            if x is not None:
                self.x = x
            else:
                x = self.x

            derivative = super().derivative(variable, update, h_fd)
            self.t = t_old
            self.x = x_old

        self._derivative_dict[variable] = derivative
        return derivative

    def closed_form_derivative(self, variable, t=None, x=None) -> np.ndarray:
        """Requires subclasses to implement a closed-form derivative with optional arguments ``t`` and ``x``."""

        return super().closed_form_derivative(variable)

    def nontrivial_omega_derivative(self, t=None, x=None) -> np.ndarray:
        """If the ODE has a nontrivial derivative with respect to omega, this method can be implemented to provide it. This is relevant for the stability analysis of periodic orbits in forced systems."""

        if not self.has_nontrivial_omega_derivative:
            return 0

        else:
            raise NotImplementedError(
                "Default to finite differences for omega derivative."
            )


class AbstractDAE(AbstractODE):
    """Abstract base class for differential-algebraic equations (DAEs). DAEs consist of differential equations coupled with algebraic constraint equations. The DAE is formulated in the form

        M*x_dot = f(t, x)

    with a non-invertible matrix M.
    The equilibrium problem can immediately solved by passing the DAE into the :py:class:`~skhippr.solvers.newton.NewtonSolver`.

    Attributes:
    -----------

    autonomous : bool
        Whether the DAE is autonomous (does not depend on time).
    n_dof : int
        Number of degrees of freedom of the DAE.
    n_constraints : int
        Number of algebraic constraints in the DAE.
    x : np.ndarray
        State vector.
    t : float
        Time variable.
    """

    def __init__(
        self,
        n_dof,
        autonomous: bool,
        stability_method=None,
        M_is_constant=False,
        invertible: bool = False,
    ):

        super().__init__(autonomous, n_dof, stability_method)
        self.M_is_constant = M_is_constant
        self.invertible = invertible

    @abstractmethod
    def M_small(self, t=None, x=None):
        if t is None:
            t = self.t
        if x is None:
            x = self.x

        return np.eye(self.n_dof)


class SecondOrderODE(AbstractODE):
    """A second-order ODE is represented as a first-order ODE with twice the number of degrees of freedom."""

    def __init__(
        self,
        t: float,
        q: np.ndarray,
        dq: np.ndarray,
        M: np.ndarray,
        D: np.ndarray,
        K: np.ndarray,
        autonomous=False,
        stability_method=None,
    ):
        n_dof = 2 * M.shape[0]
        super().__init__(autonomous, n_dof, stability_method)

        self.t = t
        self.x = np.concatenate((np.atleast_1d(q), np.atleast_1d(dq)), axis=0)
        self.M = M
        self.D = D
        self.K = K

    @abstractmethod
    def f_nonlin(self, t=None, q=None, dq=None) -> np.ndarray:
        """Everything that is not represented in M, D, K. CAUTION: Includes forcing!"""

    def derivative_f_nonlin(self, variable, t, q=None, dq=None) -> np.ndarray:
        """Derivative of the non-linear part with respect to q and dq."""
        raise NotImplementedError(
            "Derivative of the non-linear part not implemented to default to FD evaluation. To be overridden in subclass."
        )

        match variable:
            case "q":
                ...
            case "dq":
                ...
            case _:
                raise NotImplementedError

    @override
    def dynamics(self, t=None, x=None) -> np.ndarray:
        """Returns the first-order dynamics for a second-order ODE."""
        if x is None:
            x = self.x
        if t is None:
            t = self.t
        self.check_dimensions(t, x)

        q, dq = np.split(x, 2, axis=0)
        f = np.zeros_like(x)
        f[: q.shape[0], ...] = dq

        rhs = self.D @ dq + self.K @ q + self.f_nonlin(t, q, dq)
        f[q.shape[0] :, ...] = np.linalg.solve(self.M, -rhs)

        return f

    @override
    def closed_form_derivative(self, variable, t=None, x=None) -> np.ndarray:
        """Closed-form derivative for the second-order ODE."""
        if t is None:
            t = self.t
        if x is None:
            x = self.x
        self.check_dimensions(t, x)

        q, dq = np.split(x, 2, axis=0)

        match variable:
            case "x":
                n_dof = q.shape[0]
                df_dx = np.zeros((x.shape[0], *x.shape))
                for k in range(n_dof):
                    df_dx[k, k + n_dof, ...] = 1
                    df_dx[n_dof:, :n_dof, ...] = -np.linalg.solve(
                        self.M,
                        self.K + self.derivative_f_nonlin("q", t, q, dq),
                    )
                    df_dx[n_dof:, n_dof:, ...] = -np.linalg.solve(
                        self.M,
                        self.D + self.derivative_f_nonlin("dq", t, q, dq),
                    )
                return df_dx
            case _:
                dfnl_dvar = self.derivative_f_nonlin(variable, t, q, dq)
                return np.vstack([np.zeros_like(dfnl_dvar), dfnl_dvar])
