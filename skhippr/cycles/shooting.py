"""Shooting method for finding periodic solutions."""

from typing import override, Any
from collections.abc import Callable
import numpy as np
from scipy.integrate import solve_ivp

from skhippr.equations.AbstractEquation import AbstractEquation
from skhippr.odes.AbstractODE import AbstractODE
from skhippr.cycles.AbstractCycleEquation import AbstractCycleEquation
from skhippr.stability.AbstractStabilityMethod import StabilityEquilibrium
from skhippr.equations.EquationSystem import EquationSystem


class ShootingBVP(AbstractCycleEquation):
    """
    ShootingBVP implements the shooting method to solve the boundary value problem for finding periodic solutions of nonautonomous ODEs.

    The residual is determined by integrating the initial value problem over a period from ``self.ode.t`` to ``self.ode.t+self.T`` using ``scipy.integrate.solve_ivp`` and comparing the state at start and end time.
    """

    def __init__(
        self,
        ode: AbstractODE,
        T: float,
        period_k: int = 1,
        **kwargs_odesolver,
    ):
        super().__init__(
            ode=ode,
            omega=2 * np.pi / T,
            period_k=period_k,
            stability_method=StabilityEquilibrium(ode.n_dof),
        )
        self.t_0 = ode.t
        self.kwargs_odesolver = kwargs_odesolver

    @override
    def residual_function(self) -> np.ndarray:
        """
        Computes the residual function ``r = x(T) - x(0)`` of the shooting problem.
        """
        x_T = self.x_time(t_eval=self.t_0 + self.T_solution)
        return x_T[:, -1] - self.x

    @override
    def closed_form_derivative(self, variable) -> np.ndarray:
        """
        Returns the closed-form derivative of the residual function with respect to the specified variable.

        * If the variable is "x", it returns ``Phi_T - np.eye(ode.n_dof)``, where ``Phi_T`` is the monodromy matrix.
        * If the variable is "T" and the system is autonomous, it returns the dynamics evaluated at the final time.
        * Otherwise, it raises a NotImplementedError to enforce the use of finite differences.

        """
        match variable:
            case "x":
                _, _, Phi_t = self.integrate_with_fundamental_matrix(
                    x_0=self.x, t=self.t_0 + self.T_solution
                )
                return Phi_t[:, :, -1] - np.eye(Phi_t.shape[0])
            case "T":
                if self.ode.autonomous:
                    return self.ode.dynamics(
                        t=self.t_0 + np.squeeze(self.T_solution),
                        x=self.residual(update=False) + self.x,
                    )[:, np.newaxis]
                else:
                    raise NotImplementedError(
                        "Period is parameter of ode; default to FD"
                    )
            case _:
                raise NotImplementedError(
                    "Finite differences more efficient for shooting problem"
                )

    @override
    def stability_criterion(self, eigenvalues):
        """Checks if all Floquet multipliers lie inside the unit circle.
        The Floquet multipliers are computed as the eigenvalues of the monodromy matrix ``Phi_T``, i.e., the eigenvalues of the derivative w.r.t. ```x`` plus one.
        """

        floquet_multipliers = eigenvalues + 1
        if self.ode.autonomous:
            idx_freedom_of_phase = np.argmin(abs(floquet_multipliers - 1))
            floquet_multipliers = np.delete(floquet_multipliers, idx_freedom_of_phase)

        return np.all(np.abs(floquet_multipliers) < 1 + self.stability_method.tol)

    def x_time(self, t_eval=None):
        """
        Computes the state trajectory over time by integrating the system dynamics using ``scipy.integrate.solve_ivp``.

        Parameters
        ----------

        t_eval : array_like or None, optional
            Time points at which to store the computed solution. If None, a default
            linspace from 0 to ``self.T`` with 150 points is used.

        Returns
        -------

        np.ndarray
            (``n_dof``, ``L``) 2-D array containing the state trajectory evaluated at the specified time points.
        """

        if t_eval is None:
            t_eval = np.linspace(self.t_0, self.t_0 + self.T_solution, 150)

        if np.squeeze(t_eval).size == 1:
            t_eval = np.insert(np.squeeze(t_eval), 0, 0)

        sol = solve_ivp(
            fun=self.ode.dynamics,
            t_span=np.array((0, np.squeeze(self.T_solution))),
            y0=self.x,
            t_eval=t_eval,
            **self.kwargs_odesolver,
        )
        return sol.y

    def integrate_with_fundamental_matrix(
        self, x_0: np.ndarray = None, t: float = None
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Integrates the system dynamics along with its fundamental matrix.

        This method solves the initial value problem for the system's state and its associated fundamental matrix over a given time interval.

        Parameters
        ----------

        x0 : np.ndarray
            Initial state vector of the system.
        t : float
            Final time up to which the integration is performed.

        Returns
        -------

        sol.t : np.ndarray
            Array of time points at which the solution was evaluated.

        x : np.ndarray
            Array containing the state trajectory of the system at the time points. Shape is (``n_dof``, ``L``).

        fundamental_matrix : np.ndarray
            Array containing the fundamental matrix at each time point. Shape is (``n_dof``, ``n_dof``, ``L``).

        Notes
        -----
        The method integrates both the state and the fundamental matrix by augmenting the state vector with the flattened fundamental matrix and integrating both simultaneously.
        """

        if x_0 is None:
            x_0 = self.x

        if t is None:
            t = self.t_0 + self.T_solution

        t_span = np.insert(np.squeeze(t), 0, self.t_0)

        z_0 = np.hstack((x_0, np.eye(len(x_0)).flatten(order="F")))
        sol = solve_ivp(
            self._dynamics_x_phi_T,
            t_span,
            z_0,
            **self.kwargs_odesolver,
        )
        x = sol.y[: len(x_0), :]
        fundamental_matrix = np.reshape(
            sol.y[len(x_0) : (len(x_0) + 1) * len(x_0), :],
            (len(x_0), len(x_0), -1),
            order="F",
        )

        return sol.t, x, fundamental_matrix

    def _dynamics_x_phi_T(self, t: float, z: np.ndarray) -> np.ndarray:
        x = z[: self.ode.n_dof]
        Phi = np.reshape(
            z[self.ode.n_dof : (self.ode.n_dof + 1) * self.ode.n_dof],
            shape=(self.ode.n_dof, self.ode.n_dof),
            order="F",
        )

        dx_dt = self.ode.dynamics(t=t, x=x)
        df_dx = self.ode.derivative(variable="x", t=t, x=x)
        dPhi_dt = df_dx @ Phi

        dz_dt = np.hstack((dx_dt, dPhi_dt.ravel(order="F")))

        return dz_dt


class ShootingPhaseAnchor(AbstractEquation):
    """This class implements an anchor equation for the shooting method in autonomous systems. EachNewton update must be orthogonal to the flow at x_0."""

    def __init__(self, ode, x):
        super().__init__(None)
        self.ode = ode
        self.x = x

    def residual_function(self):
        """Always returns zero."""
        return np.atleast_1d(0)

    def closed_form_derivative(self, variable):
        """Return the anchor (flow at ``(self.t, self.x)``) as derivative w.r.t ``x`` and zero otherwise."""
        if variable == "x":
            return self.ode.dynamics(x=self.x)[np.newaxis, :]
        else:
            return np.atleast_2d(0)


class ShootingSystem(EquationSystem):
    """This subclass of :py:class:`~skhippr.equations.EquationSystem.EquationSystem` instantiates a :py:class:`~skhippr.cycles.shooting.ShootingBVP` and considers it as the first equation. The state vector ``x`` is the first unknown. If the underlying ODE is autonomous, the period ``T`` of the periodic solution is not known in advance and is appended to the unknowns. Correspondingly, a :py:class:`~skhippr.cycles.shooting.ShootingPhaseAnchor` equation is appended to the equations."""

    def __init__(
        self,
        ode: AbstractODE,
        T: float,
        period_k: int = 1,
        **kwargs_odesolver,
    ):
        bvp = ShootingBVP(ode, T, period_k, **kwargs_odesolver)
        equations = [bvp]
        unknowns = ["x"]

        if ode.autonomous:
            equations.append(ShootingPhaseAnchor(ode, bvp.x))
            unknowns.append("T")

        super().__init__(
            equations=equations, unknowns=unknowns, equation_determining_stability=bvp
        )
