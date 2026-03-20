import numpy as np
from typing import override

from skhippr.odes.AbstractODE import AbstractODE


class HingedHinged(AbstractODE):
    """Hinged-hinged beam reduced by modal expansion in first-order form.

    The state is ``x = [q_1, ..., q_K, dq_1, ..., dq_K]`` with ``K`` modes.
    The reduced equations are

            qdot_j = dq_j
            dqdot_j = -2*xi_j*omega_j*dq_j - omega_j^2*q_j
                              - epsilon*Nbar*omega_j*q_j + Q_j(t)

    where

            Nbar = 0.5 * sum_i omega_i * q_i^2,
            omega_j = pi^2 * j^2,
            Q_j(t) = F_j * cos(forcing_omega * t).
    """

    def __init__(
        self,
        t: float,
        x: np.ndarray,
        n_dof: int,
        xi: np.ndarray,
        epsilon: float,
        F: np.ndarray,
        forcing_omega: float,
    ):
        super().__init__(autonomous=False, n_dof=n_dof)
        self.t = t
        self.x = x

        if n_dof % 2 != 0:
            raise ValueError("n_dof must be even: x = [q, dq] with same length.")

        self.n_modes = n_dof // 2
        self.mode_numbers = np.arange(1, self.n_modes + 1, dtype=float)
        self.omega = np.pi**2 * self.mode_numbers**2

        self.xi = np.asarray(xi)
        self.epsilon = epsilon
        self.F = np.asarray(F)
        self.forcing_omega = forcing_omega

        if self.xi.shape[0] != self.n_modes:
            raise ValueError(
                f"xi must have size {self.n_modes}, got {self.xi.shape[0]}."
            )
        if self.F.shape[0] != self.n_modes:
            raise ValueError(f"F must have size {self.n_modes}, got {self.F.shape[0]}.")

    def _mode_view(self, arr: np.ndarray, vec: np.ndarray) -> np.ndarray:
        return vec.reshape((self.n_modes,) + (1,) * (arr.ndim - 1))

    @override
    def dynamics(self, t=None, x=None) -> np.ndarray:

        if t is None:
            t = self.t
        if x is None:
            x = self.x

        self.check_dimensions(t, x)

        q, dq = np.split(x, 2, axis=0)
        omega_view = self._mode_view(q, self.omega)
        F_view = self._mode_view(q, self.F)

        nbar = 0.5 * np.sum(omega_view * q**2, axis=0)
        forcing = F_view * np.cos(self.forcing_omega * t)

        ddq = np.zeros_like(q)
        ddq -= 2 * self._mode_view(q, self.xi) * omega_view * dq
        ddq -= omega_view**2 * q
        ddq -= self.epsilon * nbar[np.newaxis, ...] * omega_view * q
        ddq += forcing

        return np.vstack((dq, ddq))

    @override
    def closed_form_derivative(self, variable, t=None, x=None):

        if t is None:
            t = self.t
        if x is None:
            x = self.x

        self.check_dimensions(t, x)

        match variable:
            case "x":
                return self.df_dx(t, x)
            case "xi":
                return self.df_dxi(t, x)
            case "epsilon":
                return self.df_depsilon(t, x)
            case "F":
                return self.df_dF(t, x)
            case "forcing_omega":
                return self.df_dforcing_omega(t, x)
            case _:
                raise NotImplementedError(
                    f"Derivative w.r.t {variable} not implemented in closed form."
                )

    def df_dx(self, t=None, x=None):

        if t is None:
            t = self.t
        if x is None:
            x = self.x

        n = self.n_modes
        q, _ = np.split(x, 2, axis=0)
        omega_view = self._mode_view(q, self.omega)
        nbar = 0.5 * np.sum(omega_view * q**2, axis=0)

        df_dx = np.zeros((2 * n, *x.shape), dtype=x.dtype)

        for i in range(n):
            df_dx[i, n + i, ...] = 1

        dddq_ddq = -2 * self._mode_view(q, self.xi) * omega_view
        for j in range(n):
            df_dx[n + j, n:, ...] = 0
            df_dx[n + j, n + j, ...] = dddq_ddq[j, ...]

        for j in range(n):
            for i in range(n):
                df_dx[n + j, i, ...] = (
                    -self.epsilon
                    * self.omega[j]
                    * self.omega[i]
                    * q[j, ...]
                    * q[i, ...]
                )

            df_dx[n + j, j, ...] += (
                -self.omega[j] ** 2
                - self.epsilon * self.omega[j] * nbar
                - self.epsilon * self.omega[j] ** 2 * q[j, ...] ** 2
            )

        return df_dx

    def df_dxi(self, t=None, x=None):

        if x is None:
            x = self.x

        n = self.n_modes
        _, dq = np.split(x, 2, axis=0)

        df_dxi = np.zeros((2 * n, n, *x.shape[1:]), dtype=x.dtype)
        for j in range(n):
            df_dxi[n + j, j, ...] = -2 * self.omega[j] * dq[j, ...]

        return df_dxi

    def df_depsilon(self, t=None, x=None):

        if x is None:
            x = self.x

        n = self.n_modes
        q, _ = np.split(x, 2, axis=0)
        nbar = 0.5 * np.sum(self._mode_view(q, self.omega) * q**2, axis=0)

        df_dep = np.zeros((2 * n, 1, *x.shape[1:]), dtype=x.dtype)
        for j in range(n):
            df_dep[n + j, 0, ...] = -self.omega[j] * q[j, ...] * nbar

        return df_dep

    def df_dF(self, t=None, x=None):

        if t is None:
            t = self.t
        if x is None:
            x = self.x

        n = self.n_modes
        df_dF = np.zeros((2 * n, n, *x.shape[1:]), dtype=x.dtype)
        cos_term = np.cos(self.forcing_omega * t)
        for j in range(n):
            df_dF[n + j, j, ...] = cos_term

        return df_dF

    def df_dforcing_omega(self, t=None, x=None):

        if t is None:
            t = self.t
        if x is None:
            x = self.x

        n = self.n_modes
        df_dom = np.zeros((2 * n, 1, *x.shape[1:]), dtype=x.dtype)
        base = -t * np.sin(self.forcing_omega * t)
        for j in range(n):
            df_dom[n + j, 0, ...] = self.F[j] * base

        return df_dom
