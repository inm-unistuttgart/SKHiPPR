"""
Docstring for skhippr.odes.daes
"""

import numpy as np
from typing import override
from skhippr.odes.AbstractODE import AbstractODE, AbstractDAE


class PendulumDAE(AbstractDAE):
    """
    Docstring for PendulumDAE
    """

    def __init__(self, m, d, g, l, F, omega, phi, stability_method=None):

        M = np.diag([1, 1, m, m, 0])
        super().__init__(M=M, autonomous=False, stability_method=stability_method)
        self.d = d
        self.g = g
        self.l = l
        self.F = F
        self.omega = omega
        self.phi = phi

    @override
    def dynamics(self, t=None, x=None) -> np.ndarray:
        if t is None:
            t = self.t
        if x is None:
            x = self.x
        self.check_dimensions(t, x)

        f = np.zeros_like(x)
        f[0, ...] = x[2, ...]
        f[1, ...] = x[3, ...]
        f[2, ...] = (
            2 * x[0, ...] * x[4, ...]
            - self.d / (self.l**2) * x[2, ...]
            + self.F * np.sin(self.omega * t + self.phi)
        )
        f[3, ...] = (
            2 * x[1] * x[4]
            - self.M_small[2, 2] * self.g
            - self.d / (self.l**2) * x[3, ...]
        )
        f[4, ...] = x[0, ...] ** 2 + x[1, ...] ** 2 - self.l**2

        return f

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
            case _:
                raise NotImplementedError(
                    f"Derivative w.r.t {variable} not implemented in closed form."
                )

    def df_dx(self, t=None, x=None) -> np.ndarray:

        if t is None:
            t = self.t
        if x is None:
            x = self.x

        df_dx = np.zeros((5, *x.shape), dtype=x.dtype)
        df_dx[0, 2, ...] = 1
        df_dx[1, 3, ...] = 1
        df_dx[2, 0, ...] = 2 * x[4, ...]
        df_dx[2, 2, ...] = -self.d / (self.l**2)
        df_dx[3, 1, ...] = 2 * x[4, ...]
        df_dx[3, 3, ...] = -self.d / (self.l**2)
        df_dx[4, 0, ...] = 2 * x[0, ...]
        df_dx[4, 1, ...] = 2 * x[1, ...]
        df_dx[2, 4, ...] = 2 * x[0, ...]
        df_dx[3, 4, ...] = 2 * x[1, ...]

        return df_dx


class PendulumODE(AbstractODE):
    """Pendulum as ODE with angle phi as state variable."""

    def __init__(self, m, d, g, l, F, omega, phi, stability_method=None):

        super().__init__(autonomous=False, n_dof=2, stability_method=stability_method)
        self.m = m
        self.d = d
        self.g = g
        self.l = l
        self.F = F
        self.omega = omega
        self.phi = phi

    @override
    def dynamics(self, t=None, x=None) -> np.ndarray:

        if t is None:
            t = self.t
        if x is None:
            x = self.x
        self.check_dimensions(t, x)

        f = np.zeros_like(x)
        f[0, ...] = x[1, ...]
        f[1, ...] = (
            -self.g / self.l * np.sin(x[0, ...])
            - self.d / (self.m * self.l**2) * x[1, ...]
            + (
                self.F
                / (self.m * self.l)
                * np.sin(self.omega * t + self.phi)
                * np.cos(x[0, ...])
            )
        )

        return f

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
            case _:
                raise NotImplementedError(
                    f"Derivative w.r.t {variable} not implemented in closed form."
                )

    def df_dx(self, t=None, x=None) -> np.ndarray:
        if t is None:
            t = self.t
        if x is None:
            x = self.x

        df_dx = np.zeros((2, 2, *x.shape[1:]), dtype=x.dtype)
        df_dx[0, 1, ...] = 1
        df_dx[1, 0, ...] = -self.g / self.l * np.cos(x[0, ...]) - (
            self.F / (self.m * self.l)
        ) * np.sin(self.omega * t + self.phi) * np.sin(x[0, ...])
        df_dx[1, 1, ...] = -self.d / (self.m * self.l**2)

        return df_dx


class FrictionOscillator(AbstractDAE):
    """Oscillator chain with n_blocks blocks, of which the last one is in frictional contact with the ground."""

    def __init__(
        self,
        stiffnesses,
        dampings,
        masses,
        g,
        mu,
        forcing_amplitudes,
        forcing_phases,
        prox_parameter=1,
        stability_method=None,
    ):

        stiffnesses = np.atleast_1d(stiffnesses)
        dampings = np.atleast_1d(dampings)
        masses = np.atleast_1d(masses)
        forcing_amplitudes = np.atleast_1d(forcing_amplitudes)
        forcing_phases = np.atleast_1d(forcing_phases)

        M_small = np.diag(np.hstack((np.ones(len(masses)), masses, 0)))

        if len(masses) != len(stiffnesses):
            raise ValueError(
                f"Length of masses ({len(masses)}) must match length of stiffnesses ({len(stiffnesses)})."
            )

        if len(dampings) != len(stiffnesses):
            raise ValueError(
                f"Length of dampings ({len(dampings)}) must match length of stiffnesses ({len(stiffnesses)})."
            )

        if len(forcing_amplitudes) != len(stiffnesses):
            raise ValueError(
                f"Length of forcing amplitudes ({len(forcing_amplitudes)}) must match length of stiffnesses ({len(stiffnesses)})."
            )

        if len(forcing_phases) != len(stiffnesses):
            raise ValueError(
                f"Length of forcing phases ({len(forcing_phases)}) must match length of stiffnesses ({len(stiffnesses)})."
            )

        super().__init__(M_small, False, stability_method)
        self.stiffnesses = stiffnesses
        self.dampings = dampings
        self.forcing_amplitudes = forcing_amplitudes
        self.forcing_phases = forcing_phases
        self.mu = mu
        self.prox_parameter = prox_parameter

        self.x = np.zeros((2 * len(masses) + 1))

        stiffnesses = np.append(stiffnesses, 0)
        dampings = np.append(dampings, 0)

        self.K = np.zeros((len(masses), len(masses)))
        self.D = np.zeros(self.K.shape)
        for i in range(len(masses)):

            self.K[i, i] = stiffnesses[i + 1] + stiffnesses[i]
            if i > 0:
                self.K[i, i - 1] = -stiffnesses[i]
            if i < len(masses) - 1:
                self.K[i, i + 1] = -stiffnesses[i + 1]

            self.D[i, i] = dampings[i + 1] + dampings[i]
            if i > 0:
                self.D[i, i - 1] = -dampings[i]
            if i < len(masses) - 1:
                self.D[i, i + 1] = -dampings[i + 1]

        self.jacobian_ode = np.block(
            [[np.zeros_like(self.K), np.eye(len(masses))], [-self.K, -self.D]]
        )

        self.lam_crit = self.mu * masses[-1] * g

    @property
    def q(self):
        return self.x[: len(self.stiffnesses), ...]

    @q.setter
    def q(self, value):
        self.x[: len(self.stiffnesses), ...] = value

    @property
    def q_dot(self):
        return self.x[len(self.stiffnesses) : -1, ...]

    @q_dot.setter
    def q_dot(self, value):
        self.x[len(self.stiffnesses) : -1, ...] = value

    @property
    def lam(self):
        return np.atleast_1d(self.x[-1, ...])

    @lam.setter
    def lam(self, value):
        self.x[-1, ...] = value

    def forcing(self, t=None):
        if t is None:
            t = self.t
        F = np.zeros((self.n_dof - 1, *np.atleast_1d(t).shape))
        for i in range(len(self.stiffnesses)):
            F[len(self.stiffnesses) + i, ...] = self.forcing_amplitudes[i] * np.sin(
                self.omega * t + self.forcing_phases[i]
            )
        return F

    def constraint(self, t=None, x=None) -> np.ndarray:
        if t is None:
            t = self.t
        if x is None:
            x = self.x
        self.check_dimensions(t, x)

        g = (
            x[-2, ...]
            + np.minimum(
                0,
                self.prox_parameter * (x[-1, ...] + self.lam_crit) - x[-2, ...],
            )
            + np.maximum(
                0,
                self.prox_parameter * (x[-1, ...] - self.lam_crit) - x[-2, ...],
            )
        )
        return g

    @override
    def dynamics(self, t=None, x=None) -> np.ndarray:
        if t is None:
            t = self.t
        if x is None:
            x = self.x
        self.check_dimensions(t, x)

        f = np.zeros_like(x)
        f[:-1, ...] = self.jacobian_ode @ x[:-1, ...] + self.forcing(t)

        # friction effects
        f[-2, ...] += x[-1, ...]
        f[-1, ...] = self.constraint(t, x)

        return f

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
            case _:
                raise NotImplementedError(
                    f"Derivative w.r.t {variable} not implemented in closed form."
                )

    def df_dx(self, t=None, x=None) -> np.ndarray:
        if t is None:
            t = self.t
        if x is None:
            x = self.x

        n = len(self.stiffnesses)
        df_dx = np.zeros((2 * n + 1, 2 * n + 1, *x.shape[1:]), dtype=x.dtype)
        df_dx[: 2 * n, : 2 * n, ...] = self.jacobian_ode

        df_dx[-2, -1, ...] = 1

        # Derivative of constraint equation
        dg_dqdot = np.zeros_like(self.lam)
        dg_dqdot[
            self.prox_parameter + self.lam_crit
            > np.abs(x[-2, ...] - self.prox_parameter * x[-1, ...]),
            ...,
        ] = 1

        dg_dlam = np.zeros_like(self.lam)
        dg_dlam[
            self.prox_parameter + self.lam_crit
            < np.abs(x[-2, ...] - self.prox_parameter * x[-1, ...]),
            ...,
        ] = self.prox_parameter

        df_dx[-1, -1, ...] = dg_dlam
        df_dx[-1, -2, ...] = dg_dqdot

        return df_dx
