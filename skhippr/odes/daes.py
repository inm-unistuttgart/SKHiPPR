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

        M = np.diag([m, m, m, m, 0])
        super().__init__(M=M, autonomous=False, stability_method=stability_method)
        self.d = d
        self.g = g
        self.l = l
        self.F = F
        self.omega = (omega,)
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
        f[3, ...] = 2 * x[1] * x[4] - self.m * self.g - self.d / (self.l**2) * x[3, ...]
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

        df_dx = np.zeros((2, 2, *x.shape), dtype=x.dtype)
        df_dx[0, 1, ...] = 1
        df_dx[1, 0, ...] = -self.g / self.l * np.cos(x[0, ...]) - (
            self.F / (self.m * self.l)
        ) * np.sin(self.omega * t + self.phi) * np.sin(x[0, ...])
        df_dx[1, 1, ...] = -self.d / (self.m * self.l**2)

        return df_dx
