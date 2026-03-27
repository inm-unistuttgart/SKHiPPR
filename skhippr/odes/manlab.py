import numpy as np
from typing import Any, override

from numpy._typing._array_like import NDArray

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
        xi: np.ndarray,
        epsilon: float,
        omega: float,
        amp_forcing: np.ndarray,
        phase_forcing: np.ndarray,
        x_forcing: np.ndarray,
    ):

        n_dof = x.shape[0]
        super().__init__(autonomous=False, n_dof=n_dof)
        self.t = t
        self.x = x

        if n_dof % 2 != 0:
            raise ValueError("n_dof must be even: x = [q, dq] with same length.")

        self.n_modes = n_dof // 2
        self.mode_numbers = np.arange(1, self.n_modes + 1, dtype=float)
        self.omegas = np.pi**2 * self.mode_numbers**2

        self.xi = np.asarray(xi)
        if len(self.xi) != self.n_modes:
            raise ValueError(
                f"xi must have size {self.n_modes}, got {self.xi.shape[0]}."
            )
        self.epsilon = epsilon

        # Modal components of the forcing
        self.omega = omega
        if len(amp_forcing) != len(phase_forcing) or len(amp_forcing) != len(x_forcing):
            raise ValueError(
                f"amp_forcing, phase_forcing and x_forcing must have the same length but got {len(amp_forcing)}, {len(phase_forcing)}, {len(x_forcing)}."
            )
        amp_forcing = np.asarray(amp_forcing)
        self.phase_forcing = np.asarray(phase_forcing)
        x_forcing = np.asarray(x_forcing)

        # forcing -- beam parameters from Matlab script
        h = 1e-3
        b = 0.1

        S = h * b
        I = b * h**3 / 12

        r = np.sqrt(I / S)

        # note the difference between multiplying by r and dividing by h in lines 121-122 of Manlab
        P = np.sqrt(2) * amp_forcing * r / h

        self.Q = P[np.newaxis, :] * np.sin(
            np.pi * self.mode_numbers[:, np.newaxis] * x_forcing[np.newaxis, :]
        )

        self.D = np.diag(2 * self.xi * self.omegas)
        self.K = np.diag(self.omegas**2)

    def forcing(self, t=None) -> np.ndarray:
        if t is None:
            t = self.t
        t = np.atleast_1d(t)

        cos_terms = np.cos(
            self.omega * t[np.newaxis, :] + self.phase_forcing[:, np.newaxis]
        )
        return self.Q @ cos_terms

    def N(self, q: np.ndarray = None) -> np.ndarray:
        if q is None:
            q = self.x[: self.n_modes, ...]
        if len(q.shape) == 1:
            q = q[:, np.newaxis]
        return 0.5 * np.sum(self.mode_numbers[:, np.newaxis] * q**2, axis=0)

    def dN_dq(self, q: np.ndarray = None) -> np.ndarray:
        if q is None:
            q = self.x[: self.n_modes, ...]
        if len(q.shape) == 1:
            q = q[:, np.newaxis]
        return self.mode_numbers[:, np.newaxis] * q

    @override
    def dynamics(self, t=None, x=None) -> np.ndarray:

        if t is None:
            t = self.t
        if x is None:
            x = self.x

        self.check_dimensions(t, x)

        q, dq = np.split(x, 2, axis=0)
        if q.ndim == 1:
            q = q[:, np.newaxis]
            dq = dq[:, np.newaxis]

        ddq = (
            -self.D @ dq
            - self.K @ q
            - self.epsilon * self.N(q) * self.omegas[:, np.newaxis] * q
            + self.forcing(t)
        )

        return np.squeeze(np.vstack((dq, ddq)))

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

    def df_dx(self, t=None, x=None):

        if t is None:
            t = self.t
        if x is None:
            x = self.x

        n = self.n_modes
        q, _ = np.split(x, 2, axis=0)
        dN_dq = self.dN_dq(q)
        if q.ndim == 1:
            q = q[:, np.newaxis]
        omega_q = (self.omegas[:, np.newaxis] * q)[:, np.newaxis, :]
        dN_dq_3d = dN_dq[np.newaxis, :, :]

        ddq_dq = -self.K - self.epsilon * np.diag(self.N(q) * self.omegas)
        ddq_dq = ddq_dq[:, :, np.newaxis] - self.epsilon * (omega_q * dN_dq_3d)
        df_dx = np.vstack(
            [
                np.hstack(
                    [
                        np.zeros(ddq_dq.shape),
                        np.eye(n)[:, :, np.newaxis]
                        * np.ones((1, 1, dN_dq_3d.shape[2])),
                    ]
                ),
                np.hstack(
                    [
                        ddq_dq,
                        -self.D[:, :, np.newaxis] * np.ones((1, 1, dN_dq_3d.shape[2])),
                    ]
                ),
            ]
        )

        return np.squeeze(df_dx)


def test_hinged_hinged():
    """Test the derivative of the HingedHinged ODE against a finite difference approximation."""
    n_modes = 3
    n_dof = 2 * n_modes
    t = 0.0
    x = np.random.rand(n_dof)
    xi = np.random.rand(n_modes) * 0.1
    epsilon = 0.01
    omega = 1.0
    amp_forcing = np.random.rand(n_modes)
    phase_forcing = np.random.rand(n_modes) * 2 * np.pi
    x_forcing = np.random.rand(n_modes) * np.pi

    ode = HingedHinged(
        t=t,
        x=x,
        xi=xi,
        epsilon=epsilon,
        omega=omega,
        amp_forcing=amp_forcing,
        phase_forcing=phase_forcing,
        x_forcing=x_forcing,
    )

    df_dx_analytical = ode.closed_form_derivative("x")
    df_dx_numerical = ode.finite_difference_derivative("x", h_step=1e-6)

    assert np.allclose(
        df_dx_analytical, df_dx_numerical, atol=1e-5
    ), "df/dx does not match finite difference approximation."


if __name__ == "__main__":
    test_hinged_hinged()
