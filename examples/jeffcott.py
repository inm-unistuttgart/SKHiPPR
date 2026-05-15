"""Jeffcott rotor analysis for thesis presentation."""

import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

from skhippr.Fourier import Fourier
from skhippr.cycles.hbm import HBMEquation
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.odes.AbstractODE import AbstractODE
from skhippr.solvers.continuation import BranchPoint, pseudo_arclength_continuator
from skhippr.solvers.newton import NewtonSolver
from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic
from skhippr.visualization.continuation import (
    plot_continuation,
    plot_floquet_exponent_continuation,
    plot_floquet_multiplier_continuation,
)
from skhippr.visualization.cycles import (
    animate_floquet_exponents,
    animate_floquet_multipliers,
)


class Jeffcott2(AbstractODE):
    """3rd order model, Alcorta2023 Eq. (5)"""

    def __init__(self, D_e, D_if, D_it, omega_t, omega, e):
        super().__init__(autonomous=False, n_dof=4)
        self.D_e = D_e
        self.D_if = D_if
        self.D_it = D_it
        self.omega_t = omega_t
        self.omega = omega
        self.e = e
        self.has_nontrivial_omega_derivative = True

    def dynamics(self, t=None, x=None):
        if x is None:
            x = self.x
        if t is None:
            t = self.t
        self.check_dimensions(t, x)

        q = x[:2, ...]
        dq = x[2:, ...]

        f = np.zeros_like(x)
        f[:2, ...] = dq

        forcing = np.array(
            [
                self.e * self.omega**2 * np.cos(self.omega * t),
                self.e * self.omega**2 * np.sin(self.omega * t),
            ]
        )

        factor_nonl = 0
        for i in range(2):
            factor_nonl += self.D_it * q[i, ...] * dq[i, ...]
            factor_nonl += 2 * self.omega_t**2 * q[i, ...] ** 2

        for i in range(2):
            f[2 + i, ...] = q[i, ...]
            f[2 + i, ...] += (self.D_if + self.D_e) * dq[i, ...]
            f[2 + i, ...] += np.sign(0.5 - i) * self.D_if * q[1 - i, ...]
            f[2 + i, ...] += factor_nonl * q[i, ...]
            f[2 + i, ...] += forcing[i, ...]

        return f

    def nontrivial_omega_derivative(self, t=None, x=None):
        if x is None:
            x = self.x
        if t is None:
            t = self.t
        self.check_dimensions(t, x)

        df_domega = np.zeros_like(x)
        df_domega[2, ...] = 2 * self.e * self.omega * np.cos(self.omega * t)
        df_domega[3, ...] = 2 * self.e * self.omega * np.sin(self.omega * t)

        return df_domega


class Jeffcott(AbstractODE):
    def __init__(
        self,
        m=1,
        k_nonl=1,
        l_0=1,
        K_lin=0,
        C_lin=0,
        omega=1,
        e=0.1,
        g=0,
        t=0,
        x=np.array([0.1, 0.0, 0.0, 0.0]),
        stability_method=None,
    ):
        super().__init__(False, 4, stability_method)
        self.m = m
        self.k_nonl = k_nonl

        if np.isscalar(K_lin):
            K_lin = np.array([[K_lin, 0], [0, K_lin]])
        self.K_lin = K_lin / m

        if np.isscalar(C_lin):
            C_lin = np.array([[C_lin, 0], [0, C_lin]])

        self.C_lin = C_lin / m
        self.omega = omega
        self.e = e
        self.g = g
        self.t = t
        self.x = x
        self.l_0 = l_0
        self.has_nontrivial_omega_derivative = True

    def dynamics(self, t=None, x=None):
        if x is None:
            x = self.x
        if t is None:
            t = self.t
        self.check_dimensions(t, x)

        q = x[:2, ...]
        dq = x[2:, ...]

        f = np.zeros_like(x)
        f[:2, ...] = dq

        forcing = np.array(
            [
                2 * self.e * self.omega**2 * np.cos(self.omega * t),
                2 * self.e * self.omega**2 * np.sin(self.omega * t) - self.g,
            ]
        )

        # linear components
        f[2:, ...] = -self.K_lin @ q - self.C_lin @ dq + forcing

        # nonlinear components
        l = np.sqrt(self.l_0**2 + q[0, ...] ** 2 + q[1, ...] ** 2)
        factor = -self.l_0 / self.m * self.k_nonl / l
        f[2, ...] += factor * q[0, ...]
        f[3, ...] += factor * q[1, ...]

        return f

    def closed_form_derivative(self, variable, t=None, x=None):
        raise NotImplementedError(
            "I want to use the finite difference approximation for the Jacobian in this example, so I don't implement the closed form derivative."
        )

    def nontrivial_omega_derivative(self, t=None, x=None):
        if x is None:
            x = self.x
        if t is None:
            t = self.t
        self.check_dimensions(t, x)

        df_domega = np.zeros_like(x)
        df_domega[2, ...] = 4 * self.e * self.omega * np.cos(self.omega * t)
        df_domega[3, ...] = 4 * self.e * self.omega * np.sin(self.omega * t)

        return df_domega


def main():
    """Run a frequency response curve analysis for the Jeffcott rotor."""

    fourier = Fourier(N_HBM=25, L_DFT=300, n_dof=4, real_formulation=True)
    stability_method = KoopmanHillSubharmonic(fourier, tol=1e-4, autonomous=False)
    solver = NewtonSolver(verbose=True)

    # case Alcorta2023 - p. 5 bottom right
    l0 = 1.2
    r = 0.01
    omega_t = l0 / (np.sqrt(6) * r)
    print(omega_t)
    D_e = 0.1
    D_if = 0.1
    D_it = 0
    e = 1e-3

    ode = Jeffcott2(D_e=D_e, D_if=D_if, D_it=D_it, omega_t=omega_t, omega=0.1, e=e)

    # WAS IST D_IT????

    ts = fourier.time_samples(ode.omega)
    x0_samples = np.array(
        [
            ode.e * np.cos(ode.omega * ts),
            ode.e * np.sin(ode.omega * ts),
            -ode.e * ode.omega * np.sin(ode.omega * ts),
            ode.e * ode.omega * np.cos(ode.omega * ts),
        ]
    )
    X0 = fourier.DFT(x0_samples)

    hbm = HBMEquation(
        ode=ode,
        omega=ode.omega,
        fourier=fourier,
        initial_guess=X0,
        period_k=1,
        stability_method=stability_method,
    )

    hbm.residual(update=True)

    solver.solve_equation(equation=hbm, unknown="X")
    solver.verbose = False

    initial_system = EquationSystem(
        equations=[hbm], unknowns=["X"], equation_determining_stability=hbm
    )

    frc: list[BranchPoint] = []

    for branch_point in pseudo_arclength_continuator(
        initial_system=initial_system,
        solver=solver,
        stepsize=0.1,
        stepsize_range=(0.001, 0.1),
        continuation_parameter="omega",
        initial_direction=1,
        verbose=True,
        num_steps=200,
    ):
        frc.append(branch_point)

        if branch_point.omega > 3:
            break

    ax = plot_continuation(
        frc,
        plot_fun=lambda point: np.max(
            np.linalg.norm(point.equations[0].x_time()[:2, :], axis=0)
        ),
    )
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"max radial displacement")

    # _, animation1 = animate_floquet_multipliers(hbm_set=frc)
    # _, animation2 = animate_floquet_exponents(hbm_set=frc)

    return  # animation1, animation2


if __name__ == "__main__":
    animations = main()
    tikzplotlib.save("jeffcott.tikz", axis_width="5cm", axis_height="5cm")
    plt.show()
