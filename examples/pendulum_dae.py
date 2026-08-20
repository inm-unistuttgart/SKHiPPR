"""Illustrate HBM applied to DAEs, with the example of the simple mathematical pendulum."""

import numpy as np
import matplotlib.pyplot as plt
import warnings
import tikzplotlib
from typing import override


from skhippr.odes.AbstractODE import AbstractODE, AbstractDAE
from skhippr.Fourier import Fourier
from skhippr.solvers.newton import NewtonSolver, ScipyFsolveSolver
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.cycles.hbm import HBMEquation, HBMEquationDAE
from skhippr.solvers.continuation import pseudo_arclength_continuator

from skhippr.stability.KoopmanHillProjection import (
    KoopmanHillSubharmonic,
    KoopmanHillDAE,
)


class PendulumDAE(AbstractDAE):
    """
    Models the mathematical pendulum as a free planar point mass together with a distance constraint to the origin.
    The state vector is [x, y, x_dot, y_dot, lambda]^T, where lambda is the Lagrange multiplier associated with the distance constraint.

    The equations of motion are given by:
    x_dot  = x_dot
    y_dot  = y_dot
    x_ddot = 2 * x * lambda - d / l^2 * x_dot + F * sin(omega * t + phi)
    y_ddot = 2 * y * lambda - m * g - d / l^2 * y_dot
         0 = x^2 + y^2 - l^2
    """

    def __init__(self, m, d, g, l, F, omega, phi, stability_method=None):

        M = np.diag([1, 1, m, m, 0])
        super().__init__(
            n_dof=5,
            autonomous=False,
            stability_method=stability_method,
            M_is_constant=True,
            invertible=False,
        )
        self._M_small = M
        self.d = d
        self.g = g
        self.l = l
        self.F = F
        self.omega = omega
        self.phi = phi
        self.masses = [1, 1, m, m]

    @override
    def M_small(self, t=None, x=None):
        return self._M_small

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
            2 * x[1] * x[4] - self.masses[2] * self.g - self.d / (self.l**2) * x[3, ...]
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
    """Mathematical pendulum as an ODE with the angle phi as state variable. Describes the same dynamics as the PendulumDAE (for same parameter choice).

    The equations of motion are given by:
    phi_dot = phi_dot
    phi_ddot = -g/l * sin(phi) - d/(m*l^2) * phi_dot + F/(m*l) * sin(omega*t + phi) * cos(phi)
    """

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


def plot_single_solution():

    solver = NewtonSolver(tolerance=1e-8, max_iterations=50, verbose=True)
    # solver = ScipyFsolveSolver(
    #     tolerance=1e-8, max_iterations=1000, verbose=True, use_fprime=True
    # )

    m = 1
    g = 10
    l = 1.0
    d = 0.1
    F = 0.5
    omega = 1.15
    phi = 0.0

    N_HBM = 10

    # ODE formulation
    ode = PendulumODE(m, d, g, l, F, omega, phi)
    dae = PendulumDAE(m, d, g, l, F, omega, phi)

    # HBM systems
    fourier_ode = Fourier(
        N_HBM=N_HBM, L_DFT=1024, n_dof=ode.n_dof, real_formulation=True
    )

    hbm_ode = HBMEquation(
        ode,
        omega,
        fourier=fourier_ode,
        initial_guess=np.zeros(ode.n_dof * (2 * fourier_ode.N_HBM + 1)),
        stability_method=KoopmanHillSubharmonic(
            fourier_ode, tol=1e-4, autonomous=False
        ),
    )
    sys_ode = EquationSystem(
        equations=[hbm_ode], unknowns="X", equation_determining_stability=hbm_ode
    )
    solver.solve(sys_ode)
    print("Solved ODE. \n")

    x_ode = hbm_ode.x_time()
    phi = x_ode[0, :]
    phi_dot = x_ode[1, :]
    x_dae_init = np.vstack(
        [
            l * np.sin(phi),
            -l * np.cos(phi),
            l * np.cos(phi) * phi_dot,
            l * np.sin(phi) * phi_dot,
            np.zeros_like(phi),
        ]
    )

    fourier_dae = Fourier(N_HBM=15, L_DFT=1024, n_dof=dae.n_dof, real_formulation=True)
    hbm_dae = HBMEquationDAE(
        dae,
        omega,
        fourier=fourier_dae,
        initial_guess=fourier_dae.DFT(x_dae_init),
        stability_method=KoopmanHillDAE(
            fourier_dae, tol=0, autonomous=False, tol_drazin=1e-5
        ),
    )
    print(f"Residual before solve: {np.max(np.abs(hbm_dae.residual(update=True)))}")
    sys_dae = EquationSystem(
        equations=[hbm_dae],
        unknowns="X",
        equation_determining_stability=hbm_dae,
    )

    solver.solve(sys_dae)
    print(f"Residual after solve: {np.max(np.abs(hbm_dae.residual(update=False)))}")
    print("Solved DAE. \n")

    x_dae_solved = hbm_dae.x_time()
    t = hbm_dae.fourier.time_samples(omega=hbm_dae.omega)

    fig, axs = plt.subplots(4, 2)

    for i in range(4):
        plt.figure()
        plt.plot(t, x_dae_init[i, :])
        plt.plot(t, x_dae_solved[i, :], "--")
        plt.legend(["ODE", "DAE"])
        tikzplotlib.save(f"plots/pendulum_timeseries_state_{i}.tikz")

        plt.figure()
        plt.semilogy(t, np.abs(x_dae_init[i, :] - x_dae_init[i, :]))
        plt.semilogy(t, np.abs(x_dae_init[i, :] - x_dae_solved[i, :]), "--")
        tikzplotlib.save(f"plots/pendulum_timeseries_error_{i}.tikz")

    # axs[0][0].set_title(
    #     f"Pendulum ODE (solid) vs. DAE N = {hbm_dae.fourier.N_HBM} (dashed)"
    # )

    # axs[0][0].set_title(f"Error between DAE solution and ODE solution")

    fig_FM, ax = plt.subplots(nrows=1, ncols=1)
    phis = np.linspace(0, 2 * np.pi, 250)
    ax.plot(np.cos(phis), np.sin(phis), "gray", label="Unit circle")
    ax.plot(np.real(hbm_ode.eigenvalues), np.imag(hbm_ode.eigenvalues), "x")
    ax.plot(np.real(hbm_dae.eigenvalues), np.imag(hbm_dae.eigenvalues), "+")
    ax.set_aspect("equal", "box")
    ax.set_title(f"Floquet multipliers pendulum N_HBM = {hbm_dae.fourier.N_HBM}")
    tikzplotlib.save("plots/pendulum_Floquet_multipliers.tikz")


def plot_frc():

    solver = ScipyFsolveSolver(
        tolerance=1e-8, max_iterations=1000, verbose=False, use_fprime=True
    )

    m = 1
    g = 9.81
    l = 1.0
    d = 0.05
    F = 0.5
    omega = 1.15
    phi = 0.0

    N_HBM = 15

    # Systems
    ode = PendulumODE(m, d, g, l, F, omega, phi)
    dae = PendulumDAE(m, d, g, l, F, omega, phi)

    fourier_ode = Fourier(
        N_HBM=N_HBM, L_DFT=1000, n_dof=ode.n_dof, real_formulation=True
    )
    fourier_dae = Fourier(
        N_HBM=N_HBM, L_DFT=1000, n_dof=dae.n_dof, real_formulation=True
    )

    hbm_ode = HBMEquation(
        ode,
        omega,
        fourier=fourier_ode,
        initial_guess=np.zeros(ode.n_dof * (2 * fourier_ode.N_HBM + 1)),
        stability_method=KoopmanHillSubharmonic(
            fourier_ode, tol=1e-4, autonomous=False
        ),
    )

    solver.solve_equation(hbm_ode, unknown="X")
    phi = hbm_ode.x_time()[0, :]
    phi_dot = hbm_ode.x_time()[1, :]

    x_dae_init = np.vstack(
        [
            l * np.sin(phi),
            -l * np.cos(phi),
            l * np.cos(phi) * phi_dot,
            l * np.sin(phi) * phi_dot,
            np.zeros_like(phi),
        ]
    )

    dae_initial_guess = np.zeros(dae.n_dof * (2 * fourier_ode.N_HBM + 1))
    dae_initial_guess[4] = 1.0
    dae_initial_guess += 1e-4 * np.random.rand(dae.n_dof * (2 * fourier_ode.N_HBM + 1))
    hbm_dae = HBMEquationDAE(
        dae,
        omega,
        fourier=fourier_dae,
        initial_guess=fourier_dae.DFT(x_dae_init),
        stability_method=KoopmanHillDAE(
            fourier_dae, tol=1e-4, autonomous=False, tol_drazin=1e-6
        ),
    )

    omegas = [[], [], []]
    amps = [[], [], []]
    fig, axs = plt.subplots(1, 2)

    for idx, hbm in enumerate([hbm_ode, hbm_dae]):

        sys = EquationSystem(
            equations=[hbm], unknowns=["X"], equation_determining_stability=hbm
        )

        for branch_point in pseudo_arclength_continuator(
            initial_system=sys,
            solver=solver,
            continuation_parameter="omega",
            stepsize=0.01,
            stepsize_range=[0.001, 0.1],
            num_steps=2000,
            verbose=True,
        ):
            if branch_point.stable:
                if len(omegas[1]) == 0:
                    stb = 0
                else:
                    stb = 2
                color = "r"
            else:
                stb = 1
                color = "b"

            if idx == 1:

                x_time = branch_point.equations[0].x_time()
                phi = np.atan2(x_time[0, :], -x_time[1, :])

                omegas[stb].append(branch_point.omega)
                amps[stb].append(
                    # np.max(np.abs(branch_point.equations[0].x_time()[0, :]))
                    np.max(phi)
                )

            axs[idx].plot(
                branch_point.omega,
                np.max(np.abs(branch_point.equations[0].x_time()[0, :])),
                f"{color}.",
            )

            if branch_point.omega > 4:
                break

    # axs[0].set_title("Pendulum ODE FRC")
    # axs[0].set_xlabel("omega")
    # axs[0].set_ylabel("phi max")
    # tikzplotlib.save("plots/pendulum_frc_ode")

    plt.figure()
    colors = ["r", "b", "r"]

    for om, amp, col in zip(omegas, amps, colors):
        plt.plot(om, amp, col)

    plt.title("Pendulum DAE FRC")
    plt.xlabel("omega")
    plt.ylabel("phi")

    tikzplotlib.save("plots/pendulum_frc_dae_phi")


if __name__ == "__main__":
    # plot_single_solution()
    plot_frc()
    plt.show()
