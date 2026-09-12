"""
Find and plot the equilibrium of the Truss system using SKHiPPR.

"""

from typing import override


import numpy as np
import matplotlib.pyplot as plt

# --- System function ---
from skhippr.odes.AbstractODE import AbstractODE

# --- Solver ---
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.solvers.newton import NewtonSolver

# --- Continuation ---
from skhippr.solvers.continuation import pseudo_arclength_continuator, BranchPoint

# --- Visualization ---
from skhippr.visualization.continuation import plot_continuation
from skhippr.visualization.equilibria import plot_equilibrium, plot_eigenvalues


class Truss(AbstractODE):
    """
    Autonomous truss system as a subclass of :py:class:`~skhippr.odes.AbstractODE.AbstractODE`. A mass ``m`` can move horizontally with viscous damping (``c``). It is attached to a linear spring (``k``, ``l_0``) mounted at the point ``(0, a)``. A constant force ``F`` acts on the mass. For small values of ``F``, there are three coexisting equilibria. The equations of motion are ::

        dx[0]/dt = x[1]
        dx[1]/dt = -k/m * x[0] + k/m * x[0] * l_0 / sqrt(a**2 + x[0]**2) + F/m - c/m * x[1]

    """

    def __init__(
        self,
        x: np.ndarray,
        k: float,
        c: float,
        F: float,
        a: float,
        l_0: float,
        m: float,
    ):
        super().__init__(autonomous=True, n_dof=2)
        self.x = x
        self.k = k
        self.c = c
        self.F = F
        self.a = a
        self.l_0 = l_0
        self.m = m

    @override
    def dynamics(self, t=None, x=None):
        if x is None:
            x = self.x

        self.check_dimensions(t=t, x=x)

        q = x[0, ...]
        q_dot = x[1, ...]

        f = np.zeros_like(x)
        f[0, ...] = q_dot
        f[1, ...] = -self.k / self.m * q
        f[1, ...] += self.k / self.m * q * self.l_0 / np.sqrt(self.a**2 + q**2)
        f[1, ...] = f[1, ...] + self.F / self.m - self.c / self.m * q_dot

        return f

    @override
    def closed_form_derivative(self, variable, t=None, x=None):

        if x is None:
            x = self.x

        self.check_dimensions(t=t, x=x)

        match variable:
            case "x":
                return self.df_dx(x)
            case "F":
                return self.df_dF(x)
            case "k":
                return self.df_dk(x)
            case "c":
                return self.df_dc(x)
            case _:
                raise NotImplementedError(
                    f"Derivative w.r.t {variable} not implemented in closed form."
                )

    def df_dF(self, x=None):
        if x is None:
            x = self.x
        df_dF = np.zeros_like(x)
        df_dF[1, ...] = 1 / self.m
        return df_dF[:, np.newaxis, ...]

    def df_dc(self, x=None):
        if x is None:
            x = self.x
        df_dc = np.zeros_like(x)
        df_dc[1, ...] = -x[1, ...] / self.m
        return df_dc[:, np.newaxis, ...]

    def df_dk(self, x=None):
        if x is None:
            x = self.x
        q = x[0, ...]
        df_dk = np.zeros_like(x)
        df_dk[1, ...] = -1 / self.m * q
        df_dk[1, ...] += 1 / self.m * q * self.l_0 / np.sqrt(self.a**2 + q**2)
        return df_dk[:, np.newaxis, ...]

    def df_dx(self, x=None):
        if x is None:
            x = self.x

        q = x[0, ...]

        df_dx = np.zeros((x.shape[0], x.shape[0], *x.shape[1:]))
        df_dx[0, 1, ...] = 1
        df_dx[1, 1, ...] = -self.c / self.m
        df_dx[1, 0, ...] = -self.k / self.m
        df_dx[1, 0, ...] += self.k / self.m * self.l_0 / np.sqrt(self.a**2 + q**2)
        df_dx[1, 0, ...] -= (
            self.k / self.m * self.l_0 * q**2 / (np.sqrt(self.a**2 + q**2) ** 3)
        )
        return df_dx


def compute_equilibria():

    # --- Instantiation of the ODE at initial point ---
    ode = Truss(x=[1.0, 2.0], F=-0.5, a=1.0, l_0=1.2, k=3.0, m=1.0, c=0.5)
    solver = NewtonSolver(verbose=True)

    # --- ODEs can be directly considered in an EquationSystem for solving the right-hand side to zero ---
    equation_sys = EquationSystem(
        equations=[ode], unknowns=["x"], equation_determining_stability=ode
    )
    solver.solve(equation_sys)

    # --- start the continuation from the found equilibrium ---
    solver.verbose = False
    branch: list[BranchPoint] = []

    for branch_point in pseudo_arclength_continuator(
        initial_system=equation_sys,
        solver=solver,
        stepsize=0.01,
        stepsize_range=(0.001, 0.01),
        continuation_parameter="F",
        initial_direction=1,
        verbose=True,
        num_steps=400,
    ):
        branch.append(branch_point)
        # break if F exceeds maximum
        if branch_point.F > 0.5:
            break

    return equation_sys, branch


if __name__ == "__main__":
    initial_eq, branch = compute_equilibria()
    plot_equilibrium(
        ode=initial_eq, title=f"Equilibrium (F = {initial_eq.equations[0].F})"
    )
    plot_eigenvalues(ode=initial_eq)
    plot_continuation(branch, marker="x", stable_label="stable", ylabel="x")
    plt.show()
