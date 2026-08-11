"""Demonstrates the complete pseudo-arclength continuation workflow on a nonlinear equation.

* Use of the :py:class:`~skhippr.equations.Circle.Circle` class as a concrete subclass of :py:class:`~skhippr.equations.AbstractEquation.AbstractEquation` for a problem formulation
* Illustrates how (one or multiple) multiple :py:class:`~skhippr.equations.AbstractEquation.AbstractEquation` objects are collected into an :py:class:`~skhippr.equations.EquationSystem.EquationSystem`
* Continuation of a solution branch emerging from the :py:class:`~skhippr.equations.EquationSystem.EquationSystem` with and without an explicit continuation parameter
* Visualizing the continuation using SKHiPPR internal methods.
"""

from typing import override

import numpy as np
import matplotlib.pyplot as plt

# --- Solver---
from skhippr.solvers.newton import NewtonSolver

# --- Equations ---
from skhippr.equations.AbstractEquation import AbstractEquation
from skhippr.equations.EquationSystem import EquationSystem

# --- Continuation ---
from skhippr.solvers.continuation import pseudo_arclength_continuator, BranchPoint

# --- Visualization ---
from skhippr.visualization.continuation import plot_continuation


# --- Define the equations as classes with attributes as (possible) unknowns. Abstract methods residual_function and closed_form_derivative must be implemented. ---
class CircleEquation(AbstractEquation):
    """Has attributes ``y`` and ``radius``.
    The residual function::

        self.y[0] ** 2 + self.y[1] ** 2 - self.radius**2

    is zero when the point ``y`` lies on the circle with radius ``radius``.
    """

    def __init__(self, y: np.ndarray, radius=1):
        super().__init__(None)
        self.y = y
        self.radius = radius

    @override
    def residual_function(self):
        return np.atleast_1d(self.y[0] ** 2 + self.y[1] ** 2 - self.radius**2)

    @override
    def closed_form_derivative(self, variable):
        match variable:
            case "y":
                return np.atleast_2d(np.array([2 * self.y[0], 2 * self.y[1]]))
            case "radius":
                return np.atleast_2d(-2 * self.radius)
            case _:
                return np.atleast_2d(0)


class AngleEquation(AbstractEquation):
    """Has attributes ``y`` and ``theta``. If ``y`` encloses the angle ``theta`` with the positive x axis, then
    the residual function::


        self.y[1] * np.cos(self.theta) - self.y[0] * np.sin(self.theta)


    vanishes.
    """

    def __init__(self, y: np.ndarray, theta=1):
        super().__init__(None)
        self.y = y
        self.theta = theta

    def residual_function(self):
        return np.atleast_1d(
            self.y[1] * np.cos(self.theta) - self.y[0] * np.sin(self.theta)
        )

    def closed_form_derivative(self, variable):
        match variable:
            case "y":
                return np.atleast_2d(
                    [
                        -np.sin(np.squeeze(self.theta)),
                        np.cos(np.squeeze(self.theta)),
                    ]
                )
            case "theta":
                return np.atleast_2d(
                    -self.y[1] * np.sin(self.theta) - self.y[0] * np.cos(self.theta)
                )
            case _:
                return np.atleast_2d(0)


# --- And now both together as one equation ---
class CircleWithPhase(AbstractEquation):
    def __init__(self, x, radius, theta, other):
        super().__init__(stability_method=None)
        self.x = x
        self.radius = radius
        self.theta = theta
        self.other = other

    def residual_function(self):
        return np.squeeze(
            np.array(
                [
                    self.x[0] ** 2 + self.x[1] ** 2 - self.radius**2,
                    self.x[1] * np.cos(self.theta) - self.x[0] * np.sin(self.theta),
                ]
            )
        )

    def closed_form_derivative(self, variable):
        match variable:
            case "x":
                return np.array(
                    [
                        [2 * self.x[0], 2 * self.x[1]],
                        [
                            -np.sin(np.squeeze(self.theta)),
                            np.cos(np.squeeze(self.theta)),
                        ],
                    ]
                )
            case "radius":
                radius = np.atleast_1d(self.radius)
                return np.array([-2 * radius, [0]])
            case "theta":
                return np.array(
                    [
                        [0],
                        -self.x[1] * np.sin(self.theta)
                        - self.x[0] * np.cos(self.theta),
                    ]
                )
            case "other":
                return self.other
            case "other2":
                return self.other[2]
            case _:
                raise NotImplementedError


def main():
    """
    Runs a demonstration of pseudo-arclength continuation for solving a nonlinear problem
    using Newton's method on the circle equation.
    The process includes:

    #. Instantiating a :py:class:`~skhippr.solvers.newton.NewtonSolver` with the solver configuration
    #. Instantiating a :py:class:`Circle` object which contains the residual
    #. Demonstrating that the :py:class:`~skhippr.solvers.newton.NewtonSolver` can immediately solve the :py:class:`Circle` for the scalar unknown ``radius``, but not for the array unknown ``y``
    #. Constructing an :py:class:`~skhippr.equations.EquationSystem.EquationSystem` for the unknown ``y`` and using the :py:func:`~skhippr.cycles.continuation.pseudo_arclength_continuator` to iterate along the solution branch.
    #. Constructing another :py:class:`~skhippr.equations.EquationSystem.EquationSystem` by appending an  second :py:class:`AngleEquation` and solving the resulting system directly for ``y``.
    #. Using the :py:func:`~skhippr.cycles.continuation.pseudo_arclength_continuator` to iterate along the solution branch with the extended :py:class:`~skhippr.equations.EquationSystem.EquationSystem` and the explicit continuation parameter ``theta``
    #. Plotting the results with :py:func:`~skhippr.visualization.continuation.plot_continuation`.

    Returns
    -------
    None
    """
    print("Circle example.")

    # Instantiate solver
    solver = NewtonSolver()

    # Instantiate equation to be solved
    radius = 2
    equ = CircleEquation(y=np.array([0.9 * radius, 0]), radius=radius)

    solve_equation_for_radius(equ, solver)
    attempt_to_solve_equation_for_y(equ, solver)

    # Instantiate EquationSystem encoding the circle eq. and the unknown
    equ_sys = EquationSystem([equ], unknowns=["y"])

    # We need a slightly different syntax but still, we cannot solve this immediately...
    attempt_to_solve_system_for_y(equ_sys, solver)

    # But we can perform continuation to find multiple solutions of the equation system!
    continuation_and_plot(equ_sys=equ_sys, solver=solver, continuation_parameter=None)

    # Make the equation system well-defined by appending an equation to fix the angle
    ang = AngleEquation(y=equ.y, theta=0)
    equ_sys_with_angle = EquationSystem([equ, ang], unknowns=["y"])
    attempt_to_solve_system_for_y(equ_sys_with_angle, solver)

    # And again, we can do continuation - but now a parameter must act as (explicit) continuation parameter
    continuation_and_plot(
        equ_sys=equ_sys_with_angle,
        solver=solver,
        continuation_parameter="theta",
        param_max=2 * np.pi,
    )

    equ_sys_with_angle.equations[1].theta = np.pi / 4
    continuation_and_plot(
        equ_sys=equ_sys_with_angle,
        solver=solver,
        continuation_parameter="radius",
        param_max=8,
    )


def solve_equation_for_radius(equ: CircleEquation, solver: NewtonSolver):
    """Successfully attempt to solve the circle equation for the radius."""
    radius_old = equ.radius
    # Verify that the provided initial conditions do not solve the equation
    print(
        f"Initial guess does not solve the equation: r = {equ.radius} != {np.linalg.norm(equ.y)} = ||y||"
    )

    # We can solve immediately for the radius corresponding to a fixed y
    solver.solve_equation(equ, "radius")
    print(f"Solved for radius: r = {equ.radius} == {np.linalg.norm(equ.y)} = ||y||")
    # Note that scalar unknowns are transformed into 1-D numpy arrays.

    # Revert to original setup
    equ.radius = radius_old


def attempt_to_solve_equation_for_y(equ, solver):
    """Unsuccessfully attempt to solve the circle equation for the array unknown y."""

    # We can NOT solve immediately for y because the number of equations does not match the number of unknowns
    try:
        solver.solve_equation(equ, "y")
    except ValueError as ME:
        print(
            f"Trying to solve equation immediately for 'y' yielded \n ValueError: '{ME}' "
        )


def attempt_to_solve_system_for_y(equ_sys, solver):
    """Attempt to solve the equation system consisting of one or more equations for the array unknown y."""
    try:
        solver.solve(equ_sys)
        print(f"Solved the equation system successfully.")
    except ValueError as ME:
        print(
            f"Trying to solve equation system immediately for 'y' yielded ValueError: \n'{ME}' "
        )


def continuation_and_plot(equ_sys, solver, continuation_parameter=None, param_max=None):
    """Perform continuation on the equation system and plot the results.

    Parameters
    ----------

    equ_sys : EquationSystem
        The equation system to be solved and continued.
    solver : NewtonSolver
        The solver to be used for the continuation.
    continuation_parameter : str, optional
        The parameter to be used for continuation, e.g. 'radius' or 'theta'. Specify ``None`` if the ``equ_sys`` is underdetermined and does not need an explicit continuation parameter.
    param_max : float, optional
        The maximum value of the continuation parameter to stop the continuation.
    """
    # Set up the plot
    fig, ax = plt.subplots(1, 1)
    ax.set_title(f"Continuation with parameter {continuation_parameter}")
    y1_prev = 0

    branch: list[BranchPoint] = []
    # Iterate through points on the circle
    for branch_point in pseudo_arclength_continuator(
        initial_system=equ_sys,  # must be of type EquationSystem
        solver=solver,
        initial_direction=1,  # direction of tangent in x[-1] direction
        verbose=False,  # prints after every continuation step
        num_steps=300,
        stepsize=0.1,  # initial stepsize
        stepsize_range=(0.05, 0.15),
        continuation_parameter=continuation_parameter,
    ):

        branch.append(branch_point)

        # Stopping criteria etc. can be user-defined in the loop
        if (
            param_max is not None
            and getattr(branch_point, continuation_parameter) > param_max
        ):
            break

        if param_max is None and y1_prev < 0 and branch_point.y[1] >= 0:
            # Modify constant parameter of the equation (first equation in the equation system) and keep going
            branch_point.equations[0].radius += 2

        y1_prev = branch_point.y[1]

    plot_continuation(branch=branch, plot_fun=lambda point: point.y, ax=ax)
    plt.axis("equal")


if __name__ == "__main__":
    main()
    plt.show()
