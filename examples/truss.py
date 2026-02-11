"""
Find and plot the equilibrium of the Truss system using SKHiPPR.

"""
import numpy as np
import matplotlib.pyplot as plt

# --- System function ---
from skhippr.odes.autonomous import Truss

# --- Solver ---
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.solvers.newton import NewtonSolver

# --- Continuation ---
from skhippr.solvers.continuation import pseudo_arclength_continuator, BranchPoint

# --- Visualization ---
from skhippr.visualization.continuation import plot_continuation
from skhippr.visualization.equilibria import (
    plot_equilibrium,
    plot_eigenvalues
)

def main():
    """
    Demonstrates solving an equation for an equilibrium and visualizes the equilibrium as well as its eigenvalues using SKHiPPR visualization methods.
    Performs continuation over the parameter `F` and then visualizes the branch with its stability intervals.
    
    This function instantiates a :py:class:`~skhippr.odes.autonomous.Truss` object as a subclass of :py:class:`~skhippr.odes.AbstractODE.AbstractODE` and solves it with either :py:func:`skhippr.solvers.newton.NewtonSolver.solve_equation` or with :py:func:`skhippr.solvers.newton.NewtonSolver.solve` after embedding it into an :py:class:`~skhippr.equations.EquationSystem.EquationSystem`.
    
    It generates 3 figures:

    #. A plot of the ode equilibrium
    #. A plot of the equilibrium eigenvalues visualizing its stability
    #. A continuation plot highlighting the fold bifurcations inherent to the system.
    
    """
    # --- Instantiation of the ODE at initial point ---
    ode = Truss(
        x=[1.0, 2.0], F=-0.5,a=1.0,l_0=1.2,k=3.0,m=1.0,c=0.5
    )
    solver = NewtonSolver(verbose=True)
    
    # --- ODEs can be packed into an EquationSystem for solving ---
    equation_sys = EquationSystem(
        equations=[ode],
        unknowns=["x"],
        equation_determining_stability=ode
    )
    solver.solve(equation_sys)
    
    # --- or passed to a solver directly using a different method ---
    solver.solve_equation(equation=ode, unknown="x")
    
    # --- Standard SKHiPPR visualization calls create and return new figures for each plot ---
    plot_equilibrium(ode = ode)
    
    # --- SKHiPPR visualization methods accept EquationSystem objects that contain an AbstractODE as well ---
    plot_eigenvalues(ode = equation_sys)

    branch: list[BranchPoint] = []

    # --- Iterate through the branch ---
    for branch_point in pseudo_arclength_continuator(
        initial_system=equation_sys,
        solver=solver,
        stepsize=0.01,
        stepsize_range=(0.001, 0.01),
        continuation_parameter="F",
        initial_direction=1,
        verbose=False,
        num_steps=400,
    ):
        branch.append(branch_point)
        # break if F exceeds maximum
        if branch_point.F > 0.5:
            break
       
    # --- Plot the continuation curve ---
    plot_continuation(
        branch,
        marker = "x"
        )

if __name__ == "__main__":
    main()
    plt.show()