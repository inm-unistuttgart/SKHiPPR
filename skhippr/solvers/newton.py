import numpy as np
import matplotlib.pyplot as plt
from typing import override

from skhippr.equations.AbstractEquation import AbstractEquation
from skhippr.equations.EquationSystem import EquationSystem
from scipy.optimize import fsolve


class NewtonSolver:
    """
    Implements Newton's method for solving nonlinear equations and equation systems given by :py:class:`~skhippr.equations.EquationSystem.EquationSystem` or :py:class:`~skhippr.equations.AbstractEquation.AbstractEquation` objects.

    Attributes:
    -----------
    num_iter : int
        Number of iterations performed.
    converged : bool or None
        Indicates whether the solution procedure has converged.
    max_iterations : int
        Maximum number of allowed iterations.
    verbose : bool
        If True, prints progress information.
    tolerance : float
        Convergence tolerance for the residual norm.

    """

    def __init__(
        self,
        tolerance: float = 1e-8,
        max_iterations: int = 20,
        verbose: bool = False,
    ):

        self.converged = False
        self.num_iter: int = 0

        # Parameters
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.tolerance = tolerance

    # def __str__(self):
    #     if self.converged:
    #         text_converged = f"converged {self.label} solution"
    #     else:
    #         text_converged = f"non-converged {self.label} guess"
    #     if self.stable is None:
    #         text_stable = ""
    #     elif self.stable:
    #         text_stable = "(stable)"
    #     else:
    #         text_stable = "(unstable)"
    #     if len(self.x) < 5:
    #         text_x = f"at {self.variable} = {self.x} "
    #     else:
    #         text_x = ""
    #     return f"{text_converged} {text_x}after {self.num_iter}/{self.max_iterations} iterations {text_stable}"

    def reset(self) -> None:
        """
        Reset the state of the solver.
        """

        self.converged = False
        self.num_iter = 0

    def correction_step(self, equation_system) -> None:

        residual = self.check_converged(equation_system, update=True)
        if not equation_system.solved:
            delta_x = np.linalg.solve(equation_system.jacobian(update=True), -residual)
            equation_system.vector_of_unknowns = (
                equation_system.vector_of_unknowns + delta_x
            )

    def check_converged(self, equation_system, update=True) -> np.array:
        residual = equation_system.residual_function(update=update)
        if np.linalg.norm(residual) < self.tolerance:
            equation_system.solved = True
        return residual

    def solve_equation(
        self, equation: AbstractEquation, unknown: str
    ) -> EquationSystem:
        """Solve a single equation with a single unknown. The method creates an :py:class:`~skhippr.equations.EquationSystem.EquationSystem` from the given equation and unknown, solves it using :py:func:`~skhippr.newton.NewtonSolver.NewtonSolver.solve`, and checks for convergence.

        While the solved :py:class:`~skhippr.equations.EquationSystem.EquationSystem` is returned, the ``equation`` itself is also updated, storing the resulting system is optional.
        """
        system = EquationSystem([equation], [unknown], equation)
        self.solve(system)
        if not system.solved:
            raise RuntimeError(
                f"Could not solve {equation} within {self.num_iter} iterations."
            )
        return system

    def solve(self, equation_system: EquationSystem) -> None:
        """
        Applies Newton's method to solve the system of nonlinear equations given by a :py:class:`~skhippr.equations.EquationSystem.EquationSystem`.

        Performs iterative correction steps starting from the current vector of unknowns until the residual norm is sufficiently small or the maximum number of iterations is reached. In every step the unknown attributes of ``equation_system`` are updated. After convergence, performs a stability check if applicable.

        Prints progress and convergence information if :py:attr:`~skhippr.cycles.newton.NewtonProblem.verbose` is ``True``.
        """
        if not equation_system.well_posed:
            raise ValueError(
                "Equation system is not well-posed: Number of unknowns and number of equations differ"
            )

        self.reset()

        if self.verbose:
            print(f"Initial guess: x[-1]={equation_system.vector_of_unknowns[-1]:.3g}")

        while self.num_iter < self.max_iterations and not equation_system.solved:
            self.num_iter += 1
            if self.verbose:
                print(
                    f"Newton iteration {self.num_iter:2d}", end=""
                )  # , x = {equation_system.vector_of_unknowns}", end="")

            self.correction_step(equation_system)

            if self.verbose:
                print(
                    f", |r| = {np.linalg.norm(equation_system.residual_function(update=False)):8.3g}, x[-1]={equation_system.vector_of_unknowns[-1]:.3g}"
                )

            if equation_system.solved and self.verbose:
                print(f" Converged", end="")
                if equation_system.length_unknowns["total"] < 5:
                    print(f" to {equation_system.parse_vector_of_unknowns()}")
                else:
                    print("")

        if equation_system.solved:
            equation_system.determine_stability(update=True)
        elif self.verbose:
            print(f" Did not converge after {self.num_iter} iterations")


class ScipyFsolveSolver(NewtonSolver):
    def __init__(
        self, tolerance=1e-8, max_iterations=20, verbose=False, use_fprime=True
    ):
        super().__init__(tolerance, max_iterations, verbose)
        self.use_fprime = use_fprime

    def func(self, x, equation_system: EquationSystem):
        equation_system.vector_of_unknowns = x
        return equation_system.residual_function(update=True)

    def fprime(self, x, equation_system: EquationSystem):
        equation_system.vector_of_unknowns = x
        return equation_system.jacobian(update=True)

    @override
    def solve(self, equation_system: EquationSystem):
        if self.use_fprime:
            fprime = self.fprime
        else:
            fprime = None

        x, infodict, flag, msg = fsolve(
            func=self.func,
            x0=equation_system.vector_of_unknowns,
            args=equation_system,
            fprime=fprime,
            full_output=True,
            xtol=self.tolerance,
            maxfev=self.max_iterations,
        )

        self.num_iter = infodict.get("nfev")

        if flag == 1:
            self.converged = True
            equation_system.vector_of_unknowns = x
            equation_system.solved = True
            equation_system.determine_stability(update=True)

        if self.verbose:
            if equation_system.solved:
                print(
                    f"fsolve converged successfully with {infodict.get('nfev', 0)} function calls and {infodict.get('njev', 0)} jacobian calls."
                )
            else:
                print(f"fsolve did not converge! \n {flag}: {msg}")
