"""

The :py:module:`skhippr.solvers.continuation` module offers the functionality to perform pseudo-arclength continuation on equation systems encoded by :py:class:`~skhippr.equations.EquationSystem.EquationSystem` objects.

It provides the generator function :py:func:`~skhippr.solvers.continuation.pseudo_arclength_continuator` for continuation. This generator returns in each iteration a :py:class:`~skhippr.solvers.continuation.BranchPoint` object, which is a subclass of :py:class:`~skhippr.equations.EquationSystem.EquationSystem` and includes all equations of the initial system, along  with an additional :py:class:`~skhippr.solvers.continuation.ContinuationAnchor` equation that ensures all updates are orthogonal to the tangent.
"""

from collections.abc import Iterable, Iterator
import numpy as np

from skhippr.solvers.newton import NewtonSolver, EquationSystem
from skhippr.equations.AbstractEquation import AbstractEquation


def pseudo_arclength_continuator(
    initial_system: EquationSystem,
    solver: NewtonSolver,
    stepsize: float = None,
    stepsize_range: Iterable[float] = (1e-4, 1e-1),
    initial_direction=1,
    continuation_parameter=None,
    verbose=False,
    num_steps: int = 1000,
) -> Iterator["BranchPoint"]:
    """
    Perform pseudo-arclength continuation of the solution branch emerging from a nonlinear :py:class:`~skhippr.equations.EquationSystem.EquationSystem`.

    This generator yields a sequence of :py:class:`~skhippr.solvers.continuation.BranchPoint` objects, each representing an individual solution of the problem, by following the solution branch using the pseudo-arclength continuation method.

    Parameters
    ----------

    initial_problem : :py:class:`~skhippr.equations.EquationSystem.EquationSystem`
        The initial :py:class:`~skhippr.equations.EquationSystem.EquationSystem` to start continuation from. It need not be solved already. All branch points will duplicte the equations of this system,

        * Explicit case: If the :py:class:`~skhippr.equations.EquationSystem.EquationSystem` is ``well_posed``, an explicit ``continuation_parameter``, which must be an attribute of at least one equation, is required.
        * Implicit case: Otherwise, the problem must have exactly one equation more than unknowns.

    stepsize : float, optional
        Initial step size for continuation. If None, uses the lower bound of ``stepsize_range``.
    stepsize_range : Iterable of float, optional
        Tuple specifying the minimum and maximum allowed step sizes. Default is (1e-4, 1e-1).
    initial_direction : int, optional
        Direction (+1 or -1) to start continuation along the branch. If positive, the continuation parameter (explicit case) or the last entry of the unknowns (implicit case) increases initially.
    key_param : str, optional
        Name of the parameter to track during continuation.
    value_param : Any, optional
        Value of the parameter at the initial point. Defaults to ``initial_problem.<key_param>``.
    verbose : bool, optional
        If ``True``, prints progress and diagnostic information. Default is ``False``.
    num_steps : int, optional
        Maximum number of continuation steps to perform. Default is 1000.

    Yields
    ------

    :py:class:`~skhippr.solvers.continuation.BranchPoint`
        The next converged solution point along the continuation branch.

    Raises
    ------

    RuntimeError
        If the initial problem does not converge.

    Notes
    -----
    * If the minimal stepsize does not converge, the continuation stops.
    * The continuator does not check whether the branch point lies within any value bounds. Such checks should be made within the ``for`` loop over which the continuator is iterated.
    """
    if stepsize is None:
        stepsize = stepsize_range[0]

    initial_point = BranchPoint(
        underlying_system=initial_system,
        continuation_parameter=continuation_parameter,
        initial_direction=initial_direction,
    )

    solver.solve(initial_point)
    if not initial_point.solved:
        raise RuntimeError("Initial guess did not converge!")
    elif verbose:
        print(f"Initial guess converged after {solver.num_iter} steps.")

    yield initial_point
    last_point = initial_point
    stop_msg = "All steps finished successfully."

    k = 0
    while k < num_steps:
        if verbose:
            print(f"Continuation step {k}, step size {stepsize:.3f}:", end=" ")

        next_point = last_point.predict(stepsize=stepsize)

        if verbose and continuation_parameter is not None:
            print(
                f"{continuation_parameter}_pred = {np.squeeze(getattr(next_point, continuation_parameter)):.2f}",
                end=" ",
            )

        solver.solve(next_point)

        if next_point.solved:
            if verbose:
                print(f"success after {solver.num_iter} steps.")
            yield next_point
            k += 1

            last_point = next_point
            stepsize = min(1.2 * stepsize, stepsize_range[1])
            if continuation_parameter is not None:
                stepsize = min(
                    stepsize,
                    np.squeeze(
                        getattr(next_point, continuation_parameter) * stepsize_range[1]
                    ),
                )

        elif stepsize > stepsize_range[0]:
            if verbose:
                print("no success. Retry.")
            stepsize = max(0.5 * stepsize, stepsize_range[0])

        else:
            stop_msg = "Minimal step size did not converge."
            break

    if verbose:
        print(stop_msg)


class BranchPoint(EquationSystem):
    """
    A :py:class:`~skhippr.solvers.continuation.BranchPoint` represents a point on an implicit or explicit continuation branch.

    An object of this class contains all equations of the underlying :py:class:`~skhippr.equations.EquationSystem.EquationSystem`, and additionally one :py:class:`~skhippr.solvers.continuation.ContinuationAnchor` as the  last equation. If a continuation parameter is passed, the unknowns are extended by this continuation parameter.
    """

    def __init__(
        self,
        underlying_system: EquationSystem,
        continuation_parameter: str = None,
        initial_direction=1,
    ):

        anchor_equation = ContinuationAnchor(
            underlying_system,
            continuation_parameter,
            initial_direction=initial_direction,
        )

        unknowns = underlying_system.unknowns
        if continuation_parameter is not None:
            unknowns = unknowns + [continuation_parameter]

        super().__init__(
            equations=underlying_system.equations + [anchor_equation],
            unknowns=unknowns,
            equation_determining_stability=underlying_system.equation_determining_stability,
        )
        self.tangent = None

    def determine_tangent(self, update=False, update_tol=1e-10):
        """
        Compute, normalize and store the tangent vector at the branch point.

        This method checks if the :py:class:`~skhippr.solvers.continuation.BranchPoint` equation system is solved. If not, it raises a ``RuntimeError``.
        It then constructs the tangent vector by solving a linear system. with the same Jacobian matrix as for the Newton updates. The resulting normalized tangent vector forms an acute angle with the previous tangent vector. It is stored in the ``self.tangent`` attribute.

        Raises
        -------

        RuntimeError
            If the solution has not converged and the tangent would thus be meaningless.

        """
        if (
            not self.solved
            and np.linalg.norm(self.residual_function(update=update)) > update_tol
        ):
            raise RuntimeError(
                "Cannot determine tangent at branch point: Branch point not reached (system not solved)."
            )

        jac = self.jacobian(update=update)
        rhs = np.zeros(jac.shape[0])
        rhs[-1] = 1
        tangent = np.linalg.solve(jac, rhs)
        self.tangent = tangent / np.linalg.norm(tangent)
        # adding the tangent does not remove solution integrity
        self.solved = True
        return self.tangent

    def predict(self, stepsize: float) -> "BranchPoint":
        """
        Predict the next branch point in the continuation process.

        Parameters
        ----------

        stepsize : float
            The step size to advance along the tangent direction.

        Returns
        -------

        BranchPoint
            A new :py:class:`~skhippr.solvers.continuation.BranchPoint` instance (not converged) representing the predicted next point along the continuation path. all equations are shallow-copied to prevent that updates in the next point affect the current point.

        """
        if self.tangent is None:
            self.determine_tangent()

        branch_point_next = self.duplicate()
        branch_point_next.equations[-1].initialize_anchor(previous_tangent=self.tangent)
        branch_point_next.vector_of_unknowns = (
            self.vector_of_unknowns + stepsize * self.tangent
        )
        branch_point_next.tangent = None

        return branch_point_next


class ContinuationAnchor(AbstractEquation):
    """A scalar :py:class:`~skhippr.equations.AbstractEquation.AbstractEquation` subclass that ensures orthogonality to the tangent direction in the continuation process. While the residual function is always zero, the closed-form derivative is the tangent vector at the previous branch point, ensuring that all Newton updates of the branch point are orthogonal to the tangent.

    During instantiation, either a previous tangent vector or an initial direction must be provided.
    """

    def __init__(
        self,
        equation_system,
        continuation_parameter=None,
        previous_tangent=None,
        initial_direction=1,
    ):
        super().__init__(None)
        self.equation_system = equation_system
        self.continuation_parameter = continuation_parameter

        if previous_tangent is not None:
            self.initialize_anchor(previous_tangent)
        elif initial_direction is not None:
            self.initialize_anchor_with_direction(initial_direction)
        # otherwise (during prediction), the initialization must be called manually

    def initialize_anchor(self, previous_tangent):
        if self.continuation_parameter is None:
            self.anchor = self.equation_system.parse_vector_of_unknowns(
                previous_tangent
            )
        else:
            self.anchor = self.equation_system.parse_vector_of_unknowns(
                previous_tangent[:-1].flatten()
            )
            self.anchor[self.continuation_parameter] = np.atleast_1d(
                previous_tangent[-1]
            )

    def initialize_anchor_with_direction(self, initial_direction):

        anchor = np.zeros(self.equation_system.length_unknowns["total"])
        if self.continuation_parameter is None:
            anchor[-1] = initial_direction
        else:
            anchor = np.append(anchor, initial_direction)

        self.initialize_anchor(anchor)

    def residual_function(self):
        """Always returns zero."""
        return np.atleast_1d(0)

    def closed_form_derivative(self, variable):
        """Returns the segment of the tangent vector at the previous branch point corresponding to the unknown ``variable``."""
        try:
            return self.anchor[variable][np.newaxis, :]
        except KeyError:
            raise NotImplementedError
