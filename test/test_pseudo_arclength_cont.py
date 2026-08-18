import pytest
import numpy as np
import matplotlib.pyplot as plt

from skhippr.solvers.newton import NewtonSolver, EquationSystem
from skhippr.solvers.continuation import pseudo_arclength_continuator
from examples.truss import Truss
from skhippr.equations.AbstractEquation import AbstractEquation
from skhippr.equations.Circle import CircleEquation


@pytest.fixture
def truss_params():
    params = {
        "a": 1,
        "l_0": 1.2,
        "m": 1,
        "k": 3,
        "c": 0.01,
        "F": -2,
    }
    return params


def test_cont_circle(solver, visualize=False):
    radius = 1.5
    y0 = [0.9 * radius, 0]
    initial_equation = CircleEquation(y=y0, radius=radius)
    initial_system = EquationSystem([initial_equation], ["y"], None)

    branch = []
    num_steps = 1000
    for radius in [radius, 2.0, 3.0]:
        initial_equation.radius = radius
        for k, branch_point in enumerate(
            pseudo_arclength_continuator(
                initial_system=initial_system,
                solver=solver,
                stepsize=0.05,
                stepsize_range=(0.01, 0.1),
                initial_direction=1,
                continuation_parameter=None,
                verbose=True,
                num_steps=num_steps,
            )
        ):
            branch.append(branch_point)
            assert np.allclose(np.linalg.norm(branch_point.y), radius)

            if k > 0:
                actual_stepsize = np.linalg.norm(
                    branch_point.vector_of_unknowns - branch[-2].vector_of_unknowns
                )
                assert actual_stepsize > 0.008

                # End condition
                if branch[-2].y[-1] < 0 and branch_point.y[-1] >= 0:
                    # continue with different radius
                    break

    # Check number of steps
    # assert (
    #     len(branch) == num_steps + 1
    # ), f"Expected {num_steps+1} branch points (including initial guess), but got {len(branch)}"

    if visualize:
        plt.figure()
        plt.plot(*np.array([branch_point.y for branch_point in branch]).T)
        for k, branch_point in enumerate(branch):
            try:
                point_with_tng = np.vstack(
                    (branch_point.y, branch_point.y + 0.2 * branch_point.tangent)
                )
                plt.plot(point_with_tng[:, 0], point_with_tng[:, 1], color="g")
            except TypeError:
                # last point on each branch nas no tangent yet
                print(f"Branch point # {k}/{len(branch)} has no tangent")


def test_cont_truss(solver: NewtonSolver, truss_params, verbose=False):

    x0 = np.array([-1.0, 0.0])
    ode = Truss(x=x0, **truss_params)
    ode.mutable_attribute = [1, 2, 3]  # needed later for testing the copying

    truss_system = EquationSystem(
        equations=[ode], unknowns=["x"], equation_determining_stability=ode
    )

    solver.solve(truss_system)
    assert truss_system.solved
    if verbose:
        print(truss_system.vector_of_unknowns)

    interval_F = [-2, 2]

    branch = []
    xs = []
    Fs = []
    stable = []

    if verbose:
        plt.figure()

    for branch_point in pseudo_arclength_continuator(
        initial_system=truss_system,
        solver=solver,
        stepsize=0.03,
        stepsize_range=(0.001, 0.2),
        initial_direction=1,
        continuation_parameter="F",
        verbose=verbose,
        num_steps=300,
    ):
        branch.append(branch_point)
        xs.append(branch_point.x)
        Fs.append(branch_point.F)
        stable.append(branch_point.stable)

        branch_point.determine_tangent()
        x1_tng = branch_point.x[0] + 0.2 * branch_point.tangent[0]
        F_tng = branch_point.F + 0.2 * branch_point.tangent[-1]
        if verbose:
            plt.plot((branch_point.F, F_tng), (branch_point.x[0], x1_tng), "gray")

        if branch_point.F < interval_F[0] or branch_point.F > interval_F[1]:
            # if branch_point.x[0] > 0:
            break
    xs = np.array(xs).T
    Fs = np.array(Fs)
    stable = np.array(stable)
    unstable = np.invert(stable)

    F_stbl = Fs.copy()
    F_stbl[unstable] = np.nan
    F_ustbl = Fs.copy()
    F_ustbl[stable] = np.nan

    if verbose:
        plt.plot(F_stbl, xs[0, :], "rx", label="stable")
        plt.plot(F_ustbl, xs[0, :], "b+", label="unstable")
        plt.title("Pseudo-arclength results")
        plt.xlabel("F")
        plt.ylabel("x[0]")
        plt.legend()

    """Verify stability assertions are correct at start, middle, end"""
    assert stable[0]
    assert stable[-1]
    idx_x_0 = np.argmin(np.abs(xs[0, :]))
    assert not stable[idx_x_0]

    """Verify that the stability changes are saddle nodes: F turns around near a stab. change"""
    idx_stabchange = [
        k for k in range(2, len(stable) - 1) if stable[k] != stable[k - 1]
    ]
    # assert len(idx_stabchange) == 2
    for k in idx_stabchange:
        assert (Fs[k - 1] - Fs[k - 2]) * (Fs[k + 1] - Fs[k]) < 0
        pass

    """ Ensure that the individual branch points are decoupled shallow copies"""
    # Consider an immutable parameter ('a') that is NOT part of the unknowns.
    # 1. a is not well defined outside equations[0]
    with pytest.raises(AttributeError):
        a = branch[-1].a
    with pytest.raises(AttributeError):
        a = branch[-1].equations[1].a
    val_a = branch[-1].equations[0].a
    assert val_a == truss_params["a"]

    # 2. if a is changed within equations[0], this has no effect on the branch point and all other equations
    branch[-1].equations[0].a = 200
    with pytest.raises(AttributeError):
        a = branch[-1].a
    with pytest.raises(AttributeError):
        a = branch[-1].equations[1].a
    assert branch[-2].equations[0].a != 200

    # 3. If a is set in the branch point, it is NOT set in all its equations because it is not part of the unknowns.
    branch[-1].a = 100
    assert branch[-1].equations[0].a != 100
    with pytest.raises(AttributeError):
        a = branch[-1].equations[1].a
    with pytest.raises(AttributeError):
        a = branch[-2].a

    # Consider a parameter ('F') that IS part of the unknowns
    # 4. Check that F is consistent (same as in equations but changes throughout branch)
    F = branch[-1].F
    assert F == branch[-1].equations[0].F
    assert F == branch[-1].equations[1].F
    assert F != branch[-2].F
    assert F != branch[-2].equations[0].F
    assert F != branch[-2].equations[1].F

    # 5. Setting F directly in the equation should not modify the branch point
    F_next = 5 * F
    branch[-1].equations[0].F = F_next
    assert F_next == branch[-1].equations[0].F
    assert F_next != branch[-1].equations[1].F
    assert F_next != branch[-1].F
    assert F_next != branch[-2].F
    assert F_next != branch[-2].equations[0].F

    # 6. Setting F in the branch point should modify all its equations, but not other branch points
    F_next = 10 * F
    branch[-1].F = F_next
    assert F_next == branch[-1].F
    assert F_next == branch[-1].equations[0].F
    assert F_next == branch[-1].equations[1].F
    assert F_next != branch[-2].F
    assert F_next != branch[-2].equations[0].F
    assert F_next != branch[-2].equations[1].F

    # Consider a mutable attribute to check that the equations are indeed shallow-copied, not deep-copied
    val_attr = 100
    # mutate the attribute, not overwrite it
    branch[-1].equations[0].mutable_attribute[0] = val_attr
    assert branch[-2].equations[0].mutable_attribute[0] == val_attr


if __name__ == "__main__":
    params = {"a": 1, "l_0": 1.2, "m": 1, "k": 3, "c": 0.01, "F": -2}
    my_solver = NewtonSolver(
        tolerance=1e-8,
        verbose=False,
        max_iterations=20,
    )
    test_cont_truss(my_solver, params, verbose=True)
    # test_cont_circle(my_solver, visualize=True)
    plt.show()
