import numpy as np
import pytest
from skhippr.odes.nonautonomous import Duffing
from examples.vanderpol_minimal import Vanderpol
from examples.truss import Truss
from skhippr.equations.EquationSystem import EquationSystem


def test_solution(solver, ode_setting):
    # All considered test cases except Duffing have an equilibrium at zero
    _, ode = ode_setting

    if isinstance(ode, Duffing):
        ode.F = 0  # remove forcing

    equ_sys = EquationSystem(
        equations=[ode],
        unknowns=["x"],
        equation_determining_stability=None,
    )

    solver.solve(equ_sys)
    assert equ_sys.solved


def test_stability(solver, ode_setting):
    _, ode = ode_setting
    equ_sys = EquationSystem(
        equations=[ode],
        unknowns=["x"],
        equation_determining_stability=ode,
    )
    solver.solve(equ_sys)

    print(ode.stable)

    if isinstance(ode, Vanderpol):
        assert not ode.stable

    if isinstance(ode, Truss):
        if np.max(np.abs(ode.x)) > 0.1:
            assert ode.stable
        else:
            assert not ode.stable
