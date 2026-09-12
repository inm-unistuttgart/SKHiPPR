import pytest
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from examples.vanderpol_minimal import Vanderpol
from skhippr.odes.nonautonomous import Duffing
from skhippr.cycles.shooting import ShootingBVP, ShootingSystem
from skhippr.solvers.newton import NewtonSolver

import warnings


@pytest.mark.parametrize("period_k", [1, 5])
def test_shooting_equation(solver, period_k, visualize=False):

    x_0 = np.array([1.0, 0.0])

    solve_ivp_kwargs = {"rtol": 1e-7, "atol": 1e-7}

    match period_k:
        case 1:
            ode = Duffing(t=0, x=x_0, alpha=1, beta=3, F=1, delta=1, omega=1.3)
        case 5:
            ode = Duffing(t=0, x=x_0, alpha=-1, beta=1, F=0.37, delta=0.3, omega=1.2)
        case _:
            raise ValueError(f"Unknown value '{period_k}' for period-k solution")

    T = 2 * np.pi / ode.omega

    # Converge to periodic solution by brute-force integration
    num_periods = 100
    sol_ivp_converge = solve_ivp(
        ode.dynamics, (0, num_periods * period_k * T), x_0, **solve_ivp_kwargs
    )
    x0_p = sol_ivp_converge.y[:, -1]

    # Ensure that solution didn't diverge
    assert all(
        np.abs(x0_p) < 100
    ), f"Trajectory from initial condition diverged: {x0_p}"

    # Find and plot converged solution by integrating over period_k periods
    points_per_period = 300
    t_eval = np.linspace(
        0, period_k * T, period_k * points_per_period + 1, endpoint=True
    )
    sol_ivp = solve_ivp(
        ode.dynamics, (0, period_k * T), x0_p, t_eval=t_eval, **solve_ivp_kwargs
    )
    assert np.allclose(
        x0_p, sol_ivp.y[:, -1], atol=1e-2, rtol=1e-2
    ), f"Brute-force integration did not converge to periodic solution with error {np.linalg.norm(x0_p - sol_ivp.y[:, -1])}"

    if visualize:
        plt.figure()
        plt.plot(
            sol_ivp.y[0, :],
            sol_ivp.y[1, :],
            label=f"sol after {num_periods} periods",
        )
        plt.plot(x0_p[0], x0_p[1], "+", label="Starting point")
        plt.title(f"Duffing period-{period_k} solution")

    # Find periodic solution using shooting
    duffing_shoot = ShootingBVP(ode=ode, T=T, period_k=period_k, **solve_ivp_kwargs)
    solver.solve_equation(duffing_shoot, "x")

    if visualize:
        print(duffing_shoot)

    assert np.allclose(
        duffing_shoot.x, x0_p, atol=1e-2, rtol=1e-2
    ), f"Shooting method did not converge to attractive periodic solution"
    # assert prb.stable, "Shooting: Wrong stability assertion"

    x_time = duffing_shoot.x_time(t_eval)
    assert np.allclose(
        duffing_shoot.x, x_time[:, -1], atol=1e-5, rtol=1e-5
    ), f"Shooting method did not converge to periodic solution"
    assert np.allclose(
        x_time, sol_ivp.y, atol=1e-5, rtol=1e-5
    ), f"Shooting method time series does not match converged solution"

    # Verify that solution is indeed a period-k solution
    for i in range(1, period_k):
        assert not np.allclose(
            x_time[:, i * points_per_period],
            x_time[:, 0],
            atol=1e-5,
            rtol=1e-5,
        ), f"Shooting solution is not period-{period_k} but period-{i}"

    if visualize:
        plt.plot(x_time[0, :], x_time[1, :], "r--", label="Shooting solution")
        plt.legend()


@pytest.mark.parametrize("autonomous", (False, True))
def test_shooting_system(solver, autonomous):

    x_0 = np.array([2.1, 0.0])
    if autonomous:
        ode = Vanderpol(t=0, x=x_0, nu=0.8)
        T = 2 * np.pi
    else:
        ode = Duffing(t=0, x=x_0, alpha=1, beta=3, F=1, delta=1, omega=1.3)
        T = 2 * np.pi / ode.omega

    shooting_system = ShootingSystem(ode=ode, T=T, period_k=1, atol=1e-7, rtol=1e-7)
    shooting_system.T = T  # needed to recover later during Duffing

    solver.solve(shooting_system)

    assert shooting_system.solved
    assert shooting_system.stable

    # Verify that the solution is indeed a periodic solution
    sol = solve_ivp(
        ode.dynamics,
        (0, np.squeeze(shooting_system.T)),
        shooting_system.x,
        atol=1e-7,
        rtol=1e-7,
    )
    assert np.allclose(sol.y[:, 0], sol.y[:, -1], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    solver = NewtonSolver(tolerance=1e-8, max_iterations=20, verbose=True)
    test_shooting_equation(solver=solver, period_k=5, visualize=True)
    plt.show()
