"""Test HBM using the Duffing oscillator and the Vanderpol oscillator. See also: Matlab workshop, Task x.x"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from skhippr.cycles.hbm import HBMEquation, HBMSystem
from skhippr.cycles.shooting import ShootingBVP, ShootingSystem
from skhippr.Fourier import Fourier
from skhippr.odes.nonautonomous import Duffing
from examples.vanderpol_minimal import Vanderpol
from skhippr.solvers.newton import NewtonSolver


@pytest.mark.parametrize("period_k", [1, 5])
def test_HBM_equation(solver, duffing_ode, fourier, period_k, visualize=False):

    period_k, ode = duffing_ode
    if visualize:
        print("Duffing oscillator")

    T = 2 * np.pi / ode.omega

    # Reference: Find periodic solution using shooting method
    shooting = ShootingBVP(ode, T=T, period_k=period_k, atol=1e-7, rtol=1e-7)

    solver.solve_equation(shooting, "x")

    """ Compute periodic solution using HBM """
    t_eval = fourier.time_samples(ode.omega, periods=period_k)[::period_k]
    x_shoot = shooting.x_time(t_eval=t_eval)
    X0 = fourier.DFT(x_shoot + np.random.rand(*x_shoot.shape) * 1e-1)

    hbm = HBMEquation(
        ode=ode,
        omega=ode.omega,
        fourier=fourier,
        initial_guess=X0,
        period_k=period_k,
        stability_method=None,
    )

    if visualize:
        print(hbm)

    solver.solve_equation(hbm, "X")

    if visualize:
        print(hbm)

    # Assert that the HBM solution is indeed a solution
    x_hbm = hbm.x_time()
    if visualize:
        plt.figure()
        plt.plot(x_shoot[0, :], x_shoot[1, :], label="ODE")
        plt.plot(np.real(x_hbm[0, :]), np.real(x_hbm[1, :]), "--", label="HBM")
        plt.legend()
        plt.title(
            f"{'real' if fourier.real_formulation else 'complex'} Duffing \n {hbm}"
        )

    assert np.allclose(x_hbm, x_shoot, atol=1e-1)


@pytest.mark.parametrize("initial_error", (0, 0.05))
@pytest.mark.parametrize("autonomous", [False, True])
def test_HBMSystem(solver, fourier, autonomous, initial_error, visualize=False):

    x_0 = np.array([2.1, 0.0])
    if autonomous:
        ode = Vanderpol(t=0, x=x_0, nu=0.8)
        T = 2 * np.pi
    else:
        ode = Duffing(t=0, x=x_0, alpha=1, beta=3, F=1, delta=1, omega=1.3)
        T = 2 * np.pi / ode.omega

    # Shooting system for reference
    shooting_system = ShootingSystem(ode=ode, T=T, period_k=1, atol=1e-7, rtol=1e-7)
    shooting_system.T = T
    solver.solve(shooting_system)
    assert shooting_system.solved

    omega = 2 * np.pi / shooting_system.T

    t_eval = fourier.time_samples(omega, periods=1)
    x_shoot = shooting_system.equations[0].x_time(t_eval=t_eval)
    X0 = fourier.DFT(x_shoot)
    X0 += initial_error * np.random.rand(*X0.shape)

    hbm_system = HBMSystem(
        ode=ode,
        omega=omega,
        fourier=fourier,
        initial_guess=X0,
        stability_method=None,
    )

    solver.solve(hbm_system)
    assert hbm_system.solved

    if visualize:
        print(np.max(np.abs(hbm_system.X)))
        plt.figure()
        x_time = hbm_system.equations[0].x_time()
        plt.plot(x_time[0, :], x_time[1, :], label="hbm")
        plt.plot(x_shoot[0, :], x_shoot[1, :], "--", label="shooting")
    # assert np.allclose(hbm_system.equations[0].x_time(), x_shoot, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    my_solver = NewtonSolver(verbose=True)
    my_fourier = Fourier(N_HBM=15, L_DFT=128, n_dof=2, real_formulation=True)
    for period_k in [1, 5]:
        test_HBM_equation(
            my_solver, None, fourier=my_fourier, period_k=period_k, visualize=True
        )
    test_HBMSystem(
        my_solver, my_fourier, autonomous=True, visualize=True, initial_error=0.05
    )
    plt.show()
