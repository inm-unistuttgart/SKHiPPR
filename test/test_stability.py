"""Compare the stability assertions of all stability methods"""

import pytest
import numpy as np
import itertools
import matplotlib.pyplot as plt

from skhippr.Fourier import Fourier

from skhippr.cycles.hbm import HBMSystem
from skhippr.cycles.shooting import ShootingSystem

from skhippr.odes.nonautonomous import Duffing
from examples.vanderpol_minimal import Vanderpol

from skhippr.stability.KoopmanHillProjection import (
    KoopmanHillProjection,
    KoopmanHillSubharmonic,
)
from skhippr.stability.ClassicalHill import ClassicalHill
from skhippr.stability.SinglePass import SinglePassRK4, SinglePassRK38


@pytest.mark.parametrize("autonomous", [False, True])
@pytest.mark.parametrize(
    "stability_method",
    [
        KoopmanHillProjection,
        KoopmanHillSubharmonic,
        lambda fourier: ClassicalHill(fourier, sorting_method="imaginary"),
        SinglePassRK4,
        lambda fourier: SinglePassRK38(fourier, stepsize=1e-3),
    ],
)
def test_stability(solver, autonomous, stability_method, fourier, visualize=False):

    if autonomous:
        ode = Vanderpol(t=0, x=np.array([2.0, 0]), nu=0.05)
        omega = 1
    else:
        ode = Duffing(
            t=0,
            x=np.array([1.0, 0.0]),
            omega=0.8,
            alpha=1,
            beta=0.2,
            delta=0.1,
            F=3,
        )
        omega = ode.omega

    ode_kwargs = {"rtol": 1e-7, "atol": 1e-7}

    # Reference: Floquet multipliers obtained using Shooting
    sys_shooting = ShootingSystem(ode, T=2 * np.pi / omega, rtol=1e-7, atol=1e-7)

    solver.solve(sys_shooting)
    assert sys_shooting.solved
    omega = sys_shooting.equations[0].omega
    floquet_multipliers_shooting = sys_shooting.eigenvalues + 1

    # determine periodic solution using HBM
    method = stability_method(fourier)
    x_shooting = sys_shooting.equations[0].x_time(fourier.time_samples(omega))
    X0 = fourier.DFT(x_shooting)

    sys_HBM = HBMSystem(
        ode,
        omega,
        fourier=fourier,
        initial_guess=X0,
        stability_method=stability_method(fourier),
    )
    solver.solve(sys_HBM)
    assert sys_HBM.solved
    floquet_multipliers = sys_HBM.eigenvalues

    # Account for the fact that the arrays of FMs might be sorted differently
    for k, idx in enumerate(itertools.permutations(range(len(floquet_multipliers)))):
        fm_reordered = floquet_multipliers[idx,]
        fm_equal = np.allclose(floquet_multipliers_shooting, fm_reordered, atol=1e-5)
        if fm_equal:
            break
    assert fm_equal

    if visualize:
        print(method)
        print(floquet_multipliers_shooting)
        print(floquet_multipliers)
        if fm_equal:
            print(f"Same as shooting method (in {k}-th permutation).")
        else:
            print("Different from Shooting method!")

        plt.figure()
        phi = np.linspace(0, 2 * np.pi, 250)
        plt.plot(np.cos(phi), np.sin(phi), "k")

        plt.plot(
            np.real(floquet_multipliers_shooting),
            np.imag(floquet_multipliers_shooting),
            "x",
            label="shooting",
        )
        plt.title(str(method))
        plt.axis("equal")

        plt.plot(
            np.real(floquet_multipliers),
            np.imag(floquet_multipliers),
            ".",
            label=method.label,
        )

        plt.legend(loc="upper right")


if __name__ == "__main__":
    pytest.main([__file__])
