import pytest
import numpy as np
import matplotlib.pyplot as plt

from examples.vanderpol_minimal import Vanderpol
from skhippr.cycles.hbm import HBMSystem
from skhippr.stability.SinglePass import SinglePassRK4


def test_ode_sampling(solver, fourier):

    ode = Vanderpol(x=np.array([2.0, 0.0]), nu=0.2)
    omega0 = 1
    ts_samp = fourier.time_samples(omega=omega0)

    x0_samples = np.vstack(
        (2 * np.cos(omega0 * ts_samp), -2 * omega0 * np.sin(omega0 * ts_samp))
    )
    X0 = fourier.DFT(x0_samples)

    hbm = HBMSystem(ode, omega0, fourier, X0)
    solver.verbose = True
    solver.solve(hbm)

    x_samp = hbm.equations[0].x_time()
    assert np.max(np.abs(x_samp)) > 1e-1

    J_ref = ode.derivative(variable="x", t=ts_samp, x=x_samp)
    J_samp = hbm.equations[0].ode_samples()
    assert np.allclose(J_ref, J_samp, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
