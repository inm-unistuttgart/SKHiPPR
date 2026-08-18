import matplotlib

# Use the non-interactive Agg backend for the whole test session: tests that create
# figures must not depend on a working GUI toolkit (e.g. Tk) being installed, and must
# not pop up windows or block waiting for a display.
matplotlib.use("Agg")

import pytest
import numpy as np

from skhippr.Fourier import Fourier
from skhippr.odes.autonomous import BlockOnBelt
from skhippr.odes.nonautonomous import Duffing
from skhippr.odes.ltp import (
    HillLTI,
    MathieuODE,
    SmoothedMeissner,
    TruncatedMeissner,
    ShirleyODE,
)
from skhippr.solvers.newton import NewtonSolver


@pytest.fixture(scope="session", params=[1, 100])
def ode_setting_vectorized(request, ode_setting):
    n_samples = request.param
    if n_samples == 1:
        x = np.random.rand(2)
    else:
        x = np.random.rand(2, n_samples)

    if len(x.shape) == 1:
        t = 1
    else:
        t = np.linspace(0, 2 * np.pi, x.shape[1])

    params, ode = ode_setting
    ode.x = x
    ode.t = t

    return params, ode


@pytest.fixture(scope="session", params=[1, 5])
def duffing_ode(request):
    period_k = request.param
    x_0 = np.array([1.0, 0.0])

    match period_k:
        case 1:
            ode = Duffing(t=0, x=x_0, alpha=1, beta=3, F=1, delta=1, omega=1.3)
        case 5:
            ode = Duffing(t=0, x=x_0, alpha=-1, beta=1, F=0.37, delta=0.3, omega=1.2)
        case _:
            raise ValueError(f"Unknown value '{period_k}' for period-k solution")
    return period_k, ode


@pytest.fixture(
    scope="session",
    params=[
        "Blockonbelt",
        "DuffingPeriod1",
        "HillConst",
        "Meissner",
        "SmoothedMeissner",
        "TruncMeissner",
        "Mathieu",
        "Shirley",
    ],
)
def ode_setting(request):
    t = 1
    x = np.random.rand(2)

    match request.param:
        case "Blockonbelt":
            params = {"epsilon": 0.1, "k": 1, "m": 1, "Fs": 0.1, "vdr": 2, "delta": 0.5}
            return params, BlockOnBelt(x=x, **params)
        case "DuffingPeriod1":
            params = {
                "alpha": 1,
                "beta": 3,
                "F": 1,
                "delta": 0.1,
                "omega": 1.3,
            }
            return params, Duffing(t=t, x=x, **params)
        case "HillConst":
            params = {"a": 0.2, "b": 1, "damping": 0.02, "omega": 1}
            return params, HillLTI(t, x, **params)
        case "Meissner":
            params = {"a": 0.2, "b": 1, "damping": 0.02, "omega": 1}
            return params, SmoothedMeissner(t, x, smoothing=0, **params)
        case "SmoothedMeissner":
            params = {"a": 0.2, "b": 1, "damping": 0.02, "omega": 1, "smoothing": 0.3}
            return params, SmoothedMeissner(t, x, **params)
        case "TruncMeissner":
            params = {"a": 0.2, "b": 1, "damping": 0.02, "omega": 1, "N_harmonics": 5}
            return params, TruncatedMeissner(t, x, **params)
        case "Mathieu":
            params = {"a": 0.2, "b": 1, "damping": 0.02, "omega": 1}
            return params, MathieuODE(t, x, **params)
        case "Shirley":
            params = {"E_alpha": 0.7, "E_beta": 1.5, "b": 2, "omega": 1}
            return params, ShirleyODE(t, x, **params)


@pytest.fixture
def solver():
    return NewtonSolver(tolerance=1e-8, max_iterations=20, verbose=True)


@pytest.fixture(
    scope="module",
    params=[
        {"real_formulation": True},
        {"real_formulation": False},
    ],
)
def fourier_small(request):
    # warnings.filterwarnings(action="error", category=np.exceptions.ComplexWarning)
    n_dof = 2
    N_HBM = 3
    L_DFT = 16

    return Fourier(
        N_HBM=N_HBM,
        L_DFT=L_DFT,
        n_dof=n_dof,
        real_formulation=request.param["real_formulation"],
    )


@pytest.fixture(
    scope="module",
    params=[
        {"real_formulation": True},
        {"real_formulation": False},
    ],
)
def fourier(request):
    # warnings.filterwarnings(action="error", category=np.exceptions.ComplexWarning)
    n_dof = 2
    N_HBM = 50
    L_DFT = 400

    return Fourier(
        N_HBM=N_HBM,
        L_DFT=L_DFT,
        n_dof=n_dof,
        real_formulation=request.param["real_formulation"],
    )
