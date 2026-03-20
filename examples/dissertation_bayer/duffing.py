import numpy as np
import matplotlib.pyplot as plt

# --- Differential equation ---
from skhippr.odes.nonautonomous import Duffing

# --- Stability method ---
from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic

# --- Continuation ---
from skhippr.solvers.continuation import pseudo_arclength_continuator

# --- Newton solver ---
from skhippr.solvers.newton import NewtonSolver

# --- HBM equation system ---
from skhippr.cycles.hbm import HBMEquation
from skhippr.equations.EquationSystem import EquationSystem

# --- Fourier configuration ---
from skhippr.Fourier import Fourier

from skhippr.visualization.continuation import plot_continuation

from skhippr.visualization.data_export import save_tikz

# --- FFT, stability method and Newton solver configuration ---
fourier = Fourier(N_HBM=25, L_DFT=300, n_dof=2, real_formulation=True)
stability_method = KoopmanHillSubharmonic(fourier, tol=1e-4, autonomous=False)
solver = NewtonSolver(verbose=False)

# --- Instantiation of the ODE at initial point of branch ---
ode = Duffing(t=0, x=[1.0, 0.0], alpha=1, beta=0, delta=0.16, F=0.5, omega=0.02)

# --- Initial guess in time and frequency domain ---
ts = fourier.time_samples(ode.omega)
x0_samples = np.array([np.cos(ode.omega * ts), -ode.omega * np.sin(ode.omega * ts)])
X0 = fourier.DFT(x0_samples)

# --- Set up the Harmonic Balance equations
hbm = HBMEquation(
    ode=ode,
    omega=ode.omega,
    fourier=fourier,
    initial_guess=X0,
    period_k=1,
    stability_method=stability_method,
)
initial_system = EquationSystem(
    equations=[hbm], unknowns=["X"], equation_determining_stability=hbm
)


frc = []
for branch_point in pseudo_arclength_continuator(
    initial_system=initial_system,
    solver=solver,
    stepsize=0.1,
    stepsize_range=(0.001, 0.1),
    continuation_parameter="omega",
    initial_direction=1,
    verbose=True,
    num_steps=4000,
):
    frc.append(branch_point)

    # break if omega exceeds maximum
    if branch_point.omega > 2.5:
        break

ax = plot_continuation(
    frc, plot_fun=lambda point: np.max(point.equations[0].x_time()[0, :])
)

save_tikz(
    ax,
    f"duffing_frc_alpha_{ode.alpha}_beta_{ode.beta}_delta_{ode.delta}_F_{ode.F}.tikz",
)
plt.show()
