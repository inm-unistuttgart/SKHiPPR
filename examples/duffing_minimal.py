"""Minimal example  / template for creating a simple frequency response curve of a non-autonomous dynamical system with HBM.
This example uses a Duffing oscillator, but ``ode`` can be instantiated as any non-autonomous :py:class:`~skhippr.odes.AbstractODE.AbstractODE`.
"""

from typing import override
import numpy as np
import matplotlib.pyplot as plt

# --- Fourier configuration ---
from skhippr.Fourier import Fourier

# --- Differential equation ---
from skhippr.odes.AbstractODE import AbstractODE

# --- HBM equation system ---
from skhippr.cycles.hbm import HBMEquation
from skhippr.equations.EquationSystem import EquationSystem

# --- Stability method ---
from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic

# --- Continuation ---
from skhippr.solvers.continuation import pseudo_arclength_continuator, BranchPoint

# --- Newton solver ---
from skhippr.solvers.newton import NewtonSolver

# quick visualization
from skhippr.visualization.continuation import plot_continuation


# --- Non-autonomous Duffing oscillator âs an AbstractODE subclass, encodes the dynamics.
# --- Subclasses of AbstractODE must implement the dynamics() method, which returns the right-hand side of the ODE, and optionally closed_form_derivative(), which returns the Jacobian of the right-hand side w.r.t. a variable (x, omega, F, alpha, beta, delta).
class Duffing(AbstractODE):
    """
    Non-autonomous Duffing oscillator as concrete subclass of :py:class:`~skhippr.odes.AbstractODE.AbstractODE`. ::

        dx[0]/dt = x[1]
        dx[1]/dt = -alpha * x[0] - delta * x[1] - beta * [0]**3 + F * cos(omega * t)

    """

    def __init__(
        self,
        t: float,
        x: np.ndarray,
        omega: float,
        alpha: float,
        beta: float,
        F: float,
        delta: float,
        exponent=3,
    ):

        autonomous = False
        n_dof = 2  # number of states in the ODE
        super().__init__(autonomous=autonomous, n_dof=n_dof)
        self.t = t
        self.x = x
        self.alpha = alpha
        self.beta = beta
        self.F = F
        self.omega = omega
        self.delta = delta
        self.exponent = exponent

    @override
    def dynamics(self, t=None, x=None) -> np.ndarray:
        """The dynamics must return the right-hand side of the ODE as a numpy array of shape (n_dof, *x.shape[1:]). Can be implemented in vectorized form such that x can be a 2D array of shape (n_dof, n_samples) and t a 1-d numpy array of corresponding length.

        The block t = self.t and x = self.x is required for the Newton solution procedure.
        """

        if t is None:
            t = self.t
        if x is None:
            x = self.x
        self.check_dimensions(t, x)

        f = np.zeros_like(x)
        f[0, ...] = x[1, ...]
        f[1, ...] = (
            -self.alpha * x[0, ...]
            - self.delta * x[1, ...]
            - self.beta * x[0, ...] ** self.exponent
            + self.F * np.cos(self.omega * t)
        )

        return f

    @override
    def closed_form_derivative(self, variable, t=None, x=None):
        """Jacobian of the right-hand side of the ODE w.r.t. a variable (x, omega, F, alpha, beta, delta). Can be implemented in vectorized form such that x can be a 2D array of shape (n_dof, n_samples) and t a 1-d numpy array of corresponding length. Its implementation is optional, but can speed up the Newton solution procedure. If a NotImplementedError is raised, the Jacobian will be approximated numerically."""
        if t is None:
            t = self.t
        if x is None:
            x = self.x

        self.check_dimensions(t, x)

        match variable:
            case "x":
                return self.df_dx(t, x)
            case "alpha":
                return self.df_dalpha(t, x)
            case _:
                raise NotImplementedError(
                    f"Derivative w.r.t {variable} not implemented in closed form."
                )

    def df_dx(self, t=None, x=None):

        if t is None:
            t = self.t
        if x is None:
            x = self.x

        df_dx = np.zeros((2, *x.shape), dtype=x.dtype)
        df_dx[0, 1, ...] = 1
        df_dx[1, 0, ...] = -self.alpha - self.exponent * self.beta * x[0, ...] ** (
            self.exponent - 1
        )
        df_dx[1, 1, ...] = -self.delta

        return df_dx

    def df_dalpha(self, t=None, x=None):

        if x is None:
            x = self.x

        df_dal = np.zeros_like(x)
        df_dal[1, ...] = -x[0, ...]

        return df_dal[:, np.newaxis, ...]


def compute_frc():
    """
    Demonstration for creating a simple frequency response curve of a non-autonomous dynamical system with HBM.

    This function performs the following steps:

    #. Creation of :py:class:`~skhippr.Fourier`, :py:class:`~skhippr.solvers.newton.NewtonSolver`, and :py:class:`~skhippr.stability.KoopmmanHillProjection.KoopmanHillSubharmonic` objects to collect method parameters
    #. Instantiation of a :py:class:`~skhippr.odes.AbstractODE.AbstractODE` (here: :py:class:`~skhippr.odes.nonautonomous.Duffing`) object which contains the ODE
    #. Setup of an initial guess
    #. Setup and solution of the :py:class:`~skhippr.cycles.hbm.HBMEquation`, which formalizes the Harmonic Balance equations
    #. Creation of an :py:class:`~skhippr.equations.EquationSystem.EquationSystem` containing only the HBM equations as input to the continuation method
    #. Continuation of the frequency response curve using :py:func:`~skhippr.solvers.continuation.pseudo_arclength_continuator` and collecting the branch points
    #. Plotting the continuation curve from the collected :py:class:`~skhippr.solvers.continuation.BranchPoint` objects via SKHiPPR visualization tools.

    Returns
    -------
    initial_system : EquationSystem
        The initial (un-extended) :py:class:`~skhippr.equations.EquationSystem.EquationSystem`, containing the solved :py:class:`~skhippr.cycles.hbm.HBMEquation` at ``omega = 0.3``.
    frc : list[BranchPoint]
        The collected :py:class:`~skhippr.solvers.continuation.BranchPoint` objects along the frequency response curve.
    """

    # --- Parameters and creation of ODE (Duffing oscillator) ---
    omega = 0.3
    F = 0.5

    ode = Duffing(t=0, x=[1.0, 0.0], alpha=1, beta=2, delta=0.16, F=F, omega=omega)

    # --- FFT, stability method and Newton solver configuration ---
    N_HBM = 25
    L_DFT = 300

    fourier = Fourier(N_HBM=N_HBM, L_DFT=L_DFT, n_dof=2, real_formulation=True)
    stability_method = KoopmanHillSubharmonic(fourier, tol=1e-4, autonomous=False)
    solver = NewtonSolver(verbose=False)

    # --- Initial guess in time and frequency domain ---
    ts = fourier.time_samples(ode.omega)
    x0_samples = np.array([np.cos(ode.omega * ts), -ode.omega * np.sin(ode.omega * ts)])
    X0 = fourier.DFT(x0_samples)

    # --- Set up the Harmonic Balance equation
    hbm = HBMEquation(
        ode=ode,
        omega=ode.omega,
        fourier=fourier,
        initial_guess=X0,
        period_k=1,
        stability_method=stability_method,
    )

    # The solver needs an EquationSystem object to know which are the unknowns.
    initial_system = EquationSystem(
        equations=[hbm], unknowns=["X"], equation_determining_stability=hbm
    )

    # --- Preallocate the result of the FRC continuation ---
    # BranchPoints (a subclass of EquationSystem) extend the initial EquationSystem with one added equation for the tangency condition.
    frc: list[BranchPoint] = []

    # --- Iterate through the branch. Almost all arguments are optional. ---
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

    return initial_system, frc


if __name__ == "__main__":
    _, frc = compute_frc()
    plot_continuation(
        frc,
        plot_fun=lambda point: np.max(point.equations[0].x_time()[0, :]),
        ylabel="max_t|x_1(t)|",
    )
    plt.show()
