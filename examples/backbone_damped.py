from typing import override
import numpy as np
import matplotlib.pyplot as plt

from skhippr.odes.AbstractODE import AbstractODE
from skhippr.equations.AbstractEquation import AbstractEquation
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.cycles.hbm import HBMEquation, HBMPhaseAnchor
from skhippr.solvers.newton import NewtonSolver
from skhippr.solvers.continuation import pseudo_arclength_continuator

from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic

from skhippr.visualization.continuation import plot_continuation
from skhippr.visualization.cycles import animate_floquet_multipliers


from skhippr.Fourier import Fourier


class Duffing(AbstractODE):
    """
    Non-autonomous Duffing oscillator as concrete subclass of :py:class:`~skhippr.odes.AbstractODE.AbstractODE`. ::

        dx[0]/dt = x[1]
        dx[1]/dt = -alpha * x[0] - delta * x[1] - beta * [0]**3 + F * cos(omega * t) + modal_damping * x[1]

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
        modal_damping=0,
    ):

        if F == 0:
            autonomous = True
        else:
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
        self.modaldamping = modal_damping

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
            - np.squeeze(self.delta) * x[1, ...]
            - self.beta * x[0, ...] ** self.exponent
            + self.F * np.cos(self.omega * t)
            + np.squeeze(self.modaldamping) * x[1, ...]
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
            case "delta":
                return self.df_ddelta(t, x)
            case "modaldamping":
                return -self.closed_form_derivative("delta", t, x)
            case "omega":
                raise NotImplementedError
            case _:
                return np.zeros_like(x)

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
        df_dx[1, 1, ...] = -np.squeeze(self.delta) + np.squeeze(self.modaldamping)

        return df_dx

    def df_dalpha(self, t=None, x=None):

        if x is None:
            x = self.x

        df_dal = np.zeros_like(x)
        df_dal[1, ...] = -x[0, ...]

        return df_dal[:, np.newaxis, ...]

    def df_ddelta(self, t, x):
        if x is None:
            x = self.x
        df_del = np.zeros_like(x)
        df_del[1, ...] = -x[1, ...]

        return df_del[:, np.newaxis, ...]

    def df_dmodal(self, t, x):
        if x is None:
            x = self.x
        df_del = np.zeros_like(x)
        df_del[1, ...] = -x[1, ...]

        return -df_del[:, np.newaxis, ...]


class AmplitudeEquation(AbstractEquation):

    def __init__(self, X=None, amplitude=None):
        super().__init__(stability_method=None)
        self.X = X
        self.amplitude = amplitude

    def residual_function(self):
        return np.atleast_1d(np.inner(self.X, self.X) - self.amplitude**2)

    def closed_form_derivative(self, variable):
        match variable:
            case "X":
                return np.atleast_2d(2 * self.X)
            case "amplitude":
                return np.atleast_2d(-2 * self.amplitude)
            case _:
                return np.atleast_2d(0.0)


alpha = 1
beta = 0.2
damping = 5

N_HBM = 15
L_DFT = 512

amp_init = 0.01

ode = Duffing(
    t=0,
    x=np.zeros([0, 0]),
    omega=1,
    alpha=alpha,
    beta=beta,
    F=0,
    delta=damping,
    exponent=3,
    modal_damping=0.9 * damping,
)
fourier = Fourier(N_HBM=N_HBM, L_DFT=L_DFT, n_dof=2, real_formulation=True)

t = fourier.time_samples_normalized
x_init = amp_init * np.array([np.cos(t), -np.sin(t)])
X_init = fourier.DFT(x_init)

stability_method = KoopmanHillSubharmonic(fourier)

hbm = HBMEquation(
    ode,
    omega=1,
    fourier=fourier,
    initial_guess=X_init,
    stability_method=stability_method,
)

anchor = HBMPhaseAnchor(fourier, X_init, harmo=1, dof=0)

amp_eq = AmplitudeEquation(X=X_init, amplitude=amp_init)

sys = EquationSystem(
    [hbm, anchor, amp_eq],
    unknowns=["X", "omega", "modaldamping"],
    equation_determining_stability=hbm,
)

solver = NewtonSolver(tolerance=1e-7, max_iterations=20, verbose=True)

solver.solve(sys)
solver.verbose = False

backbone = []
for continuation_point in pseudo_arclength_continuator(
    initial_system=sys,
    solver=solver,
    stepsize=0.01,
    continuation_parameter="amplitude",
    verbose=True,
    num_steps=200,
):
    backbone.append(continuation_point)

plot_continuation(
    backbone,
    lambda cp: cp.omega,
    title="Backbone curve of the Duffing oscillator",
    ylabel="omega",
)
anim = animate_floquet_multipliers(backbone)
plt.show()
