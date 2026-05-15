import numpy as np
import matplotlib.pyplot as plt


from skhippr.odes.AbstractODE import AbstractODE
from skhippr.Fourier import Fourier
from skhippr.cycles.hbm import HBMSystem
from skhippr.solvers.continuation import pseudo_arclength_continuator
from skhippr.solvers.newton import NewtonSolver
from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic

from skhippr.visualization.continuation import plot_continuation
from skhippr.visualization.continuation import plot_floquet_multiplier_continuation
from skhippr.visualization.cycles import animate_floquet_multipliers


def main():
    ode = SmoothKrackExample(omega=1.5, alpha=6)
    fourier = Fourier(N_HBM=30, L_DFT=2**10, n_dof=ode.n_dof)
    initial_guess = np.zeros((ode.n_dof, 2 * fourier.N_HBM + 1)).flatten()
    hbm = HBMSystem(
        ode,
        omega=ode.omega,
        fourier=fourier,
        initial_guess=initial_guess,
        period_k=1,
        stability_method=KoopmanHillSubharmonic(fourier),
    )

    branch = []
    for bp in pseudo_arclength_continuator(
        initial_system=hbm,
        solver=NewtonSolver(),
        stepsize=0.05,
        stepsize_range=(0.01, 0.1),
        initial_direction=1,
        continuation_parameter="omega",
        verbose=True,
        num_steps=150,
    ):
        branch.append(bp)

    plot_continuation(branch, plot_fun)
    plot_floquet_multiplier_continuation(branch, marker=".", linestyle="none")
    _, anim = animate_floquet_multipliers(branch)
    return anim


def plot_fun(bp):
    X = np.reshape(
        bp.X,
        (bp.equations[0].fourier.n_dof, 2 * bp.equations[0].fourier.N_HBM + 1),
        order="F",
    )
    idx_cos = 1
    idx_sin = bp.equations[0].fourier.N_HBM + 1
    q_cplx = X[0, idx_cos] - 1j * X[0, idx_sin]
    return bp.omega, np.abs(q_cplx)


class SmoothKrackExample(AbstractODE):
    """Example of Eq.s (3.19, 3.20) in Krack2019"""

    def __init__(self, damping=0.1, factor=10, forcing=0.375, omega=1, alpha=10):
        super().__init__(autonomous=False, n_dof=4)
        self.damping = damping
        self.factor = factor
        self.forcing = forcing
        self.omega = omega
        self.alpha = alpha
        self.K = np.array([[1, -1], [-1, 2]])

    def dynamics(self, t=None, x=None):
        if x is None:
            x = self.x
        if t is None:
            t = self.t
        self.check_dimensions(t, x)

        q = x[:2, ...]
        dq = x[2:, ...]

        ddq = -self.damping * dq - self.K @ q
        ddq[0, ...] -= self.enumerator(q[0, ...]) / self.denominator(q[0, ...])
        ddq[1, ...] += self.forcing * np.cos(self.omega * t)

        return np.vstack((dq, ddq))

    def enumerator(self, x0):
        return self.factor * (x0 - 1) * np.exp(self.alpha * (x0 - 1))

    def d_enumerator(self, x0):
        without_factor = np.exp(self.alpha * (x0 - 1)) + self.alpha * (x0 - 1) * np.exp(
            self.alpha * (x0 - 1)
        )
        return self.factor * without_factor

    def denominator(self, x0):
        return 1 + np.exp(self.alpha * (x0 - 1))

    def d_denominator(self, x0):
        return self.alpha * np.exp(self.alpha * (x0 - 1))

    def closed_form_derivative(self, variable, t=None, x=None):
        if x is None:
            x = self.x

        match variable:
            case "x":
                dfdx = np.array(
                    [
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [
                            -self.K[0, 0],
                            -self.K[0, 1],
                            -self.damping,
                            0,
                        ],
                        [-self.K[1, 0], -self.K[1, 1], 0, -self.damping],
                    ]
                )

                # differentiate the nonlinearity
                enum = self.enumerator(x[0, ...])
                denom = self.denominator(x[0, ...])
                d_enum = self.d_enumerator(x[0, ...])
                d_denom = self.d_denominator(x[0, ...])

                dfdx[2, 0] -= (d_enum * denom - enum * d_denom) / denom**2
                return dfdx
            case _:
                raise NotImplementedError(
                    f"Derivative with respect to variable {variable} not implemented in closed form."
                )


class KrackExample(AbstractODE):
    """Example of Eq.s (3.19, 3.20) in Krack2019"""

    def __init__(self, damping=0.1, factor=10, forcing=0.375, omega=1):
        super().__init__(autonomous=False, n_dof=4)
        self.damping = damping
        self.factor = factor
        self.forcing = forcing
        self.omega = omega
        self.K = np.array([[1, -1], [-1, 2]])

    def dynamics(self, t=None, x=None):
        if x is None:
            x = self.x
        if t is None:
            t = self.t
        self.check_dimensions(t, x)

        q = x[:2, ...]
        dq = x[2:, ...]

        ddq = -self.damping * dq - self.K @ q
        ddq[0, ...] -= self.factor * np.maximum(0, q[0, ...] - 1)
        ddq[1, ...] += self.forcing * np.cos(self.omega * t)

        return np.vstack((dq, ddq))

    def closed_form_derivative(self, variable, t=None, x=None):
        if x is None:
            x = self.x

        match variable:
            case "x":
                return np.array(
                    [
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [
                            -self.K[0, 0] - self.factor * (x[0, ...] > 1),
                            -self.K[0, 1],
                            -self.damping,
                            0,
                        ],
                        [-self.K[1, 0], -self.K[1, 1], 0, -self.damping],
                    ]
                )
            case _:
                raise NotImplementedError(
                    f"Derivative with respect to variable {variable} not implemented in closed form."
                )


if __name__ == "__main__":
    anim = main()
    plt.show()
