import numpy as np
import matplotlib.pyplot as plt

from skhippr.solvers.newton import ScipyRootSolver
from skhippr.equations import AbstractEquation
from skhippr.Fourier import Fourier
from skhippr.odes.daes import FrictionOscillator, SmoothedFrictionOscillator


class FrictionDirect(AbstractEquation):
    """Substituted formulation of the friction oscillator, solving only for Lambda and inferring the rest."""

    def __init__(
        self,
        N_HBM=60,
        L_DFT=4096,
        real_formulation=True,
        stiffnesses=1,
        dampings=0.1,
        masses=(1, 1),
        g=9.81,
        mu=1,
        forcing_amplitudes=(1, 0),
        forcing_phases=(0, 0),
        omega=1,
        smoothing=np.inf,
        prox_parameter=1,
        initial_guess=None,
    ):
        super().__init__(stability_method=None)
        self.masses = np.atleast_1d(masses)
        self.n_dof = len(self.masses)
        self.g = g
        self.mu = mu
        self.forcing_amplitudes = np.atleast_1d(forcing_amplitudes)
        self.forcing_phases = np.atleast_1d(forcing_phases)
        self.smoothing = smoothing
        self.prox_parameter = prox_parameter
        self.fourier = Fourier(
            N_HBM=N_HBM, L_DFT=L_DFT, n_dof=1, real_formulation=real_formulation
        )
        self.omega = omega

        if initial_guess is None:
            initial_guess = np.zeros((2 * self.fourier.N_HBM + 1))
        self.Lambda = initial_guess

        D = self.fourier.derivative_matrix
        M = np.diag(self.masses)
        stiffnesses = np.append(stiffnesses, 0)
        dampings = np.append(dampings, 0)

        K = np.zeros((len(masses), len(masses)))
        C = np.zeros(K.shape)
        for i in range(len(masses)):

            K[i, i] = stiffnesses[i + 1] + stiffnesses[i]
            if i > 0:
                K[i, i - 1] = -stiffnesses[i]
            if i < len(masses) - 1:
                K[i, i + 1] = -stiffnesses[i + 1]

            C[i, i] = dampings[i + 1] + dampings[i]
            if i > 0:
                C[i, i - 1] = -dampings[i]
            if i < len(masses) - 1:
                C[i, i + 1] = -dampings[i + 1]
        self.M = np.kron(self.omega**2 * D @ D, M)
        self.C = np.kron(self.omega * D, C)
        self.K = np.kron(np.eye(2 * self.fourier.N_HBM + 1), K)
        self.Z = self.M + self.C + self.K

        t = self.fourier.time_samples(self.omega)
        arguments = self.forcing_phases[:, np.newaxis] + self.omega * t[np.newaxis, :]
        forcing = self.forcing_amplitudes[:, np.newaxis] * np.sin(arguments)

        forcing_FC = np.zeros((self.n_dof, 2 * self.fourier.N_HBM + 1), dtype=complex)
        for i in range(self.n_dof):
            forcing_FC[i, :] = self.fourier.DFT(np.atleast_2d(forcing[i, :]))

        self.F = forcing_FC.flatten(order="F")
        W_time = np.zeros(self.n_dof)
        W_time[-1] = 1
        self.W = np.kron(np.eye(2 * self.fourier.N_HBM + 1), W_time[:, np.newaxis])

        self.lam_crit = self.mu * self.masses[-1] * self.g
        self.N_odd = int(np.ceil(self.fourier.N_HBM / 2))

    @property
    def Lambda(self):
        Lambda_cos_odd = self.Lambda_odd[: self.N_odd]
        Lambda_sin_odd = self.Lambda_odd[self.N_odd :]

        Lambda_const = np.array([0])

        Lambda_cos = np.vstack((Lambda_cos_odd, np.zeros_like(Lambda_cos_odd)))
        Lambda_cos = np.reshape(Lambda_cos, (-1), order="F")
        if self.fourier.N_HBM % 2 != 0:
            Lambda_cos = Lambda_cos[:-1]

        Lambda_sin = np.vstack((Lambda_sin_odd, np.zeros_like(Lambda_sin_odd)))
        Lambda_sin = np.reshape(Lambda_sin, (-1), order="F")
        if self.fourier.N_HBM % 2 != 0:
            Lambda_sin = Lambda_sin[:-1]

        return np.hstack((Lambda_const, Lambda_cos, Lambda_sin))

    @Lambda.setter
    def Lambda(self, value):
        self.Lambda_odd = self.extract_odd(value)

    def extract_odd(self, Lambda):
        Lambda_const = Lambda[0]
        Lambda_cos = Lambda[1 : self.fourier.N_HBM + 1]
        Lambda_sin = Lambda[self.fourier.N_HBM + 1 :]
        Lambda_odd = np.hstack((Lambda_cos[::2], Lambda_sin[::2]))
        Lambda_even = np.hstack(((Lambda_const,), Lambda_cos[1::2], Lambda_sin[1::2]))
        if any(np.abs(Lambda_even) > 1e-14):
            # raise ValueError("Lambda_even must be zero.")
            print(f"ignored even values of magnitude {np.max(np.abs(Lambda_even))}.")
        return Lambda_odd

    def FC_X(self, Lambda=None):
        if Lambda is None:
            Lambda = self.Lambda

        X = np.linalg.solve(self.Z, self.W @ Lambda + self.F)
        X = np.reshape(X, (self.n_dof, -1), order="F")
        return X

    def FC_dX(self, Lambda=None):
        X = self.FC_X(Lambda)
        dX = np.zeros_like(X)
        for i in range(X.shape[0]):
            dX[i, :] = self.fourier.derivative_coeffs(X[i, :], omega=self.omega)
        return X, dX

    def x_time(self, Lambda=None):
        if Lambda is None:
            Lambda = self.Lambda
        X, dX = self.FC_dX(Lambda)
        x_time = np.zeros((2 * X.shape[0] + 1, self.fourier.L_DFT), dtype=X.dtype)
        for k in range(X.shape[0]):
            x_time[k, :] = self.fourier.inv_DFT(X[k, :])
            x_time[k + X.shape[0], :] = self.fourier.inv_DFT(dX[k, :])
        x_time[-1, :] = self.fourier.inv_DFT(Lambda)
        return x_time

    def FC_gamma(self, Lambda=None):
        _, dX = self.FC_dX(Lambda)
        return dX[-1, :]

    def residual_with_argument(self, Lambda=None):
        if Lambda is None:
            Lambda = self.Lambda

        Gamma = self.FC_gamma(Lambda)
        lambdas = self.fourier.inv_DFT(Lambda)
        gammas = self.fourier.inv_DFT(Gamma)
        residual_time = self.constraint(gammas, lambdas)
        residual = self.fourier.DFT(residual_time)
        return self.extract_odd(residual)

    def residual_function(self):
        return self.residual_with_argument()

    def constraint(self, gammas, lambdas):
        if self.smoothing == np.inf:
            return self.constraint_prox(gammas, lambdas)
        else:
            return self.constraint_smooth(gammas, lambdas)

    def constraint_smooth(self, gammas, lambdas):
        return lambdas + self.lam_crit * np.tanh(self.smoothing * gammas)

    def constraint_prox(self, gammas, lambdas):
        return (
            gammas
            + np.minimum(
                0,
                self.prox_parameter * (lambdas + self.lam_crit) - gammas,
            )
            + np.maximum(
                0,
                self.prox_parameter * (lambdas - self.lam_crit) - gammas,
            )
        )

    def closed_form_derivative(self, variable):
        return super().closed_form_derivative(variable)


def solve_friction(
    oscillator, fourier, omega, smoothing, solver=None, initial_guess=None
):
    """Solves the substituted formulation of the friction oscillator, i.e.,
    solves for Lambda and infers the rest.
    This is used to get a warm start for the actual HBM problem,
    which is hard to solve directly for the non-smooth case."""

    if solver is None:
        solver = ScipyRootSolver(
            tolerance=1e-8,
            max_iterations=1000000,
            verbose=True,
            use_fprime=False,
            method="lm",
        )

    try:
        prox_parameter = oscillator.prox_parameter
    except AttributeError:
        if smoothing == np.inf:
            raise ValueError("Must specify prox_parameter for non-smooth oscillator.")
        prox_parameter = 0

    g = oscillator.lam_crit / (oscillator.mu * oscillator.masses[-1])

    warmstart = smoothing == np.inf and initial_guess is None
    if initial_guess is None:
        initial_guess = np.zeros(2 * fourier.N_HBM + 1)

    equ = FrictionDirect(
        N_HBM=fourier.N_HBM,
        L_DFT=fourier.L_DFT,
        real_formulation=fourier.real_formulation,
        stiffnesses=oscillator.stiffnesses,
        dampings=oscillator.dampings,
        masses=oscillator.masses,
        g=g,
        mu=oscillator.mu,
        forcing_amplitudes=oscillator.forcing_amplitudes,
        forcing_phases=oscillator.forcing_phases,
        omega=omega,
        smoothing=smoothing,
        prox_parameter=prox_parameter,
        initial_guess=initial_guess,
    )

    if warmstart:  # warm-start with smoothed solution
        if solver.verbose:
            print("Solving smoothed lambda problem for warm-start...")

        equ.smoothing = 40
        solver.solve_equation(equ, unknown="Lambda_odd")
        equ.smoothing = smoothing

    if solver.verbose:
        print(
            f"Solving lambda problem (smoothing = {equ.smoothing}). Residual before: {np.linalg.norm(equ.residual(update=True))}..."
        )

    solver.solve_equation(equ, unknown="Lambda_odd")

    return equ


def plot_solve_friction():
    # # Legrand
    masses = [1, 1]
    stiffnesses = [1, 1]
    dampings = [0.02, 0.02]
    forcings = [20, 0]
    omega = 0.299
    phases = [0.5 * np.pi, 0]
    mu = 0.9
    smoothing = 10
    prox_parameter = 1
    normal_force = 10.5
    g = normal_force / masses[1]

    # warm-start from smoothed oscillator
    N_HBM = 40
    L_DFT = 4096
    fourier = Fourier(N_HBM=N_HBM, L_DFT=L_DFT, n_dof=5, real_formulation=True)

    dae_smooth = SmoothedFrictionOscillator(
        stiffnesses=stiffnesses,
        dampings=dampings,
        masses=masses,
        g=g,
        mu=mu,
        forcing_amplitudes=forcings,
        forcing_phases=phases,
        smoothing=smoothing,
    )

    dae_nonsmooth = FrictionOscillator(
        stiffnesses=stiffnesses,
        dampings=dampings,
        masses=masses,
        g=g,
        mu=mu,
        forcing_amplitudes=forcings,
        forcing_phases=phases,
        prox_parameter=prox_parameter,
    )

    for smoothing, dae in zip(
        (dae_smooth.smoothing, np.inf), [dae_smooth, dae_nonsmooth]
    ):

        equ = solve_friction(dae, fourier, omega, smoothing, initial_guess=None)

        x_time = equ.x_time()
        ts = equ.fourier.time_samples(equ.omega)
        _, axs = plt.subplots(5, 1)
        for k in range(5):
            axs[k].plot(ts, x_time[k, :])
            if k == 0:
                axs[k].set_title(f"Direct Friction oscillator smoothing = {smoothing}")
            axs[k].set_xlabel("time")
            axs[k].set_ylabel(f"x[{k}]")
