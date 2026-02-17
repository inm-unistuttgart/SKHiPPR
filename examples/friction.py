"""FRC of frictional oscillator. See Schütz (2025), Bachelor's thesis, and Legrand2023."""

import numpy as np
import matplotlib.pyplot as plt
import warnings


from skhippr.odes.daes import FrictionOscillator, SmoothedFrictionOscillator
from skhippr.Fourier import Fourier
from skhippr.solvers.newton import ScipyFsolveSolver, NewtonSolver, ScipyRootSolver
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.equations.AbstractEquation import AbstractEquation
from skhippr.cycles.hbm import HBMEquation, HBMEquationDAE
from skhippr.solvers.continuation import pseudo_arclength_continuator

from skhippr.stability.KoopmanHillProjection import (
    KoopmanHillSubharmonic,
    KoopmanHillDAE,
)


def plot_solution():

    # solver = ScipyRootSolver(
    #     tolerance=1e-8, max_iterations=1000, verbose=True, use_fprime=True, method="lm"
    # )
    solver = ScipyRootSolver(
        tolerance=1e-8,
        max_iterations=10000000,
        verbose=True,
        use_fprime=True,
        method="lm",
    )

    # BA Schütz case 1 (p. 44)
    # masses = [1, 1]
    # g = 10
    # stiffnesses = [1, 1]
    # dampings = [0.02, 0.02]
    # forcings = [20, 50]
    # omega = 2 * np.pi
    # phases = [-0.5 * np.pi, np.pi]
    # mu = 4
    # smoothing = 10

    # # BA Schütz case 2 (p. 50)
    # masses = [1, 1]
    # g = 10
    # stiffnesses = [1, 1]
    # dampings = [0.02, 0.02]
    # forcings = [20, 10]
    # omega = 2 * np.pi
    # phases = [0.5 * np.pi, 0]
    # mu = 0.9
    # smoothing = 40
    # prox_parameter = 10

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
    Ns_HBM = [40]
    L_DFT = 4096

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

    for k, N_HBM in enumerate(Ns_HBM):
        for l, dae in enumerate([dae_smooth]):

            if k + l == 0:
                initial_guess = 0.1 * np.random.rand(dae.n_dof * (2 * N_HBM + 1))
            else:
                initial_guess = np.zeros(dae.n_dof * (2 * N_HBM + 1))

                idx_cos_end = dae.n_dof * (hbm.fourier.N_HBM + 1)
                idx_sin_start = dae.n_dof * (N_HBM + 1)
                idx_sin_end = dae.n_dof * (N_HBM + 1 + hbm.fourier.N_HBM)
                initial_guess[:idx_cos_end] = hbm.X[:idx_cos_end]
                initial_guess[idx_sin_start:idx_sin_end] = hbm.X[idx_cos_end:]

            fourier = Fourier(
                N_HBM=N_HBM, L_DFT=L_DFT, n_dof=dae.n_dof, real_formulation=True
            )

            hbm = HBMEquationDAE(
                dae,
                omega,
                fourier=fourier,
                initial_guess=initial_guess,
                stability_method=KoopmanHillDAE(
                    fourier, tol=1e-4, autonomous=False, tol_drazin=1e-6
                ),
            )

            try:
                alpha = dae.smoothing
            except AttributeError:
                alpha = "inf"

            print(
                f"N = {N_HBM}, alpha = {alpha}: -- Residual before solving: {np.linalg.norm(hbm.residual(update=True), np.inf)}"
            )

            # plt.figure()
            # initial_res = hbm.residual(update=False)
            # initial_res = np.reshape(initial_res, (dae.n_dof, -1), order="F")
            # plt.imshow(initial_res)
            # plt.colorbar()
            # plt.title(f"Initial residual N_HBM = {N_HBM}")
            # plt.xlabel("harmonic")
            # plt.ylabel("state")

            try:
                solver.solve_equation(hbm, unknown="X")
            except RuntimeError as R:
                print(R)

            print(
                f"N = {N_HBM}, alpha = {alpha}: -- Residual after solving: {np.linalg.norm(hbm.residual(update=True), np.inf)}"
            )

            _, axs = plt.subplots(2, 2)
            x_time = hbm.x_time()
            fourier = hbm.fourier

            for i in range(2):
                axs[i][0].plot(x_time[i, :], x_time[i + 2, :], "-")
                axs[i][0].set_title(f"Phase plot of x_[{i}]")
                axs[i][0].set_ylabel(f"dx_[{i}]")
                axs[i][0].set_xlabel(f"x_[{i}]")

            floquet_multipliers = hbm.eigenvalues
            axs[0][1].plot(
                np.real(floquet_multipliers), np.imag(floquet_multipliers), "x"
            )
            axs[0][1].set_title("Floquet multipliers")
            axs[0][1].plot(
                np.cos(fourier.time_samples_normalized),
                np.sin(fourier.time_samples_normalized),
                "k",
            )
            axs[0][1].axis("equal")
            axs[0][0].set_title(f"Friction oscillator N_HBM = {N_HBM}, alpha = {alpha}")

            _, axs = plt.subplots(hbm.n_dof, 1)
            x_time = hbm.x_time()
            for i in range(hbm.n_dof):
                axs[i].plot(hbm.fourier.time_samples(hbm.omega), x_time[i, :])
            axs[0].set_title(f"Friction oscillator N_HBM = {N_HBM}, alpha = {alpha}")


def plot_frc():

    solver = ScipyRootSolver(
        tolerance=1e-7, max_iterations=100, verbose=True, use_fprime=True, method="lm"
    )

    # BA Schütz case 2 (p. 50)
    masses = [1, 1]
    g = 10
    stiffnesses = [1, 1]
    dampings = [0.02, 0.02]
    forcings = [20, 0]
    omega = 0.1
    phases = [0.5 * np.pi, 0]
    mu = 0.9
    smoothing = 10
    prox_parameter = 1

    # Systems
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

    fourier = Fourier(
        N_HBM=60, L_DFT=2000, n_dof=dae_smooth.n_dof, real_formulation=True
    )

    fig, axs = plt.subplots(1, 2)
    alphas = [smoothing, "nonsmooth"]

    for k, dae in enumerate([dae_smooth, dae_nonsmooth]):

        if k == 0:
            initial_guess = 0.1 * np.random.rand(dae.n_dof * (2 * fourier.N_HBM + 1))
        else:
            initial_guess = hbm.X

        hbm = HBMEquationDAE(
            dae,
            omega,
            fourier=fourier,
            initial_guess=initial_guess,
            stability_method=KoopmanHillDAE(
                fourier, tol=1e-4, autonomous=False, tol_drazin=1e-6
            ),
        )

        sys = EquationSystem(
            equations=[hbm], unknowns=["X"], equation_determining_stability=hbm
        )

        solver.verbose = True
        solver.solve(sys)
        solver.verbose = False

        for branch_point in pseudo_arclength_continuator(
            initial_system=sys,
            solver=solver,
            continuation_parameter="omega",
            stepsize=0.1,
            stepsize_range=[0.001, 2],
            num_steps=10,
            verbose=True,
        ):
            if branch_point.stable:
                color = "r"
            else:
                color = "b"
            axs[k].plot(
                branch_point.omega,
                np.max(np.abs(branch_point.equations[0].x_time()[0, :])),
                f"{color}.",
            )

            if branch_point.omega > 1:
                break

        axs[k].set_title(f"Friction oscillator FRC smoothing = {alphas[k]}")
        axs[k].set_xlabel("omega")
        axs[k].set_ylabel("|x[0]| max")


def solve_friction():
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

    N_HBM = 40
    L_DFT = 4096

    initial_guess = np.zeros(2 * N_HBM + 1)
    initial_guess[1] = -10

    solver = ScipyRootSolver(
        tolerance=1e-8,
        max_iterations=1000000,
        verbose=True,
        use_fprime=False,
        method="lm",
    )

    for smoothing in [10]:

        equ = FrictionDirect(
            N_HBM=N_HBM,
            L_DFT=L_DFT,
            real_formulation=True,
            stiffnesses=stiffnesses,
            dampings=dampings,
            masses=masses,
            g=g,
            mu=mu,
            forcing_amplitudes=forcings,
            forcing_phases=phases,
            omega=omega,
            smoothing=smoothing,
            prox_parameter=prox_parameter,
            initial_guess=initial_guess,
        )
        solver.solve_equation(equ, unknown="Lambda")
        x_time = equ.x_time()
        ts = equ.fourier.time_samples(equ.omega)
        _, axs = plt.subplots(5, 1)
        initial_guess = equ.Lambda
        for k in range(5):
            axs[k].plot(ts, x_time[k, :])
            if k == 0:
                axs[k].set_title(f"Direct Friction oscillator smoothing = {smoothing}")
            axs[k].set_xlabel("time")
            axs[k].set_ylabel(f"x[{k}]")


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
        return residual

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


if __name__ == "__main__":
    solve_friction()
    plot_solution()
    # plot_frc()
    plt.show()
