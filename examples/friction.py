"""FRC of frictional oscillator. See Schütz (2025), Bachelor's thesis, and Legrand2023."""

import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
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


def init_oscillator(name_case='A', smoothing=np.inf):
    masses = [1, 1]
    stiffnesses = [1, 1]
    dampings = [0.02, 0.02]
    omega = 0.299
    phases = [0.5 * np.pi, 0]
    mu = 0.9
    prox_parameter = 1

    match name_case:
        case 'A':
            omega = 0.618
            phases = [0.5 * np.pi, 0]
            forcings = [20, 0]
            normal_force = 8
            g = normal_force / masses[1]
        case 'B':
            omega = 0.293
            phases = [0.5 * np.pi, 0]
            forcings = [20, 0]
            normal_force = 8
            g = normal_force / masses[1]
        case 'C':
            omega = 0.299
            phases = [0.5 * np.pi, 0]
            forcings = [20, 0]
            normal_force = 10.5
            g = normal_force / masses[1]
        case 'D':
            omega = 0.308
            phases = [0.5 * np.pi, 0]
            forcings = [20, 0]
            normal_force = 10.5
            g = normal_force / masses[1]
        
        case 'Schuetz_1':
            g = 10
            omega = 2*np.pi
            forcings = [20, 50]
            phases = [-0.5 * np.pi, np.pi]
            mu = 4

        case 'Schuetz_2':
            g = 10
            omega = 2*np.pi
            forcings = [20, 50]
            phases = [0.5*np.pi, 0]
            smoothing = 40
            prox_parameter=10

        case _:
            raise ValueError(f"Case {name_case} not defined!")

    if smoothing == np.inf:
        dae = FrictionOscillator(
            stiffnesses=stiffnesses,
            dampings=dampings,
            masses=masses,
            g=g,
            mu=mu,
            forcing_amplitudes=forcings,
            forcing_phases=phases,
            prox_parameter=prox_parameter,
        )
    else:
        dae = SmoothedFrictionOscillator(
            stiffnesses=stiffnesses,
            dampings=dampings,
            masses=masses,
            g=g,
            mu=mu,
            forcing_amplitudes=forcings,
            forcing_phases=phases,
            smoothing=smoothing,
        )
        
    dae.omega = omega
    return dae

def solve_hbm(name_case, smoothing=np.inf, fourier=None, hbm_ref=None, solver=None):
    if fourier is None:
        if hbm_ref is None:
            raise ValueError("fourier and hbm_ref may not be both None!")
        else:
            fourier = hbm_ref.fourier

    if not fourier.real_formulation:
        raise ValueError("Only real formulation supported!")

    if solver is None:
        solver = ScipyRootSolver(
            tolerance=1e-8, max_iterations=1000, verbose=True, use_fprime=True, method="lm"
        )

    dae = init_oscillator(name_case, smoothing=smoothing)

    # Initial guess for lambda
    if hbm_ref is None:
        initial_guess_lambda = None
    else:
        X_old = hbm_ref.X
        X_old = np.reshape(X_old, (dae.n_dof, -1), order="F")
        Lambda_old = X_old[-1, :]

        initial_guess_lambda = np.zeros(2 * fourier.N_HBM + 1)

        idx_cos_end_old = min(hbm_ref.fourier.N_HBM + 1, fourier.N_HBM + 1)
        idx_cos_end_new = idx_cos_end_old
        
        idx_sin_start_old = hbm_ref.fourier.N_HBM + 1
        idx_sin_end_old = idx_cos_end_old + hbm_ref.fourier.N_HBM

        idx_sin_start_new = fourier.N_HBM+1
        idx_sin_end_new = idx_cos_end_old + fourier.N_HBM+1

        initial_guess_lambda[:idx_cos_end_new] = Lambda_old[:idx_cos_end_old]
        initial_guess_lambda[idx_sin_start_new:idx_sin_end_new] = Lambda_old[idx_sin_start_old:idx_sin_end_old]

    # Solve with substituted formulation
    equ_lambda = solve_friction(
        oscillator=dae,
        fourier=fourier,
        omega=dae.omega,
        smoothing=smoothing,
        solver=solver,
        initial_guess=initial_guess_lambda,
    )

    X, dX = equ_lambda.FC_dX()
    initial_guess = np.real(np.vstack(((X, dX, equ_lambda.Lambda))).flatten(order="F"))

    # Solve actual problem
    hbm = HBMEquationDAE(
        dae,
        dae.omega,
        fourier=fourier,
        initial_guess=initial_guess,
        stability_method=KoopmanHillDAE(
            fourier, tol=1e-4, autonomous=False, tol_drazin=1e-6
        ),
    )

    if solver.verbose:
        print(
            f"Case {name_case}, N = {fourier.N_HBM}, smoothing = {smoothing}: -- Residual before solving: {np.linalg.norm(hbm.residual(update=True), np.inf)}"
        )

    solver.solve_equation(hbm, unknown="X")

    if solver.verbose:
        print(
            f"Case {name_case}, N = {fourier.N_HBM}, smoothing = {smoothing}: -- Residual after solving: {np.linalg.norm(hbm.residual(update=True), np.inf)}"
        )

    return hbm
    
def plot_and_export_hbm(name_case='C', smoothing=np.inf, N_HBM=160, L_DFT=2**14):
    fourier = Fourier(N_HBM, L_DFT, n_dof=5, real_formulation=True)
    hbm = solve_hbm(name_case, smoothing, fourier)
    x_time = hbm.x_time()
    ts = hbm.fourier.time_samples(hbm.omega)
    residual = np.linalg.norm(hbm.residual(update=True))

    description = f"case-{name_case}-N-{N_HBM}-smoothing-{smoothing}"

    # Plot position
    _, ax = plt.subplots(1,1)
    ax.plot(ts, x_time[0, :], label='x0')
    ax.plot(ts, x_time[1, :], label='x1')
    ax.set_xlabel('t')
    ax.set_ylabel('position')
    ax.set_title(f"position {description} r = {residual}")
    ax.legend()
    tikzplotlib.save(f"plots/position_{description}.tikz")

    # Plot velocity
    _, ax = plt.subplots(1,1)
    ax.plot(ts, x_time[2, :], label='x2')
    ax.plot(ts, x_time[3, :], label='x3')
    ax.set_xlabel('t')
    ax.set_ylabel('velocity')
    ax.set_title(f"velocity {description} r = {residual}")
    ax.legend()
    tikzplotlib.save(f"plots/velocity_{description}.tikz")

    # Plot tangential force
    _, ax = plt.subplots(1,1)
    ax.plot(ts, x_time[4, :], label='lambda')
    ax.set_xlabel('t')
    ax.set_ylabel('lambda')
    ax.set_title(f"lambda {description} r = {residual}")
    ax.legend()
    tikzplotlib.save(f"plots/lambda_{description}.tikz")

    # Plot force law
    _, ax = plt.subplots(1,1)
    ax.plot(x_time[3,:], x_time[4, :], label='lambda')
    ax.set_xlabel('x3')
    ax.set_ylabel('lambda')
    ax.set_title(f"force law {description} r = {residual}")
    ax.legend()
    tikzplotlib.save(f"plots/forcelaw_{description}.tikz")

def convergence_study_N(name_case='C', smoothing=np.inf, Ns_HBM=(30, 40, 50), L_DFT=2**14):

    N_max = Ns_HBM[-1]
    description = f"{name_case}-Nmax{N_max}-smoothing{smoothing}-L{L_DFT}"
    hbm_ref = solve_hbm(name_case, smoothing, Fourier(N_max, L_DFT, n_dof=5, real_formulation=True))
    FM_ref = sort_FMs(hbm_ref.eigenvalues)


    e_hbm = np.zeros((5, len(Ns_HBM)))
    e_hbm_fourier = np.zeros((5, len(Ns_HBM)))
    e_stab = np.zeros((5, len(Ns_HBM)))
    FMs_all = np.zeros((5, len(Ns_HBM)), dtype=complex)

    for k, N_HBM in enumerate(Ns_HBM):
        if k < len(Ns_HBM)-1:
            fourier = Fourier(N_HBM, L_DFT, n_dof=5, real_formulation=True)
            hbm = solve_hbm(name_case, smoothing, fourier, hbm_ref)
        else:
            hbm = hbm_ref

        x_time = hbm.x_time()
        X_comp = hbm_ref.fourier.DFT(x_time)
        e_comp = np.reshape(hbm_ref.X - X_comp, (5, -1), order='F')

        e_hbm[:, k] = np.max(np.abs(x_time - hbm_ref.x_time()), axis=1)
        e_hbm_fourier[:,k] = np.linalg.norm(e_comp, axis=1)
        FMs = sort_FMs(hbm.eigenvalues)
        FMs_all[:, k] = FMs
        e_stab[:, k] = np.abs(FMs - FM_ref)

    # Plot Floquet multipliers
    _, ax = plt.subplots(1,1)
    phis = np.linspace(0, 2*np.pi, 250)
    ax.plot(np.cos(phis), np.sin(phis))
    ax.plot(np.real(FM_ref), np.imag(FM_ref), 'o', label=f"ref(N={N_max})")
    for l in range(FMs_all.shape[0]):
        ax.plot(np.real(FMs_all[l, :]), np.imag(FMs_all[l, :]), '-x', label=f"FM {l}")
    ax.set_aspect('equal')
    ax.set_title(description)
    ax.legend()
    tikzplotlib.save(f"plots/FMs_conv_{description}.tikz")

    # Plot HBM convergence in time
    _, ax = plt.subplots(1,1)
    for l in range(e_hbm.shape[0]):
        ax.semilogy(Ns_HBM[:-1], e_hbm[l, :-1], label=f"x{l}")
    ax.legend()
    ax.set_xlabel('N')
    ax.set_ylabel('error HBM')
    ax.set_title(f"HBM convergence {description}")
    tikzplotlib.save(f"plots/HBM_error_{description}.tikz")

    # Plot HBM convergence in freq domain
    _, ax = plt.subplots(1,1)
    for l in range(e_hbm_fourier.shape[0]):
        ax.semilogy(Ns_HBM[:-1], e_hbm_fourier[l, :-1], label=f"x{l}")
    ax.legend()
    ax.set_xlabel('N')
    ax.set_ylabel('error HBM FCs')
    ax.set_title(f"HBM FC convergence {description}")
    tikzplotlib.save(f"plots/HBM_FC_error_{description}.tikz")

    # Plot HBM convergence in FCs
    _, ax = plt.subplots(1,1)
    for l in range(e_hbm.shape[0]):
        ax.semilogy(Ns_HBM[:-1], e_hbm[l, :-1], label=f"x{l}")
    ax.legend()
    ax.set_xlabel('N')
    ax.set_ylabel('error HBM')
    ax.set_title(f"HBM convergence {description}")
    tikzplotlib.save(f"plots/HBM_error_{description}.tikz")

    # Plot FM convergence
    _, ax = plt.subplots(1,1)
    for l in range(e_stab.shape[0]):
        ax.semilogy(Ns_HBM[:-1], e_stab[l, :-1], label=f"FM {l}")
    ax.legend()
    ax.set_xlabel('N')
    ax.set_ylabel('error FMs')
    ax.set_title(f"FM convergence {description}")
    tikzplotlib.save(f"plots/FMs_error_{description}.tikz")


def sort_FMs(FMs, significant_digits=2):
    # reference: https://gist.github.com/ttamg/3f65227fd580b3d8dc8ba91e01507280
    FMs_rounded = np.zeros_like(FMs)
    for k, FM in enumerate(FMs):
        if abs(FM) > 0:
            round_digits = -int(np.floor(np.log10(np.abs(FM)))) + significant_digits - 1
            FMs_rounded[k] = np.round(FM, round_digits)
        else:
            FMs_rounded[k] = FM
    idx_sort = np.lexsort((np.angle(FMs_rounded), np.abs(FMs_rounded)))
    FMs = FMs[idx_sort]
    FMs_rounded = FMs_rounded[idx_sort]
    # Separate complex and real eigenvalues
    FMs = np.hstack((FMs[np.imag(FMs_rounded) == 0], FMs[np.imag(FMs_rounded)!= 0]))
    return FMs



def plot_everything():

    Ns_HBM = [40]
    L_DFT = 2**13
    

    smoothings = [np.inf]

    for name_case in ['A', 'B', 'C', 'Schuetz_1', 'Schuetz_2']:
        hbm = None

        for k, N_HBM in enumerate(Ns_HBM):
            fourier = Fourier(N_HBM, L_DFT, n_dof=5, real_formulation=True)

            for l, alpha in enumerate(smoothings):
                hbm = solve_hbm(name_case, alpha, fourier, hbm)

                _, axs = plt.subplots(2, 2)
                x_time = hbm.x_time()

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
                    np.cos(hbm.fourier.time_samples_normalized),
                    np.sin(hbm.fourier.time_samples_normalized),
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


def solve_friction(
    oscillator, fourier, omega, smoothing, solver=None, initial_guess=None
):

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

    warmstart = (smoothing == np.inf)
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

        equ.smoothing = 10
        solver.solve_equation(equ, unknown="Lambda")
        equ.smoothing = smoothing

    if solver.verbose:
        print(f"Solving lambda problem (smoothing = {equ.smoothing})...")

    solver.solve_equation(equ, unknown="Lambda")

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
    for name_case in ['C']:
        # plot_and_export_hbm(name_case=name_case, smoothing=np.inf, N_HBM=40, L_DFT=2048)
        convergence_study_N(name_case, Ns_HBM=range(1,50),L_DFT=512)
    # plot_everything()
    # plot_frc()
    plt.show()
