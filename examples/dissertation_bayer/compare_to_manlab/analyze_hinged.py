import numpy as np

import numpy as np
import matplotlib.pyplot as plt

# --- Differential equation ---
from skhippr.odes.manlab import HingedHinged

# --- Fourier configuration ---
from skhippr.Fourier import Fourier

# --- Stability method ---
from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic

# --- Continuation + solver ---
from skhippr.solvers.continuation import pseudo_arclength_continuator
from skhippr.solvers.newton import NewtonSolver

# --- HBM equation system ---
from skhippr.cycles.hbm import HBMSystem

# --- Visualization ---
from skhippr.visualization.continuation import plot_continuation
from skhippr.visualization.data_export import save_tikz


# --- Reference solution creation ---
from create_reference import iterate_reference_solution


def main():
    # ode = init_hinged(2, 0.005, 0.1)
    # frc = compute_frc(
    #     ode,
    #     N_HBM=16,
    #     L_DFT=300,
    #     verbose=True,
    #     omega_max_normalized=1.8,
    #     max_stepsize=0.4,
    # )
    frc = create_hinged_reference(n_modes=3, N_HBM=10, L_DFT=1024)
    ax = plot_continuation(frc, plot_fun=plot_fun)

    save_tikz(
        axes=ax,
        filepath=f"hinged_hinged_ref_N_{frc[0].equations[0].fourier.N_HBM}.tikz",
    )

    plt.show()


def init_hinged(n_modes=10, xi_0=0.005, omega_0_normalized=0.6):
    epsilon = 12
    omegas = np.pi**2 * np.arange(1, n_modes + 1) ** 2
    xis = xi_0 * omegas[0] / omegas
    omega_0 = omega_0_normalized * omegas[0]

    amp_forcing = np.array([-13.63, 9.62])
    phase_forcing = np.array([0.0, 0.0])
    x_forcing = np.array([0.25, 0.75])

    ode = HingedHinged(
        t=0,
        x=np.zeros(2 * n_modes),
        xi=xis,
        epsilon=epsilon,
        omega=omega_0,
        amp_forcing=amp_forcing,
        phase_forcing=phase_forcing,
        x_forcing=x_forcing,
    )
    del ode.eigenvalues
    del ode.stability_method
    del ode.residual_value
    return ode


def init_hbm_sys(ode, fourier, stability_method=None):
    """evaluate initial guess (first mode = cos(omega*t)) in time and frequency domain and return HBMSystem object."""
    ts = fourier.time_samples(ode.omega)
    c = np.cos(ode.omega * ts)
    s = np.sin(ode.omega * ts)
    x0_samples = np.vstack(
        [
            c,
            np.zeros((ode.n_modes - 1, len(ts))),
            -ode.omega * s,
            np.zeros((ode.n_modes - 1, len(ts))),
        ]
    )
    X0 = fourier.DFT(x0_samples)
    return HBMSystem(
        ode=ode,
        omega=ode.omega,
        fourier=fourier,
        initial_guess=X0,
        period_k=1,
        stability_method=stability_method,
    )


def compute_frc(
    ode, N_HBM=25, L_DFT=300, verbose=True, omega_max_normalized=1.8, max_stepsize=0.1
):

    fourier = Fourier(N_HBM=N_HBM, L_DFT=L_DFT, n_dof=ode.n_dof, real_formulation=True)
    stability_method = KoopmanHillSubharmonic(fourier, tol=1e-4, autonomous=False)
    solver = NewtonSolver(verbose=False)

    hbm = init_hbm_sys(ode, fourier, stability_method)

    frc = []
    for branch_point in pseudo_arclength_continuator(
        initial_system=hbm,
        solver=solver,
        stepsize=0.1,
        stepsize_range=(0.001, max_stepsize),
        continuation_parameter="omega",
        initial_direction=1,
        verbose=verbose,
        num_steps=np.inf,
    ):
        frc.append(branch_point)

        # break if omega exceeds maximum
        if branch_point.omega > ode.omegas[0] * omega_max_normalized:
            break
    return frc


def plot_fun(bp, x_eval=0.75):
    ode = bp.equations[0].ode
    omega_normalized = bp.omega / ode.omegas[0]
    x_time = bp.equations[0].x_time()
    phis = np.sqrt(12) * np.sqrt(2) * np.sin(np.pi * ode.mode_numbers * x_eval)
    q_phi = x_time[: ode.n_modes, :] * phis[:, np.newaxis]
    w = np.sum(q_phi, axis=0)
    return (omega_normalized, np.max(np.abs(w)))


def create_hinged_reference(n_modes=3, N_HBM=10, L_DFT=1024):
    solver = NewtonSolver(tolerance=1e-13, verbose=False)
    ode = init_hinged(n_modes=n_modes, xi_0=0.005, omega_0_normalized=0.1)
    fourier = Fourier(N_HBM=N_HBM, L_DFT=L_DFT, n_dof=ode.n_dof)

    hbm = init_hbm_sys(ode, fourier, stability_method=KoopmanHillSubharmonic(fourier))

    shooting_tol = 1e-6

    frc_ref = []

    for bp in iterate_reference_solution(
        filename=f"examples/dissertation_bayer/compare_to_manlab/hinged_nmodes_{ode.n_modes}_N_{fourier.N_HBM}_tol_{shooting_tol}.csv",
        initial_system=hbm,
        solver=solver,
        stepsize=0.1,
        stepsize_range=(0.001, 0.1),
        initial_direction=1,
        continuation_parameter="omega",
        verbose=True,
        num_steps=5,
        atol=shooting_tol,
        rtol=shooting_tol,
    ):
        if bp.omega / ode.omegas[0] > 1.8:
            break

        frc_ref.append(bp)
    return frc_ref


if __name__ == "__main__":
    main()
