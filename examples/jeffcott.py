"""Jeffcott rotor analysis for thesis presentation."""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"


from matplotlib.animation import FuncAnimation
import tikzplotlib

from skhippr.Fourier import Fourier
from skhippr.cycles.hbm import HBMEquation
from skhippr.cycles.shooting import ShootingBVP

from skhippr.equations.EquationSystem import EquationSystem
from skhippr.odes.AbstractODE import AbstractODE
from skhippr.solvers.continuation import BranchPoint, pseudo_arclength_continuator
from skhippr.solvers.newton import NewtonSolver
from skhippr.stability.KoopmanHillProjection import (
    KoopmanHillSubharmonic,
    KoopmanHillProjection,
)
from skhippr.visualization.continuation import (
    plot_continuation,
)
from skhippr.visualization.cycles import (
    animate_floquet_multipliers,
    plot_phase,
)

from skhippr.visualization.data_export import save_animation

from skhippr.equations.PseudoSpectrumEquation import (
    finite_support_error_bound,
    compute_pseudospectrum,
)


class Jeffcott2(AbstractODE):
    """3rd order model, Alcorta2023 Eq. (5)"""

    def __init__(
        self, D_e, D_if, D_it, omega_t, omega, e, r=np.inf, normal_stiffness=1, eta=1e-5
    ):
        super().__init__(autonomous=False, n_dof=4)
        self.D_e = D_e
        self.D_if = D_if
        self.D_it = D_it
        self.omega_t = omega_t
        self.omega = omega
        self.e = e
        self.has_nontrivial_omega_derivative = True
        self.r = r
        self.normal_stiffness = normal_stiffness
        self.eta = eta

    def dynamics(self, t=None, x=None):
        if x is None:
            x = self.x
        if t is None:
            t = self.t
        self.check_dimensions(t, x)

        q = x[:2, ...]
        dq = x[2:, ...]

        f = np.zeros_like(x)
        f[:2, ...] = dq

        forcing = np.array(
            [
                self.e * self.omega**2 * np.cos(self.omega * t),
                self.e * self.omega**2 * np.sin(self.omega * t),
            ]
        )

        factor_nonl = 0
        for i in range(2):
            factor_nonl += self.D_it * q[i, ...] * dq[i, ...]
            factor_nonl += 2 * self.omega_t**2 * q[i, ...] ** 2

        for i in range(2):
            f[2 + i, ...] = -q[i, ...]
            f[2 + i, ...] -= (self.D_if + self.D_e) * dq[i, ...]
            f[2 + i, ...] -= np.sign(0.5 - i) * self.omega * self.D_if * q[1 - i, ...]
            f[2 + i, ...] -= factor_nonl * q[i, ...]
            f[2 + i, ...] += forcing[i, ...]

        # contact force
        if self.r < np.inf:
            amplitude = np.linalg.norm(q, axis=0)
            gap = self.r - amplitude

            contact_force = np.zeros_like(amplitude)
            for kk, amp in enumerate(np.atleast_1d(amplitude)):
                # avoid division by zero, otherwise the gap is open
                if amp > self.eta:
                    smoothed_gap = 0.5 * (
                        -gap[kk] + np.sqrt(gap[kk] ** 2 + 4 * self.eta**2)
                    )
                    contact_force[kk] = self.normal_stiffness * smoothed_gap

            f[2:, ...] += contact_force * q / amplitude

        return f

    def nontrivial_omega_derivative(self, t=None, x=None):
        if x is None:
            x = self.x
        if t is None:
            t = self.t
        self.check_dimensions(t, x)

        df_domega = np.zeros_like(x)
        df_domega[2, ...] = 2 * self.e * self.omega * np.cos(self.omega * t)
        df_domega[3, ...] = 2 * self.e * self.omega * np.sin(self.omega * t)

        return df_domega

    def closed_form_derivative(self, variable, t=None, x=None):
        """Closed-form derivative of the ODE w.r.t. state `x`.

        Returns an array of shape (n_dof, n_dof, L) where L is the number
        of time samples in `t` / the second dimension of `x`.
        """

        if self.r < np.inf:
            raise NotImplementedError(
                "This is not yet implemented for the Jeffcott rotor example WITH CONTACT."
            )

        if variable != "x":
            raise NotImplementedError("Only variable='x' is implemented here.")

        if x is None:
            x = self.x
        if t is None:
            t = self.t
        self.check_dimensions(t, x)

        # Ensure x has shape (n_dof, L)
        x = np.atleast_2d(x)
        n_dof, L = x.shape

        q = x[:2, :]
        dq = x[2:, :]

        q0 = q[0, :]
        q1 = q[1, :]
        dq0 = dq[0, :]
        dq1 = dq[1, :]

        # factor_nonl = D_it*(q0*dq0 + q1*dq1) + 2*omega_t^2*(q0^2 + q1^2)
        factor_nonl = self.D_it * (q0 * dq0 + q1 * dq1) + 2 * (self.omega_t**2) * (
            q0**2 + q1**2
        )

        Js = np.zeros((n_dof, n_dof, L))

        # f0 = dq0 -> df0/dx = [0,0,1,0]
        Js[0, 2, :] = 1.0
        # f1 = dq1 -> df1/dx = [0,0,0,1]
        Js[1, 3, :] = 1.0

        # Common coefficients
        Dsum = self.D_if + self.D_e

        # dR/dx for f2 (index 2)
        Js[2, 0, :] = (
            -1.0 - factor_nonl - q0 * (self.D_it * dq0 + 4 * (self.omega_t**2) * q0)
        )
        Js[2, 1, :] = -self.omega * self.D_if - q0 * (
            self.D_it * dq1 + 4 * (self.omega_t**2) * q1
        )
        Js[2, 2, :] = -Dsum - self.D_it * q0**2
        Js[2, 3, :] = -self.D_it * q0 * q1

        # dR/dx for f3 (index 3)
        Js[3, 0, :] = self.omega * self.D_if - q1 * (
            self.D_it * dq0 + 4 * (self.omega_t**2) * q0
        )
        Js[3, 1, :] = (
            -1.0 - factor_nonl - q1 * (self.D_it * dq1 + 4 * (self.omega_t**2) * q1)
        )
        Js[3, 2, :] = -self.D_it * q0 * q1
        Js[3, 3, :] = -Dsum - self.D_it * q1**2

        return Js


def init_ode(
    l0=1.2,
    r=0.01,
    D_e=0.1,
    D_if=0.1,
    D_it=0,
    e=5e-4,
    radius_contact=np.inf,
    smoothing=1e-5,
):
    # case Alcorta2023 - p. 5 bottom right
    omega_t = l0 / (np.sqrt(6) * r)
    return Jeffcott2(
        D_e=D_e,
        D_if=D_if,
        D_it=D_it,
        omega_t=omega_t,
        omega=0.1,
        e=e,
        r=radius_contact,
        eta=smoothing,
    )


def init_hbm(ode=None, omega=None, fourier=None, solver=None, subh=True):
    if ode is None:
        ode = init_ode()

    if omega is not None:
        ode.omega = omega

    if fourier is None:
        fourier = Fourier(N_HBM=25, L_DFT=300, n_dof=4, real_formulation=True)

    if solver is None:
        solver = NewtonSolver(verbose=True, max_iterations=50)

    ts = fourier.time_samples(ode.omega)
    x0_samples = np.array(
        [
            ode.e * np.cos(ode.omega * ts),
            ode.e * np.sin(ode.omega * ts),
            -ode.e * ode.omega * np.sin(ode.omega * ts),
            ode.e * ode.omega * np.cos(ode.omega * ts),
        ]
    )
    X0 = fourier.DFT(x0_samples)

    if subh:
        stability_method = KoopmanHillSubharmonic(fourier, tol=1e-4, autonomous=False)
    else:
        stability_method = KoopmanHillProjection(fourier, tol=1e-4, autonomous=False)

    hbm = HBMEquation(
        ode=ode,
        omega=ode.omega,
        fourier=fourier,
        initial_guess=X0,
        period_k=1,
        stability_method=stability_method,
    )

    hbm.residual(update=True)
    solver.solve_equation(equation=hbm, unknown="X")

    return hbm


def continuation(hbm=None, solver=None, **kwargs_cont):
    if hbm is None:
        hbm = init_hbm()

    if solver is None:
        solver = NewtonSolver(verbose=False, max_iterations=8)

    initial_system = EquationSystem(
        equations=[hbm], unknowns=["X"], equation_determining_stability=hbm
    )

    continuation_args = {
        "stepsize": 0.1,
        "stepsize_range": (0.00001, 0.2),
        "initial_direction": 1,
        "verbose": True,
        "num_steps": 30,
    }

    continuation_args.update(kwargs_cont)

    for branch_point in pseudo_arclength_continuator(
        initial_system=initial_system,
        solver=solver,
        continuation_parameter="omega",
        **continuation_args,
    ):
        yield branch_point


def yield_hbm_at_omegas(hbm=None, solver=None, omegas_return=(), **kwargs_cont):
    omegas_return = sorted(omegas_return)
    if solver is None:
        solver = NewtonSolver(verbose=False, max_iterations=15)
    for branch_point in continuation(hbm=hbm, solver=solver, **kwargs_cont):
        if len(omegas_return) == 0:
            break
        if branch_point.omega > omegas_return[0]:
            omega = omegas_return.pop(0)
            hbm = branch_point.equations[0]
            hbm.omega = omega
            solver.solve_equation(equation=hbm, unknown="X")
            yield hbm


def plot_frc(hbm=None, solver=None, path_tikz=None, **kwargs_continuation):
    """Run a frequency response curve analysis for the Jeffcott rotor."""
    frc = [bp for bp in continuation(hbm, solver, **kwargs_continuation)]

    ax = plot_continuation(
        frc,
        plot_fun=lambda point: np.max(
            np.linalg.norm(point.equations[0].x_time()[:2, :], axis=0)
        ),
        marker="x",
    )
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"max radial displacement")

    if path_tikz is not None:
        tikzplotlib.save(path_tikz, axis_width="5cm", axis_height="5cm")

    return ax, frc


def compute_time_solution(
    ode,
    omega,
    x_0=(0, 0, 0, 0),
    num_periods=15,
    points_per_period=200,
    t_0=0,
    t_end=None,
):
    ode.omega = omega
    ode.t = t_0
    ode.x = np.array(x_0)

    shoot = ShootingBVP(ode=ode, T=2 * np.pi / ode.omega)

    if t_end is None:
        t_end = num_periods * shoot.T_solution
        num_points = num_periods * points_per_period + 1
    else:
        num_points = 10 * int((t_end - t_0)) + 1

    t_eval = np.linspace(t_0, t_end, num_points)

    x = shoot.x_time(t_eval=t_eval)
    return t_eval, x


def coords_washingmachine(t, x, omega, radius=0.003):
    phis = np.linspace(0, 2 * np.pi, 100)
    theta = omega * t
    circle_coords = x[:, np.newaxis] + radius * np.array([np.cos(phis), np.sin(phis)])
    e = 0.5 * radius
    S = x[:2] + np.array([e * np.cos(theta), e * np.sin(theta)])
    return circle_coords, S


def animate_washingmachine(
    ts, xs, omegas, length_tail=10, lim=0.005, interval=1, speedup=3
):

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2), dpi=100, constrained_layout=True)
    fig.patch.set_facecolor("#f7f7f9")
    ax.set_facecolor("#fdfdfd")
    ax.set_ylim(-lim, lim)
    ax.set_xlim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", labelsize=14, width=1.2, length=6)
    ax.grid(True, color="#d4d6db", linewidth=0.8, alpha=0.7)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("#2a2a2a")

    radius = lim * 2 / 3

    if np.isscalar(omegas):
        omega = omegas
    else:
        omega = omegas[0]

    circle_x, S = coords_washingmachine(ts[0], xs[:2, 0], omega, radius)

    (tail,) = ax.plot(
        xs[0, :length_tail],
        xs[1, :length_tail],
        color="#0b5fa5",
        linewidth=2.2,
        alpha=0.95,
    )
    (dot,) = ax.plot(
        xs[0, 0],
        xs[1, 0],
        "o",
        color="#0b5fa5",
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=1.0,
    )
    (circle,) = ax.plot(*circle_x, color="#2a2a2a", linestyle="-", linewidth=1.6)
    (com,) = ax.plot(
        *S,
        "o",
        color="#c43c39",
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=0.9,
    )

    ax.set_title(rf"$\omega={omega:.2f}$", fontsize=20, pad=14)
    ax.set_xlabel(r"$y$", fontsize=18)
    ax.set_ylabel(r"$z$", fontsize=18)

    def update(frame):
        idx_start = speedup * frame
        idx_end = min(speedup * frame + length_tail, xs.shape[1] - 1)

        current_omega = omegas if np.isscalar(omegas) else omegas[idx_end]

        ax.set_title(rf"$\omega={current_omega:.2f}$", fontsize=20, pad=14)

        circle_x, S = coords_washingmachine(
            ts[idx_end], xs[:2, idx_end], current_omega, radius
        )

        tail.set_data(xs[0, idx_start:idx_end], xs[1, idx_start:idx_end])
        dot.set_data([xs[0, idx_end]], [xs[1, idx_end]])
        circle.set_data(*circle_x)
        com.set_data(*[[x] for x in S])

        return tail, dot, circle, com

    anim = FuncAnimation(fig, update, frames=len(ts), interval=interval, repeat=True)

    return anim


def animate_phase_portrait(ode, omega, length_tail=10, lim=None):
    t_eval, x = compute_time_solution(
        ode,
        omega=omega,
        num_periods=60,
        points_per_period=min(30, int(length_tail / 1.5)),
    )
    if lim is None:
        lim = 1.2 * np.max(x[1, :])
    anim = animate_washingmachine(
        ts=t_eval,
        xs=x,
        omegas=omega,
        length_tail=length_tail,
        lim=0.005,
        interval=1,
        speedup=3,
    )

    return anim


def plot_time_history(ode, omegas):
    for omega in omegas:
        ode.omega = omega
        ode.t = 0
        ode.x = np.array([0, 0, 0, 0])
        shoot = ShootingBVP(ode=ode, T=2 * np.pi / ode.omega)
        t_eval = np.linspace(0, 10 * shoot.T_solution, 2000)
        x = shoot.x_time(t_eval=t_eval)
        plt.figure()
        plt.plot(x[0, :], x[1, :])
        plt.xlabel("y")
        plt.ylabel("z")
        plt.title(f"Phase portrait for omega={omega}")


def phase_portrait_animations_for_talk(
    ode=None, omegas=(0.5, 1.0, 1.5, 2.5), speedup=5, path=None
):
    if ode is None:
        ode = init_ode()

    anims = []
    for omega in omegas:
        t_eval, x = compute_time_solution(
            ode,
            omega=omega,
            t_end=40 * np.pi,
            x_0=(-0.0006, 0, 0, 0),
        )
        plt.figure()
        plt.plot(x[0, :], x[1, :])
        plt.xlabel("y")
        plt.ylabel("z")
        plt.title(f"Phase portrait for omega={omega}")
        anims.append(
            animate_washingmachine(
                ts=t_eval,
                xs=x,
                omegas=omega,
                length_tail=200,
                lim=0.005,
                interval=1,
                speedup=speedup,
            )
        )
        if path is not None:
            save_animation(
                anims[-1],
                f"{path}_omega_{omega:.2f}.gif",
                fps=30,
            )
    return anims


def analyze_convergence(
    Ns,
    fourier_ref=None,
    omegas=(0.85,),
    ode=None,
    solver=None,
    subh=False,
    animate=False,
):

    # Initialization
    if fourier_ref is None:
        fourier_ref = Fourier(
            N_HBM=np.max(Ns), L_DFT=300, n_dof=4, real_formulation=True
        )
    hbm = init_hbm(ode=ode, omega=0.1, fourier=fourier_ref)

    if np.isscalar(Ns):
        Ns = list(range(1, Ns + 1))

    if solver is None:
        solver = NewtonSolver(verbose=True, max_iterations=10)

    anims = []

    for hbm in yield_hbm_at_omegas(hbm=hbm, omegas_return=omegas, num_steps=1000):
        print(f"Computed HBM solution for with omega={hbm.omega:.4f}")
        hbms = []
        for N in Ns:
            fourier = hbm.fourier.__replace__(N_HBM=N)
            if subh:
                stability_method = KoopmanHillSubharmonic(
                    fourier, tol=1e-4, autonomous=False
                )
            else:
                stability_method = KoopmanHillProjection(
                    fourier, tol=1e-4, autonomous=False
                )
            hbms.append(
                HBMEquation(
                    ode=hbm.ode,
                    omega=hbm.omega,
                    fourier=fourier,
                    initial_guess=fourier.DFT(hbm.x_time()),
                    period_k=1,
                    stability_method=stability_method,
                )
            )

            solver.solve_equation(equation=hbms[-1], unknown="X")
            print(hbms[-1].eigenvalues)

        # FM animation
        if animate:
            plot_phase(hbms[-1])
            ax, anim = animate_floquet_multipliers(hbm_set=hbms)
            ax.set_title(f"Floquet multipliers (omega={hbms[-1].omega})")
            anims.append(anim)

        # FM error
        FM_ref = hbms[-1].eigenvalues
        plt.figure()
        errors = [FM_error(hbm.eigenvalues, FM_ref) for hbm in hbms]
        plt.plot(Ns, errors, "x-")
        plt.xlabel("N_HBM")
        plt.ylabel("Error in Floquet multipliers")
        plt.title(f"FM Convergence (omega = {hbms[-1].omega})")

        # J Fourier coefficients
        J_coeffs = hbms[-1].ode_coeffs()
        labels = (
            ["J_0"]
            + [f"J_c{k}" for k in range(1, N + 1)]
            + [f"J_s{k}" for k in range(1, N + 1)]
        )
        plt.figure()
        plt.bar(
            labels,
            [np.linalg.norm(J_coeffs[:, :, k], 2) for k in range(J_coeffs.shape[2])],
        )
        plt.yscale("log")
        lines = hbms[-1].exponential_decay_parameters(threshold=1e-14)

        plt.title(f"Jacobian FCs (omega={hbms[-1].omega})")
        print(lines)

    return anims


def FM_error(FM, FM_ref):

    # only consider positive imag part
    FM = FM[np.imag(FM) >= 0]
    FM_ref = FM_ref[np.imag(FM_ref) >= 0]

    # sort by absolute value
    FM = FM[np.argsort(np.abs(FM))]
    FM_ref = FM_ref[np.argsort(np.abs(FM_ref))]

    return np.linalg.norm(FM - FM_ref)


def compute_jeffcott_pseudospectrum(hbm, subharmonic=False):

    coeffs = hbm.ode_coeffs()
    J_0 = coeffs[:, :, 0]
    J_c = coeffs[:, :, 2]
    J_s = coeffs[:, :, hbm.fourier.N_HBM + 2]

    J_1 = 0.5 * (J_c - 1j * J_s)

    E = finite_support_error_bound(
        J_0, J_1, hbm.fourier.N_HBM, t=hbm.T_solution, subharmonic=subharmonic, k=2
    )

    print(f"N = {hbm.fourier.N_HBM}: E = {E}")

    Phi_T = hbm.stability_method.fundamental_matrix(t_over_period=1, hbm=hbm)
    FMs, _ = np.linalg.eig(Phi_T)

    z_pseudospectrum = []
    if 1e-14 < E < 10:
        for FM in FMs:
            z_pseudospectrum += [
                np.squeeze(lam)
                for lam in compute_pseudospectrum(
                    Phi_T, epsilon=E, z_init=FM, verbose=False, max_step=2e-4
                )
            ]
            z_pseudospectrum += [np.nan]

    return np.array(z_pseudospectrum)


def animate_FMs_with_guarantee(ode=None, fourier=None, subh=False):
    hbm = init_hbm(ode=ode, fourier=fourier, subh=subh)

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    phis = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(phis), np.sin(phis), "k", linewidth=0.5)
    pspec = compute_jeffcott_pseudospectrum(hbm, subharmonic=subh)

    ax.plot(np.real(pspec), np.imag(pspec), "r")


if __name__ == "__main__":

    ode = init_ode(e=5e-4, D_it=0.1, r=0.01, radius_contact=np.inf, smoothing=1e-3)
    ode.omega = 0.85

    N_HBM = 18

    hbm = init_hbm(
        ode=ode, fourier=Fourier(N_HBM=N_HBM, L_DFT=1024, n_dof=4), omega=0.85
    )

    # _, frc = plot_frc(hbm=hbm)
    anims = []
    # # anims = phase_portrait_animations_for_talk(ode)
    # anims.append(animate_floquet_multipliers(hbm_set=frc, interval=200))
    anims += analyze_convergence(
        Ns=10, omegas=(0.85,), ode=ode, subh=False, animate=True
    )

    ode.omega = 1.6
    animate_FMs_with_guarantee(ode=ode, fourier=hbm.fourier, subh=False)
    animate_FMs_with_guarantee(ode=ode, fourier=hbm.fourier, subh=True)

    plt.show()
