"""Jeffcott rotor analysis for thesis presentation."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tikzplotlib
from scipy.integrate import solve_ivp

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
    plot_floquet_exponent_continuation,
    plot_floquet_multiplier_continuation,
)
from skhippr.visualization.cycles import (
    animate_floquet_exponents,
    animate_floquet_multipliers,
    plot_phase,
)

from skhippr.visualization.data_export import save_tikz, save_animation


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


def main(ode=None, fourier=None, omegas_return=()):
    """Run a frequency response curve analysis for the Jeffcott rotor."""
    if ode is None:
        ode = init_ode()
    ode.omega = 0.1

    if fourier is None:
        fourier = Fourier(N_HBM=25, L_DFT=300, n_dof=4, real_formulation=True)

    omegas_return = list(omegas_return)
    hbms_return = []

    stability_method = KoopmanHillSubharmonic(fourier, tol=1e-4, autonomous=False)
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
    solver.verbose = False
    solver.max_iterations = 10

    initial_system = EquationSystem(
        equations=[hbm], unknowns=["X"], equation_determining_stability=hbm
    )

    frc_forward: list[BranchPoint] = []

    for branch_point in pseudo_arclength_continuator(
        initial_system=initial_system,
        solver=solver,
        stepsize=0.1,
        stepsize_range=(0.00001, 0.2),
        continuation_parameter="omega",
        initial_direction=1,
        verbose=True,
        num_steps=30,
    ):
        frc_forward.append(branch_point)

        if len(omegas_return) > 0 and branch_point.omega > omegas_return[0]:
            hbms_return.append(branch_point.equations[0])
            omegas_return.pop(0)

        if branch_point.omega > 2.4:
            pass

        if branch_point.omega > 12 or branch_point.omega < 0.1:
            break

    ax = plot_continuation(
        frc_forward,
        plot_fun=lambda point: np.max(
            np.linalg.norm(point.equations[0].x_time()[:2, :], axis=0)
        ),
        marker="x",
    )
    # ax = plot_continuation(
    #     frc_backward,
    #     ax=ax,
    #     plot_fun=lambda point: np.max(
    #         np.linalg.norm(point.equations[0].x_time()[:2, :], axis=0)
    #     ),
    #     marker="x",
    #     color="yellow",
    # )
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"max radial displacement")

    tikzplotlib.save("jeffcott.tikz", axis_width="5cm", axis_height="5cm")

    _, animation1 = animate_floquet_multipliers(hbm_set=frc_forward)
    # _, animation2 = animate_floquet_exponents(hbm_set=frc)

    # animation1 = None
    animation2 = None

    return ode, [animation1, animation2], hbms_return


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
    ode.t = 0
    ode.x = np.array(x_0)

    shoot = ShootingBVP(ode=ode, T=2 * np.pi / ode.omega)

    if t_end is None:
        t_end = num_periods * shoot.T_solution
        num_points = num_periods * points_per_period + 1
    else:
        num_points = 10 * int((t_end - t_0)) + 1

    t_eval = np.linspace(0, t_end, num_points)

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
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"

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


def animate_phase_portrait(ode, omega, length_tail=10, lim=None, path=None):
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


def compute_frequency_sweep(ode, t_0=0, t_end=300, omega_start=0.1, omega_end=3):
    ts = np.linspace(t_0, t_end, (t_end - t_0) * 3)
    x_0 = np.array([0.001, 0, 0, 0])

    def dynamics(t, x):
        ode.omega = omega_sweep(t, t_0, t_end, omega_start, omega_end)
        return ode.dynamics(t, x)

    sol = solve_ivp(
        fun=dynamics,
        t_span=(t_0, t_end),
        y0=x_0,
        t_eval=ts,
    )
    omegas = [omega_sweep(t, t_0, t_end, omega_start, omega_end) for t in sol.t]

    return sol.t, sol.y, np.array(omegas)


def omega_sweep(t, t_0, t_end, omega_start, omega_end):
    return omega_start + (omega_end - omega_start) * omega_sweep_function(
        t - t_0
    ) / omega_sweep_function(t_end - t_0)


def omega_sweep_function(delta_t):
    """Modify the shape of the sweep function. Must be zero at delta_t=0 and increase monotonously."""
    return np.sqrt(delta_t)
    # return delta_t


def animate_frequency_sweep(
    ode, t_0=0, t_end=2000, omega_start=0.1, omega_end=3, length_tail=100
):
    t, x, omegas = compute_frequency_sweep(ode, t_0, t_end, omega_start, omega_end)

    anim = animate_washingmachine(
        t, x, omegas, length_tail=length_tail, lim=0.005, interval=1
    )
    return anim


def plot_omega_sweep(t_0, t_end, omega_start, omega_end):
    ts = np.linspace(t_0, t_end, 1000)
    omegas = [omega_sweep(t, t_0, t_end, omega_start, omega_end) for t in ts]
    plt.figure()
    plt.plot(ts, omegas)
    plt.xlabel("Time")
    plt.ylabel("Omega")
    plt.title("Frequency sweep over time")


def frc_animations_for_talk(omegas=(0.5, 1.0, 1.5, 2.5)):
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
                speedup=5,
            )
        )
        save_animation(
            anims[-1],
            f"jeffcott_fast_omega_{omega:.2f}.gif",
            fps=100,
        )
    return anims


def analyze_convergence(Ns=35, omega=0.85, ode=None):

    if np.isscalar(Ns):
        Ns = list(range(1, Ns + 1))

    fourier = Fourier(N_HBM=np.max(Ns), L_DFT=300, n_dof=4, real_formulation=True)
    _, anims, hbms = main(ode=ode, fourier=fourier, omegas_return=[omega])

    solver = NewtonSolver(verbose=True, max_iterations=50)
    ax_phase = None
    # ts = fourier.time_samples(ode.omega)
    # x0_samples = 10 * np.array(
    #     [
    #         ode.e * np.cos(ode.omega * ts),
    #         ode.e * np.sin(ode.omega * ts),
    #         -ode.e * ode.omega * np.sin(ode.omega * ts),
    #         ode.e * ode.omega * np.cos(ode.omega * ts),
    #     ]
    # )

    try:

        hbm: HBMEquation = hbms[0]

    except IndexError:
        print("Did not get complete branch")
        return anims

    hbms = []

    for N in Ns:
        fourier = hbm.fourier.__replace__(N_HBM=N)
        hbms.append(
            HBMEquation(
                ode=hbm.ode,
                omega=hbm.omega,
                fourier=fourier,
                initial_guess=fourier.DFT(hbm.x_time()),
                period_k=1,
                stability_method=KoopmanHillSubharmonic(
                    fourier, tol=1e-4, autonomous=False
                ),
            )
        )

        solver.solve_equation(equation=hbms[-1], unknown="X")
        print(hbms[-1].eigenvalues)

    # Phase plot
    ax_phase = plot_phase(hbms[-1], ax=ax_phase)

    # FM animation
    anims.append(animate_floquet_multipliers(hbm_set=hbms))

    # FM error
    FM_ref = hbms[-1].eigenvalues
    plt.figure()
    errors = [FM_error(hbm.eigenvalues, FM_ref) for hbm in hbms]
    plt.plot(Ns, errors, "x-")
    plt.xlabel("N_HBM")
    plt.ylabel("Error in Floquet multipliers")
    plt.title("Convergence of Floquet multipliers with N_HBM")

    # J Fourier coefficients
    J_coeffs = hbms[-1].ode_coeffs()
    labels = (
        ["J_0"]
        + [f"J_c{k}" for k in range(1, N + 1)]
        + [f"J_s{k}" for k in range(1, N + 1)]
    )
    plt.figure()
    plt.bar(
        labels, [np.linalg.norm(J_coeffs[:, :, k], 2) for k in range(J_coeffs.shape[2])]
    )
    plt.yscale("log")
    lines = hbm.exponential_decay_parameters(threshold=1e-14)
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


def animate_FMs_with_guarantee(ode=None, fourier=None, subh=False):
    if ode is None:
        ode = init_ode()

    ode.omega = 0.1

    if fourier is None:
        fourier = Fourier(N_HBM=25, L_DFT=300, n_dof=4, real_formulation=True)

    if subh:
        stability_method = KoopmanHillSubharmonic(fourier, tol=1e-4, autonomous=False)
    else:
        stability_method = KoopmanHillProjection(fourier, tol=1e-4, autonomous=False)
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
    solver.verbose = False
    solver.max_iterations = 10

    initial_system = EquationSystem(
        equations=[hbm], unknowns=["X"], equation_determining_stability=hbm
    )

    frc_forward: list[BranchPoint] = []

    for branch_point in pseudo_arclength_continuator(
        initial_system=initial_system,
        solver=solver,
        stepsize=0.1,
        stepsize_range=(0.00001, 0.2),
        continuation_parameter="omega",
        initial_direction=1,
        verbose=True,
        num_steps=30,
    ):
        frc_forward.append(branch_point)

        if branch_point.omega > 3:
            break

    plot_continuation(
        frc_forward,
        plot_fun=lambda point: np.max(
            np.linalg.norm(point.equations[0].x_time()[:2, :], axis=0)
        ),
        marker="x",
    )
    anim = animate_floquet_multipliers(hbm_set=frc_forward, interval=200)
    return anim


if __name__ == "__main__":

    ode = init_ode(e=5e-4, D_it=0.1, r=0.01, radius_contact=np.inf, smoothing=1e-3)
    anim = animate_FMs_with_guarantee(ode=ode, subh=True)
    # anims = analyze_convergence(ode=ode, Ns=10, omega=1.5)
    # anims = frc_animations_for_talk()

    # anim = animate_phase_portrait(ode, omega=omega, length_tail=40)
    # plot_omega_sweep(t_0=0, t_end=t_end, omega_start=0.5, omega_end=3)
    # anim2 = animate_frequency_sweep(
    #     ode,
    #     t_end=t_end,
    #     omega_start=0,
    #     omega_end=2.5,
    #     length_tail=200,
    # )
    # plot_time_history(ode, omegas=[0.5, 1.0, 1.9, 2.5])
    plt.show()
