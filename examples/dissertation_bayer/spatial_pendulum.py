import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from scipy.integrate import solve_ivp


from skhippr.odes.spatialpendulum import (
    SpatialPendulumWithoutConstraints,
    SpatialPendulumWithMassInverted,
    SpatialPendulumWithConstraints,
)

from skhippr.equations.EquationSystem import EquationSystem
from skhippr.cycles.hbm import HBMEquationDAE, HBMEquation
from skhippr.solvers.newton import NewtonSolver
from skhippr.Fourier import Fourier

from skhippr.stability.KoopmanHillProjection import (
    KoopmanHillSubharmonic,
    KoopmanHillDAE,
    KoopmanHillDAESubharmonic,
)


def main():
    initial_angles_deg = np.array([15, 0, 0])
    pend = SpatialPendulumWithoutConstraints(
        t=0,
        angles=np.pi / 180 * initial_angles_deg,
        d_angles=[0, 0, 0],
        shape_cuboid=[1, 2, 3],
        density=0.5,
        delta=2,
        epsilon=1,
        omega=1,
        damping=5,
        spring=1,
        stability_method=None,
    )

    pend_inv = SpatialPendulumWithMassInverted(
        t=0,
        angles=np.pi / 180 * initial_angles_deg,
        d_angles=[0, 0, 0],
        shape_cuboid=[1, 2, 3],
        density=0.5,
        delta=2,
        epsilon=1,
        omega=1,
        damping=5,
        spring=1,
        stability_method=None,
    )

    fourier = Fourier(n_dof=pend.n_dof, N_HBM=10, L_DFT=1024)
    t_integrate = (0, 10 * np.pi)
    ode_res = solve_ivp(
        fun=pend_inv.dynamics,
        t_span=t_integrate,
        y0=pend.x,
        # t_eval=np.linspace(*t_integrate, num=fourier.L_DFT, endpoint=False),
    )

    _, ax = plt.subplots(3, 1)
    for i, label in enumerate(["alpha", "beta", "gamma"]):
        ax[i].plot(ode_res.t, ode_res.y[i, :], label=label)
        ax[i].set_xlabel("Time")
        ax[i].set_ylabel("Angle (rad)")
        ax[i].legend()

    print("integrated to converge.")
    t_peri = (0, 2 * np.pi)
    ode_res = solve_ivp(
        fun=pend_inv.dynamics,
        t_span=t_integrate,
        y0=ode_res.y[:, -1],
        t_eval=np.linspace(*t_peri, num=fourier.L_DFT, endpoint=False),
    )
    print("Integrated for peri. sol.")

    anims = []
    anims.append(animate_pendulum(pend_inv, ode_res.t, ode_res.y))

    # find periodic solution - standard case
    fourier = Fourier(n_dof=pend.n_dof, N_HBM=10, L_DFT=1024)
    initial_guess = fourier.DFT(ode_res.y)
    hbm_inv = HBMEquation(
        ode=pend_inv,
        omega=pend.omega,
        fourier=fourier,
        initial_guess=initial_guess,
        stability_method=KoopmanHillSubharmonic(fourier),
    )
    solver = NewtonSolver(verbose=True, max_iterations=20)
    solver.solve_equation(equation=hbm_inv, unknown="X")
    print("solved inv equation.")

    anims.append(
        animate_pendulum(
            pend, hbm_inv.fourier.time_samples(omega=pend.omega), hbm_inv.x_time()
        )
    )

    hbm = HBMEquationDAE(
        dae=pend,
        omega=pend.omega,
        fourier=fourier,
        initial_guess=hbm_inv.X + 0.001 * np.random.randn(*hbm_inv.X.shape),
        stability_method=KoopmanHillDAESubharmonic(fourier),
    )
    solver = NewtonSolver(verbose=True, max_iterations=20)
    solver.solve_equation(equation=hbm, unknown="X")
    print("solved ang equation.")

    anims.append(
        animate_pendulum(
            pend, hbm_inv.fourier.time_samples(omega=pend.omega), hbm_inv.x_time()
        )
    )

    print("Preparing constrained initial condition")
    x_angle = hbm.x_time()
    ts = hbm.fourier.time_samples(omega=pend.omega)
    I_r_OS = np.zeros_like(x_angle[:3, :])
    I_v_S = np.zeros_like(x_angle[3:, :])
    for i in range(x_angle.shape[1]):
        I_r_OS[:, i] = pend.I_r_OS(t=ts[i], angles=x_angle[:3, i])
        I_v_S[:, i] = pend.I_v_S(
            t=ts[i], angles=x_angle[:3, i], d_angles=x_angle[3:, i]
        )

    fourier_constr = fourier.__replace__(n_dof=15)
    # 3 angles, 3 COM positions, 3 angular velocities, 3 COM velocities, 3 lambdas
    x = 0.001 * np.random.randn(fourier_constr.n_dof, fourier_constr.L_DFT)
    x[:3, :] = x_angle[:3, :]
    x[3:6, :] = I_r_OS
    x[6:9, :] = x_angle[3:, :]
    x[9:12, :] = I_v_S
    # x[12: , :] = lambdas initialized at zero
    X_init = fourier_constr.DFT(x)

    pend_constr = SpatialPendulumWithConstraints(
        t=0,
        angles=np.pi / 180 * initial_angles_deg,
        d_angles=[0, 0, 0],
        I_r_OS=[0, 0, 0],
        I_v_S=[0, 0, 0],
        lam=[0, 0, 0],
        shape_cuboid=[pend.a, pend.b, pend.c],
        density=pend.density,
        delta=pend.delta,
        epsilon=pend.epsilon,
        omega=pend.omega,
        damping=pend.damping,
        spring=pend.spring,
    )

    hbm_constr = HBMEquationDAE(
        dae=pend_constr,
        omega=pend_constr.omega,
        fourier=fourier_constr,
        initial_guess=X_init,
        stability_method=KoopmanHillDAESubharmonic(fourier),
    )
    solver = NewtonSolver(verbose=True, max_iterations=20)
    solver.solve_equation(equation=hbm_constr, unknown="X")
    print("solved constr equation.")

    animate_pendulum(
        pend_constr,
        hbm_constr.fourier.time_samples(omega=pend_constr.omega),
        hbm_constr.x_time()[:3, :],
    )

    _, ax_FM = plt.subplots(1, 1)
    ax_FM.plot(
        np.real(hbm_inv.eigenvalues), np.imag(hbm_inv.eigenvalues), "x", label="inv"
    )
    ax_FM.plot(np.real(hbm.eigenvalues), np.imag(hbm.eigenvalues), "+", label="ang")
    ax_FM.plot(np.real(hbm.eigenvalues), np.imag(hbm.eigenvalues), "s", label="constr")
    ax_FM.legend()

    return anims


def only_constr():

    initial_angles_deg = np.array([125, 150, 20])

    pend = SpatialPendulumWithoutConstraints(
        t=0,
        angles=np.pi / 180 * initial_angles_deg,
        d_angles=[0, 0, 0],
        shape_cuboid=[1, 2, 3],
        density=0.5,
        delta=2,
        epsilon=1,
        omega=1,
        damping=5,
        spring=1,
    )

    pend_constr = SpatialPendulumWithConstraints(
        t=0,
        angles=np.pi / 180 * initial_angles_deg,
        d_angles=[0, 0, 0],
        I_r_OS=[0, 0, 0],
        I_v_S=[0, 0, 0],
        lam=[0, 0, 0],
        shape_cuboid=[pend.a, pend.b, pend.c],
        density=pend.density,
        delta=pend.delta,
        epsilon=pend.epsilon,
        omega=pend.omega,
        damping=pend.damping,
        spring=1,
    )

    fourier = Fourier(n_dof=pend_constr.n_dof, N_HBM=5, L_DFT=1024)
    initial_guess = np.zeros(fourier.n_dof * (2 * fourier.N_HBM + 1))

    hbm_constr = HBMEquationDAE(
        dae=pend_constr,
        omega=pend_constr.omega,
        fourier=fourier,
        initial_guess=initial_guess,
        stability_method=None,
    )
    solver = NewtonSolver(verbose=True, max_iterations=20)
    solver.solve_equation(equation=hbm_constr, unknown="X")


def animate_pendulum(pend, t, angles, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # ax.view_init(elev=20, azim=170, roll=-90)

    a, b, c = pend.a, pend.b, pend.c
    K_edges = 0.5 * np.array(
        [
            [a, -a, -a, a, a, a, a, a, -a, -a, -a, -a],
            [b, b, b, b, b, -b, -b, b, b, -b, -b, b],
            [c, c, -c, -c, c, c, -c, -c, -c, -c, c, c],
        ]
    )

    # Pre-compute all positions
    positions = []
    edges = []
    suspensions = []

    for i in range(angles.shape[1]):
        I_r_OS = pend.I_r_OS(t=t[i], angles=angles[:3, i])
        A_IK = pend.A_IK(angles=angles[:3, i])

        positions.append(I_r_OS)
        suspensions.append(I_r_OS + A_IK @ pend.K_r_SP)
        edges.append(I_r_OS[:, np.newaxis] + A_IK @ K_edges)

    # Initialize plot data
    positions = np.array(positions)
    line_x, line_y, line_z = [], [], []
    (line,) = ax.plot(line_x, line_y, line_z, "b-", linewidth=2)
    (point,) = ax.plot([], [], [], "ro", markersize=8)
    (suspension_point,) = ax.plot([], [], [], "ko", markersize=8)
    (cube,) = ax.plot([], [], [], "k-", linewidth=1)

    # Set axis limits
    all_x, all_y, all_z = positions[:, 0], positions[:, 1], positions[:, 2]
    ax.set_xlim(all_x.min() - 1, all_x.max() + 1)
    ax.set_ylim(all_y.min() - 1, all_y.max() + 1)
    ax.set_zlim(all_z.min() - 1, all_z.max() + 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Spatial Pendulum Center of Mass Trajectory")

    def update(frame):
        line_x.append(positions[frame, 0])
        line_y.append(positions[frame, 1])
        line_z.append(positions[frame, 2])

        line.set_data(line_x, line_y)
        line.set_3d_properties(line_z)

        point.set_data([positions[frame, 0]], [positions[frame, 1]])
        point.set_3d_properties([positions[frame, 2]])

        suspension_point.set_data([suspensions[frame][0]], [suspensions[frame][1]])
        suspension_point.set_3d_properties([suspensions[frame][2]])

        cube.set_data(edges[frame][0, :], edges[frame][1, :])
        cube.set_3d_properties(edges[frame][2, :])

        return line, point, suspension_point, cube

    anim = FuncAnimation(fig, update, frames=len(positions), interval=10, blit=True)
    return anim


if __name__ == "__main__":
    only_constr()
    # anims = main()
    plt.show()
