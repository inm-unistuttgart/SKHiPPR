import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import tikzplotlib

from scipy.integrate import solve_ivp


from skhippr.odes.spatialpendulum import (
    SpatialPendulumWithoutConstraints,
    SpatialPendulumWithMassInverted,
    SpatialPendulumWithConstraints,
)

from skhippr.equations.EquationSystem import EquationSystem
from skhippr.cycles.hbm import HBMEquationDAE, HBMEquation
from skhippr.solvers.newton import NewtonSolver, ScipyRootSolver
from skhippr.solvers.continuation import pseudo_arclength_continuator
from skhippr.Fourier import Fourier

from skhippr.stability.KoopmanHillProjection import (
    KoopmanHillProjection,
    KoopmanHillSubharmonic,
    KoopmanHillDAE,
    KoopmanHillDAESubharmonic,
)


def main():
    # initial_angles_deg = np.array([0, 0, 0])
    initial_angles_deg = 180 / np.pi * np.array([-0.01240729, -0.25595918, 0.54776127])
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
    t_integrate = (0, 2 * np.pi)
    ode_res = solve_ivp(
        fun=pend_inv.dynamics,
        t_span=t_integrate,
        y0=pend.x,
        # t_eval=np.linspace(*t_integrate, num=fourier.L_DFT, endpoint=False),
    )

    print(f"End angles: {ode_res.y[:3, -1]}")

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

    # anims = []
    # anims.append(animate_pendulum(pend_inv, ode_res.t, ode_res.y))

    # # find periodic solution - standard case
    # fourier = Fourier(n_dof=pend.n_dof, N_HBM=10, L_DFT=1024)
    # initial_guess = fourier.DFT(ode_res.y)
    # hbm_inv = HBMEquation(
    #     ode=pend_inv,
    #     omega=pend.omega,
    #     fourier=fourier,
    #     initial_guess=initial_guess,
    #     stability_method=KoopmanHillSubharmonic(fourier),
    # )
    # solver = NewtonSolver(verbose=True, max_iterations=20)
    # solver.solve_equation(equation=hbm_inv, unknown="X")
    # print("solved inv equation.")

    # anims.append(
    #     animate_pendulum(
    #         pend, hbm_inv.fourier.time_samples(omega=pend.omega), hbm_inv.x_time()
    #     )
    # )

    # hbm = HBMEquationDAE(
    #     dae=pend,
    #     omega=pend.omega,
    #     fourier=fourier,
    #     initial_guess=hbm_inv.X + 0.001 * np.random.randn(*hbm_inv.X.shape),
    #     stability_method=KoopmanHillDAESubharmonic(fourier),
    # )
    # solver = NewtonSolver(verbose=True, max_iterations=20)
    # solver.solve_equation(equation=hbm, unknown="X")
    # print("solved ang equation.")

    # anims.append(
    #     animate_pendulum(
    #         pend, hbm_inv.fourier.time_samples(omega=pend.omega), hbm_inv.x_time()
    #     )
    # )

    print("Preparing constrained initial condition")

    solver = ScipyRootSolver(
        verbose=True, max_iterations=20000, use_fprime=False, method="broyden1"
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
        spring=pend.spring,
    )
    fourier_constr = fourier.__replace__(n_dof=pend_constr.n_dof)
    x_angle = ode_res.y
    ts = ode_res.t
    I_r_OS = np.zeros_like(x_angle[:3, :])
    I_v_S = np.zeros_like(x_angle[3:, :])
    lambdas = np.zeros_like(x_angle[:3, :])

    x = np.zeros((fourier_constr.n_dof, fourier_constr.L_DFT))
    for i in range(x_angle.shape[1]):
        I_r_OS[:, i] = pend.I_r_OS(t=ts[i], angles=x_angle[:3, i])
        I_v_S[:, i] = pend.I_v_S(
            t=ts[i], angles=x_angle[:3, i], d_angles=x_angle[3:, i]
        )
        x[:, i] = np.hstack(
            (x_angle[:3, i], I_r_OS[:, i], x_angle[3:, i], I_v_S[:, i], lambdas[:, i])
        )

        if i >= 1:

            a_S = (I_v_S[:, i] - I_v_S[:, i - 1]) / (ts[i] - ts[i - 1])
            dd_angles = (x_angle[3:, i] - x_angle[3:, i - 1]) / (ts[i] - ts[i - 1])
            # W = I # W = pend_constr.W_constraints(angles=x_angle[:3, i])[3:6, :]
            # M = pend_constr.M_small(t=ts[i], x=x)[9:12, 9:12]
            M = pend_constr.total_mass * np.eye(3)
            h = pend_constr.dynamics(t=ts[i], x=x[:, i])[9:12]
            # M*a_S = h + W @ lambdas[:, i]
            x[12:, i] = M @ a_S - h

            d_x = np.hstack((x_angle[3:, i], I_v_S[:, i], dd_angles, a_S, x[12:, i]))

            dynamics_error = pend_constr.M_small(
                t=ts[i], x=x[:, i]
            ) @ d_x - pend_constr.dynamics(t=ts[i], x=x[:, i])

            x_inv = ode_res.y[:, i]
            dx_inv = np.hstack((x_inv[3:,], dd_angles))
            dynamics_inv_error = pend_inv.M_small(
                t=ts[i], x=x_inv
            ) @ dx_inv - pend_inv.dynamics(t=ts[i], x=x_inv)

    # 3 angles, 3 COM positions, 3 angular velocities, 3 COM velocities, 3 lambdas
    x = 0.001 * np.random.randn(fourier_constr.n_dof, fourier_constr.L_DFT)
    x[:3, :] = x_angle[:3, :]
    x[3:6, :] = I_r_OS
    x[6:9, :] = x_angle[3:, :]
    x[9:12, :] = I_v_S
    x[12:, :] = lambdas
    X_init = fourier_constr.DFT(x)

    hbm_constr = HBMEquationDAE(
        dae=pend_constr,
        omega=pend_constr.omega,
        fourier=fourier_constr,
        initial_guess=X_init,
        stability_method=KoopmanHillDAESubharmonic(fourier),
    )
    print(hbm_constr.residual_function())
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
    # ax_FM.plot(np.real(hbm.eigenvalues), np.imag(hbm.eigenvalues), "+", label="ang")
    ax_FM.plot(
        np.real(hbm_constr.eigenvalues),
        np.imag(hbm_constr.eigenvalues),
        "s",
        label="constr",
    )
    ax_FM.legend()

    return anims


def only_constr():

    fourier = Fourier(n_dof=15, N_HBM=10, L_DFT=1024)

    a = 1
    b = 2
    c = 3

    pend = SpatialPendulumWithConstraints(
        t=0,
        angles=np.array([0, -0.27055, 0.588]),
        d_angles=[0, 0, 0],
        I_r_OS=[0, 0, 0],
        I_v_S=[0, 0, 0],
        lam=[0, 0, 0],
        shape_cuboid=[a, b, c],
        density=0.5,
        delta=2,
        epsilon=1,
        omega=1,
        damping=5,
        spring=0.0001,
    )

    l = 0.5 * np.sqrt(a**2 + b**2 + c**2)

    # Verify that pendulum is vertical
    I_r_SP = pend.A_IK() @ pend.K_r_SP
    print("I_r_SP in inertial frame: ", I_r_SP)
    print("error to expected position:", np.linalg.norm(I_r_SP - np.array([0, 0, l])))

    angles = np.array([0, -0.27055, 0.588])
    angles = angles[:, np.newaxis] @ np.ones((1, fourier.L_DFT))
    dangles = np.zeros_like(angles)

    delta = 2
    epsilon = 1
    omega = 1

    t = fourier.time_samples(omega=omega)
    e = delta + epsilon * np.cos(omega * t)
    de = -epsilon * omega * np.sin(omega * t)
    dde = -epsilon * (omega**2) * np.cos(omega * t)

    r_OS = np.zeros_like(angles)
    r_OS[2, :] = -l + e

    v_S = np.zeros_like(angles)
    v_S[2, :] = de

    lambdas = np.zeros_like(angles)
    lambdas[2, :] = pend.total_mass * (9.81 + dde)

    x = np.vstack((angles, r_OS, dangles, v_S, lambdas))

    hbm = HBMEquationDAE(
        dae=pend,
        omega=pend.omega,
        fourier=fourier,
        initial_guess=fourier.DFT(x),
        stability_method=KoopmanHillDAE(fourier),
    )
    res = hbm.residual_function()
    idx_max = np.argmax(np.abs(res))
    # print(res[idx_max])
    # print(idx_max)
    # print(np.sort(res))

    solver = NewtonSolver(verbose=True, max_iterations=5)
    solver.solve_equation(equation=hbm, unknown="X")

    print(np.abs(hbm.eigenvalues))


def compare_FMs():

    a = 1
    b = 2
    c = 3
    l = 0.5 * np.sqrt(a**2 + b**2 + c**2)

    delta = 2
    epsilon = 1
    omega = 1

    fourier = Fourier(n_dof=6, N_HBM=10, L_DFT=1024)
    fourier_constr = fourier.__replace__(n_dof=15)
    solver = NewtonSolver(verbose=True, max_iterations=20)

    pend_constr = SpatialPendulumWithConstraints(
        t=0,
        angles=np.array([0, -0.27055, 0.588]),
        d_angles=[0, 0, 0],
        I_r_OS=[0, 0, 0],
        I_v_S=[0, 0, 0],
        lam=[0, 0, 0],
        shape_cuboid=[a, b, c],
        density=0.5,
        delta=delta,
        epsilon=epsilon,
        omega=omega,
        damping=0,
        spring=0.0001,
    )

    pend = SpatialPendulumWithoutConstraints(
        t=0,
        angles=pend_constr.angles,
        d_angles=[0, 0, 0],
        shape_cuboid=[a, b, c],
        density=pend_constr.density,
        delta=pend_constr.delta,
        epsilon=pend_constr.epsilon,
        omega=pend_constr.omega,
        damping=pend_constr.damping,
        spring=pend_constr.spring,
        stability_method=None,
    )

    pend_inv = SpatialPendulumWithMassInverted(
        t=0,
        angles=pend_constr.angles,
        d_angles=[0, 0, 0],
        shape_cuboid=[a, b, c],
        density=pend_constr.density,
        delta=pend_constr.delta,
        epsilon=pend_constr.epsilon,
        omega=pend_constr.omega,
        damping=pend_constr.damping,
        spring=pend_constr.spring,
        stability_method=None,
    )

    angles = np.array([0, -0.27055, 0.588])
    angles = angles[:, np.newaxis] @ np.ones((1, fourier.L_DFT))
    dangles = np.zeros_like(angles)

    delta = 2
    epsilon = 1
    omega = 1

    t = fourier.time_samples(omega=omega)
    e = delta + epsilon * np.cos(omega * t)
    de = -epsilon * omega * np.sin(omega * t)
    dde = -epsilon * (omega**2) * np.cos(omega * t)

    r_OS = np.zeros_like(angles)
    r_OS[2, :] = -l + e

    v_S = np.zeros_like(angles)
    v_S[2, :] = de

    lambdas = np.zeros_like(angles)
    lambdas[2, :] = pend.total_mass * (9.81 + dde)

    x_constr = np.vstack((angles, r_OS, dangles, v_S, lambdas))
    x = np.vstack((angles, dangles))

    hbm_constr = HBMEquationDAE(
        dae=pend_constr,
        omega=pend_constr.omega,
        fourier=fourier_constr,
        initial_guess=fourier_constr.DFT(x_constr),
        stability_method=KoopmanHillDAE(fourier_constr),
    )

    hbm_inv = HBMEquation(
        ode=pend_inv,
        omega=pend_inv.omega,
        fourier=fourier,
        initial_guess=fourier.DFT(x),
        stability_method=KoopmanHillProjection(fourier),
    )

    hbm = HBMEquationDAE(
        dae=pend,
        omega=pend.omega,
        fourier=fourier,
        initial_guess=fourier.DFT(x),
        stability_method=KoopmanHillDAE(fourier),
    )

    _, ax = plt.subplots(1, 1)
    for eq, label, marker in zip(
        [hbm_constr, hbm_inv, hbm], ["constr", "inv", "ang"], ["x", "+", "o"]
    ):
        print(f"Solving {label} equation")
        solver.solve_equation(equation=eq, unknown="X")
        print(f"Floquet mutlipliers:")
        for FM in np.sort(np.abs(eq.eigenvalues)):
            print(FM)
        ax.plot(np.real(eq.eigenvalues), np.imag(eq.eigenvalues), marker, label=label)

    tikzplotlib.save("plots/spatial_pendulum_FMs.tex")

    return hbm_constr, hbm_inv, hbm


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


def periodic_initial_condition():

    # NOT QUITE on the peri. sol, just for solver init
    initial_angles = np.array([-0.01240729, -0.25595918, 0.54776127])
    # ODE formulation to integrate
    pend = SpatialPendulumWithMassInverted(
        t=0,
        angles=initial_angles,
        d_angles=[0, 0, 0],
        shape_cuboid=[
            1,
            2,
            3,
        ],  # diagonal elements of inertia tensor in body fixed frame
        density=0.5,
        delta=2,
        epsilon=1,
        omega=1,
        damping=5,
        spring=1,
        stability_method=None,
    )

    # integrate over a period to get close to peri. sol
    fourier = Fourier(n_dof=pend.n_dof, N_HBM=10, L_DFT=1024)
    t_integrate = (0, 2 * np.pi)
    ode_res = solve_ivp(
        fun=pend.dynamics,
        t_span=t_integrate,
        y0=pend.x,
        t_eval=np.linspace(*t_integrate, num=fourier.L_DFT, endpoint=False),
    )

    # Solve HBM to obtain periodic solution
    fourier = Fourier(n_dof=pend.n_dof, N_HBM=10, L_DFT=1024)
    initial_guess = fourier.DFT(ode_res.y)
    hbm_inv = HBMEquation(
        ode=pend,
        omega=pend.omega,
        fourier=fourier,
        initial_guess=initial_guess,
        stability_method=None,
    )

    print("Newton solution procedure")
    solver = NewtonSolver(verbose=True, max_iterations=20)
    solver.solve_equation(equation=hbm_inv, unknown="X")
    print("solved inv equation.")

    ### Print initial condition
    print(
        "---------------------   Initial conditions for periodic solution -----------------"
    )
    print(
        f"Excitation : ({pend.delta} + {pend.epsilon} * cos({pend.omega} * t)) in z direction of inertial frame"
    )
    print("point on body that is excited in body-fixed frame: K_r_SP = ", pend.K_r_SP)
    print("mass of body: ", pend.total_mass)
    print("inertia tensor in body-fixed frame: ", pend.K_inertia)
    print("Kardan angles [alpha, beta, gamma] (rad):", hbm_inv.x_time()[:3, 0])
    print(
        "Angular velocities [d_alpha, d_beta, d_gamma] (rad/s):",
        hbm_inv.x_time()[3:, 0],
    )
    print(
        "rotation in body-fixed frame K_Omega",
        pend.K_Omega(angles=hbm_inv.x_time()[:3, 0], d_angles=hbm_inv.x_time()[3:, 0]),
    )
    print(
        "I_r_OS in inertial frame: ", pend.I_r_OS(t=0, angles=hbm_inv.x_time()[:3, 0])
    )
    print(
        "I_v_S in inertial frame: ",
        pend.I_v_S(
            t=0, angles=hbm_inv.x_time()[:3, 0], d_angles=hbm_inv.x_time()[3:, 0]
        ),
    )
    print(
        "----------------------------------------------------------------------------------"
    )


def iterate_FMs(epsilon=0.5):

    hbms = compare_FMs()
    labels = ['constr', 'inv', 'ang']
    solver = NewtonSolver(verbose=False, max_iterations=20)

    for idx_hbm, hbm in enumerate(hbms):
        print(f"Iterating on case {labels[idx_hbm]} with delta as parameter")
        sys = EquationSystem(equations=[hbm], unknowns=["X"])
        sys.equations[0].delta = 0
        sys.equations[0].epsilon = 0.5
        sys.equations[0].g = -2
        results = []
        for k, branch_point in enumerate(pseudo_arclength_continuator(
            initial_direction=1,
            initial_system=sys,
            solver=solver,
            stepsize=0.05,
            stepsize_range=(0.05, 0.05),
            continuation_parameter="g",
            verbose=True,
            num_steps=10000,
        )):
            FMs = branch_point.equations[0].stability_method.determine_eigenvalues(branch_point.equations[0])
            print(f"g={branch_point.g}, FMs: {FMs}")
            results.append(np.array([np.squeeze(branch_point.g), *FMs]))

            if k % 100 == 0:

                # Convert results to numpy array and save as CSV
                results_array = np.array(results)
                filename = f"results_{labels[idx_hbm]}_epsi_{branch_point.equations[0].epsilon}.csv"
                np.savetxt(
                    filename,
                    results_array,
                    delimiter=",",
                    header="g,"
                    + ",".join([f"FM_{i}" for i in range(results_array.shape[1] - 1)]),
                    comments="",
                )
                print(f"Results saved to {filename}")
            if branch_point.equations[0].g > 10:
                break


if __name__ == "__main__":
    # only_constr()
    # anims = main()
    # periodic_initial_condition()
    iterate_FMs()
    plt.show()
