import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import tikzplotlib

from scipy.integrate import solve_ivp
from scipy.linalg import (
    lu_factor,
    lu_solve,
)


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
    drazin,
)


def integrate_numerically_for_peri_sol(
    pend_inv=None, num_periods_to_converge=0, animate=True, initial_angles=None
):
    # initial_angles_deg = np.array([0, 0, 0])
    N_HBM = 10
    L_DFT = 1024

    if pend_inv is None:
        pend_inv = init_pendulum(pend_case="inv", L_sol=L_DFT)[0]
    if initial_angles is not None:
        pend_inv.x[:3] = initial_angles

    fourier = Fourier(n_dof=pend_inv.n_dof, N_HBM=N_HBM, L_DFT=L_DFT)

    if num_periods_to_converge > 0:
        t_integrate = (0, num_periods_to_converge * 2 * np.pi / pend_inv.omega)
        ode_res = solve_ivp(
            fun=pend_inv.dynamics,
            t_span=t_integrate,
            y0=pend_inv.x,
            # t_eval=np.linspace(*t_integrate, num=fourier.L_DFT, endpoint=False),
        )
        initial_condition = ode_res.y[:, -1]
        print("integrated to converge.")

        if animate:
            _, ax = plt.subplots(3, 1)
            for i, label in enumerate(["alpha", "beta", "gamma"]):
                ax[i].plot(ode_res.t, ode_res.y[i, :], label=label)
                ax[i].set_xlabel("Time")
                ax[i].set_ylabel("Angle (rad)")
                ax[i].legend()

    else:
        initial_condition = pend_inv.x

    t_peri = (0, 2 * np.pi / pend_inv.omega)
    ode_res = solve_ivp(
        fun=pend_inv.dynamics,
        t_span=t_peri,
        y0=initial_condition,
        t_eval=np.linspace(*t_peri, num=fourier.L_DFT + 1, endpoint=True),
    )
    print(
        f"Integrated for peri. sol. -- error {np.linalg.norm(ode_res.y[:, -1] - ode_res.y[:, 0])}"
    )
    print(f"End angles: {ode_res.y[:3, -1]}")
    print(f"End velocity: {ode_res.y[3:, -1]}")

    x_peri = ode_res.y[:, :-1]

    anims = []

    if animate:
        anims.append(animate_pendulum(pend_inv, ode_res.t, ode_res.y))

    return anims, x_peri


def angle_solution_to_constrained_solution(x_angle, pend_ang, pend_constr):

    L_sol = x_angle.shape[1]

    ts = np.linspace(0, 2 * np.pi / pend_ang.omega, L_sol, endpoint=False)
    delta_t = ts[1] - ts[0]
    I_r_OS = np.zeros_like(x_angle[:3, :])
    I_v_S = np.zeros_like(x_angle[3:, :])
    lambdas = np.zeros_like(x_angle[:3, :])

    dynamics_error = np.zeros((15, L_sol))

    for i in range(L_sol):
        I_r_OS[:, i] = pend_ang.I_r_OS(t=ts[i], angles=x_angle[:3, i])
        I_v_S[:, i] = pend_ang.I_v_S(
            t=ts[i], angles=x_angle[:3, i], d_angles=x_angle[3:, i]
        )

        # populate x with lambda=zero for later determination of lambda
        x_i = np.hstack(
            (x_angle[:3, i], I_r_OS[:, i], x_angle[3:, i], I_v_S[:, i], lambdas[:, i])
        )

        # second derivatives by finite differences
        # under the assumption that the solution is periodic, this works also for i == 0
        a_S = (I_v_S[:, i] - I_v_S[:, i - 1]) / delta_t
        dd_angles = (x_angle[3:, i] - x_angle[3:, i - 1]) / delta_t

        M = pend_constr.total_mass * np.eye(3)
        h = pend_constr.dynamics(t=ts[i], x=x_i)[9:12]

        # M*a_S = h + W @ lambdas[:, i]
        lambdas[:, i] = M @ a_S - h

        # update x_i
        x_i[12:] = lambdas[:, i]

        d_x = np.hstack((x_angle[3:, i], I_v_S[:, i], dd_angles, a_S, np.zeros(3)))

        dynamics_error[:, i] = pend_constr.M_small(
            t=ts[i], x=x_i
        ) @ d_x - pend_constr.dynamics(t=ts[i], x=x_i)

    # 3 angles, 3 COM positions, 3 angular velocities, 3 COM velocities, 3 lambdas
    x = np.vstack((x_angle[:3, :], I_r_OS, x_angle[3:, :], I_v_S, lambdas))
    return x, dynamics_error


def test_angle_to_constr():
    pend, sol_ang = init_pendulum(pend_case="ang", L_sol=1024)
    pend_constr, sol_constr = init_pendulum(pend_case="constr", L_sol=1024)

    sol_constr_computed, dynamics_error = angle_solution_to_constrained_solution(
        sol_ang, pend, pend_constr
    )

    _, axs = plt.subplots(15, 3)
    for i in range(sol_constr_computed.shape[0]):
        axs[i][0].plot(sol_constr[i, :], label="ref")
        axs[i][0].plot(sol_constr_computed[i, :], "--")
        axs[i][1].semilogy(np.abs(sol_constr_computed[i, :] - sol_constr[i, :]))
        axs[i][2].plot(dynamics_error[i, :])

    axs[0][0].set_title("Constrained solution")
    axs[0][1].set_title("Error to reference")
    axs[0][2].set_title("Dynamics error")

    # # hbm_constr = HBMEquationDAE(
    # #     dae=pend_constr,
    # #     omega=pend_constr.omega,
    # #     fourier=fourier_constr,
    # #     initial_guess=X_init,
    # #     stability_method=KoopmanHillDAESubharmonic(fourier),
    # # )
    # # print(hbm_constr.residual_function())
    # # solver.solve_equation(equation=hbm_constr, unknown="X")
    # # print("solved constr equation.")

    # # animate_pendulum(
    # #     pend_constr,
    # #     hbm_constr.fourier.time_samples(omega=pend_constr.omega),
    # #     hbm_constr.x_time()[:3, :],
    # # )

    # # _, ax_FM = plt.subplots(1, 1)
    # # ax_FM.plot(
    # #     np.real(hbm_inv.eigenvalues), np.imag(hbm_inv.eigenvalues), "x", label="inv"
    # # )
    # # # ax_FM.plot(np.real(hbm.eigenvalues), np.imag(hbm.eigenvalues), "+", label="ang")
    # # ax_FM.plot(
    # #     np.real(hbm_constr.eigenvalues),
    # #     np.imag(hbm_constr.eigenvalues),
    # #     "s",
    # #     label="constr",
    # # )
    # # ax_FM.legend()


def only_constr():

    fourier = Fourier(n_dof=1, N_HBM=10, L_DFT=1024)

    pend, sol = init_pendulum(pend_case="constr", L_sol=fourier.L_DFT)
    l = 0.5 * np.sqrt(pend.a**2 + pend.b**2 + pend.c**2)

    # Verify that pendulum is vertical
    I_r_SP = pend.A_IK() @ pend.K_r_SP
    print("I_r_SP in inertial frame: ", I_r_SP)
    print("error to expected position:", np.linalg.norm(I_r_SP - np.array([0, 0, l])))

    hbm = init_hbm(
        pend, x_sol=sol, fourier_ref=fourier, stability_method_class=KoopmanHillDAE
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

    fourier_ref = Fourier(n_dof=6, N_HBM=5, L_DFT=1024)
    solver = NewtonSolver(verbose=True, max_iterations=20)

    _, ax = plt.subplots(1, 1)
    hbms = []
    for label, marker in zip(["constr", "inv", "ang"], ["x", "+", "o"]):
        pend, x_sol = init_pendulum(pend_case=label, L_sol=fourier_ref.L_DFT)

        if type(pend) == SpatialPendulumWithMassInverted:
            stability_method_class = KoopmanHillProjection
        else:
            stability_method_class = KoopmanHillDAE

        hbm = init_hbm(pend, x_sol, fourier_ref, stability_method_class)
        print(f"Solving {label} equation")
        solver.solve_equation(equation=hbm, unknown="X")
        print(f"Floquet mutlipliers:")
        for FM in np.sort(np.abs(hbm.eigenvalues)):
            print(FM)
        ax.plot(np.real(hbm.eigenvalues), np.imag(hbm.eigenvalues), marker, label=label)
        hbms.append(hbm)

    tikzplotlib.save("plots/spatial_pendulum_FMs.tex")

    return hbms


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


def iterate_FMs(hbm="constr", epsilon=0.5, description=None):

    if type(hbm) == str:
        pend = init_pendulum(pend_case=hbm, L_sol=1024)[0]
        hbm = init_hbm(
            pend, x_sol=None, fourier_ref=Fourier(n_dof=6, N_HBM=5, L_DFT=1024)
        )

    solver = NewtonSolver(verbose=False, max_iterations=20)

    if description is None:
        description = f"{hbm.ode.__class__.__name__}"

        print(f"Iterating on {description} with delta as parameter")
        sys = EquationSystem(equations=[hbm], unknowns=["X"])
        sys.equations[0].delta = 0
        sys.equations[0].epsilon = 0.5
        sys_equations[0].g = -2
        results = []
        for k, branch_point in enumerate(
            pseudo_arclength_continuator(
                initial_direction=1,
                initial_system=sys,
                solver=solver,
                stepsize=0.05,
                stepsize_range=(0.05, 0.05),
                continuation_parameter="g",
                verbose=True,
                num_steps=10000,
            )
        ):
            FMs = branch_point.equations[0].stability_method.determine_eigenvalues(
                branch_point.equations[0]
            )
            print(f"g={branch_point.g}, FMs: {FMs}")
            results.append(np.array([np.squeeze(branch_point.g), *FMs]))

            if k % 100 == 0:

                # Convert results to numpy array and save as CSV
                results_array = np.array(results)
                filename = f"results_{description}_epsilon_{branch_point.equations[0].epsilon}.csv"
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


def precision():
    solver = NewtonSolver(verbose=False, max_iterations=20)
    for idx_hbm, hbm in enumerate(compare_FMs()):
        fourier_ref = hbm.fourier
        x_ref = fourier_ref.inv_DFT(hbm.X)
        idx_sort = np.argsort(np.angle(hbm.eigenvalues))
        FMs_ref = hbm.eigenvalues[idx_sort]
        Ns = np.arange(fourier_ref.N_HBM, 0, -1)
        errors = np.zeros((x_ref.shape[0], len(Ns)), dtype=complex)
        FMs = np.zeros_like(errors)
        for k, N_HBM in enumerate(Ns):
            fourier = fourier_ref.__replace__(N_HBM=N_HBM)
            hbm.fourier = fourier
            hbm.X = fourier.DFT(x_ref)
            hbm.stability_method = type(hbm.stability_method)(fourier)

            solver.solve_equation(hbm, "X")
            print(
                f"{hbm.__class__.__name__} with N_HBM={N_HBM}, FMs: {hbm.eigenvalues}"
            )
            idx_sort = np.argsort(np.angle(hbm.eigenvalues))
            FMs[:, k] = hbm.eigenvalues[idx_sort]
            errors[:, k] = FMs[:, k] - FMs_ref
        np.savetxt(
            f"FMs_errors_case_{idx_hbm}.csv",
            FMs,
            delimiter=";",
            header=";".join([f"N_HBM = {n}" for n in Ns]),
            comments="",
        )
        fig, ax = plt.subplots(1, 1)
        ax.semilogy(Ns, np.max(np.abs(errors), axis=0), "x-", label=f"case{idx_hbm}")


def init_pendulum(
    pend_case="constr",
    a=1,
    b=2,
    c=3,
    delta=2,
    epsilon=1,
    omega=1,
    L_sol=512,
    damping=0,
    spring=0.0001,
    density=0.5,
):

    l = 0.5 * np.sqrt(a**2 + b**2 + c**2)
    angles = np.array([0, -0.27055, 0.588])
    angles = angles[:, np.newaxis] @ np.ones((1, L_sol))
    dangles = np.zeros_like(angles)
    x_sol = np.vstack((angles, dangles))

    match pend_case:
        case "constr":
            pend = SpatialPendulumWithConstraints(
                t=0,
                angles=angles[:, 0],
                d_angles=dangles[:, 0],
                I_r_OS=[0, 0, 0],
                I_v_S=[0, 0, 0],
                lam=[0, 0, 0],
                shape_cuboid=[a, b, c],
                density=density,
                delta=delta,
                epsilon=epsilon,
                omega=omega,
                damping=damping,
                spring=spring,
            )

            t = np.linspace(0, 2 * np.pi / omega, L_sol, endpoint=False)
            e = delta + epsilon * np.cos(omega * t)
            de = -epsilon * omega * np.sin(omega * t)
            dde = -epsilon * (omega**2) * np.cos(omega * t)

            r_OS = np.zeros_like(angles)
            r_OS[2, :] = -l + e

            v_S = np.zeros_like(angles)
            v_S[2, :] = de

            lambdas = np.zeros_like(angles)
            lambdas[2, :] = pend.total_mass * (9.81 + dde)

            x_sol = np.vstack((angles, r_OS, dangles, v_S, lambdas))

        case "inv":
            pend = SpatialPendulumWithMassInverted(
                t=0,
                angles=angles[:, 0],
                d_angles=dangles[:, 0],
                shape_cuboid=[a, b, c],
                density=density,
                delta=delta,
                epsilon=epsilon,
                omega=omega,
                damping=damping,
                spring=spring,
                stability_method=None,
            )
        case "ang":
            pend = SpatialPendulumWithoutConstraints(
                t=0,
                angles=angles[:, 0],
                d_angles=dangles[:, 0],
                shape_cuboid=[a, b, c],
                density=density,
                delta=delta,
                epsilon=epsilon,
                omega=omega,
                damping=damping,
                spring=spring,
                stability_method=None,
            )

        case _:
            raise ValueError(
                f"Unknown pendulum case: {pend_case}. Allowed: 'constr', 'inv', 'ang'"
            )

    return pend, x_sol


def init_hbm(pend, x_sol, fourier_ref: Fourier, stability_method_class=None):

    fourier = fourier_ref.__replace__(n_dof=pend.n_dof)
    if stability_method_class is None:
        stability_method_class = lambda fourier: None

    if x_sol is None:
        x_sol = np.zeros((fourier.n_dof, fourier.L_DFT))

    if type(pend) == SpatialPendulumWithMassInverted:
        hbm = HBMEquation(
            ode=pend,
            omega=pend.omega,
            fourier=fourier,
            initial_guess=fourier.DFT(x_sol),
            stability_method=stability_method_class(fourier),
        )
    else:
        hbm = HBMEquationDAE(
            dae=pend,
            omega=pend.omega,
            fourier=fourier,
            initial_guess=fourier.DFT(x_sol),
            stability_method=stability_method_class(fourier),
        )
    return hbm


def visualize_drazin(
    pend,
    x_sol,
    Ns,
    fourier_ref: Fourier,
    solver: NewtonSolver,
    tol_cond=1e6,
    tol_drazin=1e-7,
):

    # tol_cond = 1e6
    # tol_drazin = 1e-7

    for N in Ns:
        fourier_constr = fourier_ref.__replace__(N_HBM=N)

    hbm_constr = HBMEquationDAE(
        dae=pend,
        omega=pend.omega,
        fourier=fourier_constr,
        initial_guess=fourier_constr.DFT(x_sol),
        stability_method=None,
    )

    solver.solve_equation(equation=hbm_constr, unknown="X")
    hill_matrix = hbm_constr.hill_matrix(update=True)
    mass_matrix = hbm_constr.M()

    # Copy&pasted from KoopmanHillDAE.generalized_exponential()
    a_vals = [1.0, 10.0, 0.1, 100, 0.01, 1000, 0.001]
    success = False
    for a in a_vals:
        pencil = a * mass_matrix - hill_matrix
        if np.linalg.cond(pencil) < tol_cond:
            success = True
            print(f"N = {N}: a = {a}")
            break
    if not success:
        raise RuntimeError(
            f"Could not find suitable scaling factor 'a' for Drazin inverse with condition < {tol_cond}."
        )

    pencil_lu = lu_factor(a * mass_matrix - hill_matrix)
    pencil_M = lu_solve(pencil_lu, mass_matrix)
    pencil_drazin, ratio = drazin(pencil_M, tol_drazin, ax=True)
    print(f"Ratio of Drazin inverse: {ratio}")


if __name__ == "__main__":
    # only_constr()
    # anims = integrate_numerically_for_peri_sol()
    # anims2 = integrate_numerically_for_peri_sol(
    #     initial_angles=[0, 0, 0], num_periods_to_converge=1
    # )

    # test_angle_to_constr()
    # compare_FMs()
    iterate_FMs("ang")
    iterate_FMs("constr")
    # periodic_initial_condition()
    # iterate_FMs()
    # precision()
    # visualize_drazin()
    plt.show()
