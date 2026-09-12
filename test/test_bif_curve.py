"""Test HBM using the Duffing oscillator. See also: Matlab workshop, Task x.x"""

import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import pytest

from skhippr.Fourier import Fourier
from skhippr.cycles.hbm import HBMSystem
from skhippr.cycles.shooting import ShootingSystem
from skhippr.odes.nonautonomous import Duffing
from examples.vanderpol_minimal import Vanderpol
from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic
from skhippr.solvers.continuation import pseudo_arclength_continuator
from skhippr.solvers.newton import NewtonSolver


@pytest.fixture(params=[True, False], name="ode")
def ode_fixture(request):
    return ode(autonomous=request.param)


def ode(autonomous):
    if autonomous:
        return Vanderpol(t=0, x=np.array([2.0, 0]), nu=0.05)
    else:
        return Duffing(
            t=0,
            x=np.array([1.0, 0.0]),
            omega=0.8,
            alpha=1,
            beta=0.2,
            delta=0.1,
            F=3,
        )


@pytest.fixture(name="shooting_system")
def shooting_system_fixture(ode):
    return shooting_system(ode)


def shooting_system(ode):
    if ode.autonomous:
        omega = 1
    else:
        omega = ode.omega
    return ShootingSystem(ode=copy(ode), T=2 * np.pi / omega, atol=1e-4, rtol=1e-5)


@pytest.fixture(name="hbm_system")
def hbm_system_fixture(ode):
    return hbm_system(ode)


def hbm_system(ode):
    if ode.autonomous:
        omega = 1
    else:
        omega = ode.omega

    fourier = Fourier(N_HBM=25, L_DFT=300, n_dof=ode.n_dof, real_formulation=True)

    ts = fourier.time_samples(omega)
    x0_samples = np.vstack((np.cos(omega * ts), -omega * np.sin(omega * ts)))
    X0 = fourier.DFT(x0_samples)

    stability_method = KoopmanHillSubharmonic(fourier=fourier)

    return HBMSystem(
        ode=copy(ode),
        initial_guess=X0,
        omega=omega,
        fourier=fourier,
        stability_method=stability_method,
    )


def run_continuation(system, solver, parameter, param_range, max_steps=200):
    branch = []

    for bp in pseudo_arclength_continuator(
        initial_system=system,
        solver=solver,
        stepsize=0.1,
        stepsize_range=(0.001, 0.4),
        initial_direction=1,
        num_steps=max_steps,
        continuation_parameter=parameter,
        verbose=True,
    ):
        branch.append(bp)

        if not param_range[0] <= getattr(bp, parameter) <= param_range[1]:
            break
    return branch


def plot_bif_curve_nonaut():

    solver = NewtonSolver()
    duffing = ode(False)
    hbm_sys = hbm_system(duffing)
    fourier = hbm_sys.equations[0].fourier
    shooting_sys = shooting_system(duffing)

    F_range = (0, 20)

    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    ax_3d.set_xlabel("x_1")
    ax_3d.set_ylabel("x_2")
    ax_3d.set_zlabel("F")

    fig_stability = plt.figure()

    for system in (hbm_sys, shooting_sys):

        branch = run_continuation(system, solver, "F", F_range, 200)
        Fs = np.array([np.squeeze(bp.F) for bp in branch])

        xs_time = [bp.equations[0].x_time() for bp in branch]

        # Tangents
        is_hbm = isinstance(system, type(hbm_sys))
        for branch_point, x_time in zip(branch[:-1], xs_time[:-1]):

            x0 = np.append(x_time[:, 0], branch_point.F)
            x_tng = branch_point.vector_of_unknowns + 0.2 * branch_point.tangent

            if is_hbm:
                x0_tng = np.append(fourier.inv_DFT(x_tng[:-1])[:, 0], x_tng[-1])
            else:
                x0_tng = x_tng

            ax_3d.plot(*zip(x0, x0_tng), "gray")

        x0s = np.array([x[:, 0] for x in xs_time])
        ax_3d.plot(x0s[:, 0], x0s[:, 1], Fs, linewidth=1)

        # Stability plot
        plt.figure(fig_stability)
        amplitudes = [np.max(x[0, :]) for x in xs_time]

        stable = np.array([bp.stable for bp in branch])
        F_stable = np.where(stable, Fs, np.nan)
        F_unstable = np.where(~stable, Fs, np.nan)

        plt.plot(F_stable, amplitudes, label="stable")
        plt.plot(F_unstable, amplitudes, "--", label="unstable")


def test_bifurcation_curve_comparison(hbm_system, shooting_system):

    if hbm_system.equations[0].ode.autonomous:
        parameter = "nu"
        param_range = (0, 3.5)
    else:
        parameter = "F"
        param_range = (0, 20)

    solver = NewtonSolver()

    branches = [
        run_continuation(system, solver, parameter, param_range, max_steps=200)
        for system in [hbm_system, shooting_system]
    ]

    # Compare the curves using interpolation
    errors = compare_branches(branch_hbm=branches[0], branch_shoot=branches[1])
    assert np.all(np.array(errors) < 1e-2)


def hbm_x0_vec(bp_hbm, x0_ref=None):
    hbm = bp_hbm.equations[0]
    x_t = hbm.x_time()
    if hbm.ode.autonomous:
        # Eliminate freedom of phase by approximating the closest point to x_ref
        x = closest_point_on_solution(x0_ref, x_t)
        x = np.append(x, 2 * np.pi / hbm.omega)
    else:
        x = x_t[:, 0]
    # non-autonomous: take state at t=0 and append continuation parameter
    return np.append(x, getattr(bp_hbm, bp_hbm.unknowns[-1]))


def compare_branches(branch_hbm, branch_shoot):
    """Compare each shooting branch point to the HBM branch using interpolation."""
    parameter = branch_shoot[0].unknowns[-1]
    param_range = (
        getattr(branch_hbm[0], parameter),
        getattr(branch_hbm[-1], parameter),
    )
    n_dof = branch_shoot[0].equations[0].ode.n_dof
    idx_hbm = 0
    errors = []

    for k, bp_shoot in enumerate(branch_shoot):
        value_param = getattr(bp_shoot, parameter)
        if not param_range[0] <= value_param <= param_range[1]:
            print(f"{k}-th shooting point outside {parameter} range")
            continue

        x_shoot = bp_shoot.vector_of_unknowns
        window = range(max(0, idx_hbm - 5), min(len(branch_hbm) - 1, idx_hbm + 6))
        get_vec = lambda i: hbm_x0_vec(branch_hbm[i], x_shoot[:n_dof])

        err, idx_hbm, _ = best_segment(x_shoot, get_vec, window)
        print(f"{k}: {parameter} = {value_param}, d = {err}, idx = {idx_hbm}")
        errors.append(err)
    return errors


def project_segment(x, a, b):
    """Project vector x onto segment a -> b. Returns (alpha in [0,1], x_proj, dist)."""
    d = b - a
    norm = np.linalg.norm(d) ** 2
    if norm == 0:
        raise ValueError("'a' and 'b' are identical!")

    alpha = np.dot(x - a, d) / norm
    x_proj = a + alpha * d
    return x_proj, np.linalg.norm(x - x_proj)


def best_segment(x, get_vec, window):
    """
    Search within the window for the segment i→i+1 with the best projection of x.
    get_vec(i) must return the vector at index i.
    length is the number of points; segments are (i, i+1).
    """
    best = (np.inf, np.nan, None)  # (dist, idx, alpha, x_proj)

    for i in window:
        a, b = get_vec(i), get_vec(i + 1)
        x_proj, dist = project_segment(x, a, b)
        if dist < best[0]:
            best = (dist, i, x_proj)

    return best  # dist, idx, x_proj


def closest_point_on_solution(x, x_time):
    """
    Interpolate along time samples x_time[:, i] to match x.
    It is assumed that x_time encodes a periodic solution, i.e., there is a branch segment connecting x_time[:, -1] and x_time[:, 0].
    """
    L = x_time.shape[1]
    window = range(L)

    def get_vec(i):
        return x_time[:, np.mod(i, L)]

    _, _, x_proj = best_segment(x, get_vec, window)
    return x_proj


if __name__ == "__main__":
    plot_bif_curve_nonaut()
    # plt.figure()
    # animation = bif_curve_aut(real_formulation=True, sparse=False)
    plt.show()
    print("All tests successful.")
