import numpy as np
import pytest
from skhippr.odes.spatialpendulum import *


def finite_difference(func, res_expected, arg=0, stepsize=1e-5):
    val_0 = func(arg)
    val_1 = func(arg + stepsize)
    res = (val_1 - val_0) / stepsize
    assert np.allclose(
        res, res_expected, atol=1e-2
    ), f"Finite difference error: {np.linalg.norm(res - res_expected)}"


def test_fd():
    func = lambda x: np.array([np.sin(x), np.cos(x), np.exp(x)])
    res_expected = np.array([np.cos(1), -np.sin(1), np.exp(1)])
    finite_difference(func, res_expected, arg=1)


def test_trafo_x():
    alphas = np.linspace(0, 2 * np.pi, num=300)
    d_alpha = alphas[1] - alphas[0]
    vec_0 = np.array([1, 1, 0])
    vec_before = vec_0
    for alpha in alphas:
        vec_trafo = trafo_around_x(alpha) @ vec_0
        vec_expected = np.array([1, np.cos(alpha), np.sin(alpha)])
        assert np.isclose(np.linalg.norm(vec_trafo), np.linalg.norm(vec_expected))

        if alpha > 0:
            deriv = d_trafo_around_x(alpha) @ vec_0
            deriv_expected = (vec_trafo - vec_before) / d_alpha
            error = np.linalg.norm(deriv - deriv_expected)
            try:
                assert np.allclose(deriv, deriv_expected, atol=1e-1)
            except AssertionError as AE:
                print(f"L2 norm error at alpha={alpha:.4f}: {error}")
                raise AE

        vec_before = vec_trafo


def test_trafo_z():
    alphas = np.linspace(0, 2 * np.pi, num=300)
    d_alpha = alphas[1] - alphas[0]
    vec_0 = np.array([1, 0, 1])
    vec_before = vec_0
    for alpha in alphas:
        vec_trafo = trafo_around_z(alpha) @ vec_0
        vec_expected = np.array([np.cos(alpha), np.sin(alpha), 1])
        assert np.isclose(np.linalg.norm(vec_trafo), np.linalg.norm(vec_expected))

        if alpha > 0:
            deriv = d_trafo_around_z(alpha) @ vec_0
            deriv_expected = (vec_trafo - vec_before) / d_alpha
            error = np.linalg.norm(deriv - deriv_expected)
            try:
                assert np.allclose(deriv, deriv_expected, atol=1e-1)
            except AssertionError as AE:
                print(f"L2 norm error at alpha={alpha:.4f}: {error}")
                raise AE

        vec_before = vec_trafo


def test_tilde_operator():
    for k in range(5):
        vec1 = np.random.rand(3)
        vec2 = np.random.rand(3)
        res1 = tilde_operator(vec1) @ vec2
        res3 = tilde_operator(vec2) @ vec1
        res_exp = np.cross(vec1, vec2)
        assert np.allclose(res1, res_exp)
        assert np.allclose(res3, -res_exp)


@pytest.fixture
def pend_without_constraints():
    vec_angles = np.random.rand(3)
    vec_d_angles = np.random.rand(3)
    return SpatialPendulumWithoutConstraints(
        0, vec_angles, vec_d_angles, [1, 2, 3], 0.1, 1, 1, 1, 1, stability_method=None
    )


def test_angle_extraction(pend_without_constraints):
    assert (
        np.linalg.norm(pend_without_constraints.angles - pend_without_constraints.x[:3])
        == 0
    )
    assert (
        np.linalg.norm(
            pend_without_constraints.d_angles - pend_without_constraints.x[3:]
        )
        == 0
    )
    try:
        pend_without_constraints.angles = np.array([0.1, 0.2, 0.3])
        raise AssertionError("Should not be able to set angles directly.")
    except AttributeError as AE:
        pass  # expected behavior

    try:
        pend_without_constraints.d_angles = np.array([0.1, 0.2, 0.3])
        raise AssertionError("Should not be able to set angle derivatives directly.")
    except AttributeError as AE:
        pass  # expected behavior


def test_A_KI(pend_without_constraints):
    range_angles = np.arange(0, 2 * np.pi, 13)
    for alpha in range_angles:
        for beta in range_angles:
            for gamma in range_angles:
                angles = np.array([alpha, beta, gamma])
                res_A_IK = pend_without_constraints.A_IK(angles)
                res_prod = (
                    pend_without_constraints.A_I1(angles)
                    @ pend_without_constraints.A_12(angles)
                    @ pend_without_constraints.A_2K(angles)
                )
                res_exp = (
                    trafo_around_z(alpha) @ trafo_around_x(beta) @ trafo_around_z(gamma)
                )
                assert np.allclose(res_A_IK, res_exp)
                assert np.allclose(res_prod, res_exp)


def test_K_Omega(pend_without_constraints):

    for angles in [np.random.rand(3)]:
        for d_angles in [np.random.rand(3)]:
            if d_angles is None:
                d_angles = pend_without_constraints.d_angles
            d_alpha, d_beta, d_gamma = d_angles

            omega_1 = np.array([0, 0, d_alpha])
            omega_2 = pend_without_constraints.A_12(angles).T @ omega_1 + np.array(
                [d_beta, 0, 0]
            )
            omega_ref = pend_without_constraints.A_2K(angles).T @ omega_2 + np.array(
                [0, 0, d_gamma]
            )
            assert np.allclose(
                pend_without_constraints.K_Omega(angles, d_angles), omega_ref
            )


def test_d_K_Omega_d_angle(pend_without_constraints):
    eye = np.eye(3)
    for angles, d_angles in zip([None, np.random.rand(3)], [None, np.random.rand(3)]):
        for k, variable in enumerate(["alpha", "beta", "gamma"]):
            if angles is None:
                fd_func = lambda var: pend_without_constraints.K_Omega(
                    angles=pend_without_constraints.angles + var * eye[:, k],
                    d_angles=d_angles,
                )
            else:
                fd_func = lambda var: pend_without_constraints.K_Omega(
                    angles=angles + var * eye[:, k],
                    d_angles=d_angles,
                )

            d_K_Omega = pend_without_constraints.d_K_Omega(
                angles=angles, d_angles=d_angles, variable=variable
            )
            d_K_J = pend_without_constraints.d_K_J_R(angles=angles, variable=variable)
            if d_angles is None:
                d_K_Omega_exp = d_K_J @ pend_without_constraints.d_angles
            else:
                d_K_Omega_exp = d_K_J @ d_angles
            finite_difference(fd_func, d_K_Omega_exp)
            assert np.allclose(d_K_Omega, d_K_Omega_exp)
            finite_difference(fd_func, d_K_Omega)


def test_d_K_Omega_d_dangle(pend_without_constraints):
    eye = np.eye(3)
    for angles, d_angles in zip([None, np.random.rand(3)], [None, np.random.rand(3)]):
        for k, variable in enumerate(["d_alpha", "d_beta", "d_gamma"]):
            if d_angles is None:
                fd_func = lambda var: pend_without_constraints.K_Omega(
                    angles=angles,
                    d_angles=pend_without_constraints.d_angles + var * eye[:, k],
                )
            else:
                fd_func = lambda var: pend_without_constraints.K_Omega(
                    angles=angles,
                    d_angles=d_angles + var * eye[:, k],
                )

            d_K_Omega = pend_without_constraints.d_K_Omega(
                angles=angles, d_angles=d_angles, variable=variable
            )
            if angles is None:
                d_K_Omega_exp = (
                    pend_without_constraints.K_J_R(
                        angles=pend_without_constraints.angles
                    )
                    @ eye[:, k]
                )
            else:
                d_K_Omega_exp = (
                    pend_without_constraints.K_J_R(angles=angles) @ eye[:, k]
                )
            finite_difference(fd_func, d_K_Omega_exp)
            assert np.allclose(d_K_Omega, d_K_Omega_exp)
            finite_difference(fd_func, d_K_Omega)


def test_K_J_R(pend_without_constraints):
    for angles in [np.random.rand(3), None]:
        eye = np.eye(3)
        K_J_R = pend_without_constraints.K_J_R(angles)
        for col in range(3):
            col_exp = pend_without_constraints.K_Omega(angles, eye[:, col])
            assert np.allclose(K_J_R[:, col], col_exp)


def test_d_K_J_R(pend_without_constraints):
    eye = np.eye(3)
    for angles in [None, np.random.rand(3)]:
        for k, variable in enumerate(["alpha", "beta", "gamma"]):
            if angles is None:
                fd_func = lambda var: pend_without_constraints.K_J_R(
                    angles=pend_without_constraints.angles + var * eye[:, k]
                )
            else:
                fd_func = lambda var: pend_without_constraints.K_J_R(
                    angles=angles + var * eye[:, k]
                )

            d_K_J_R = pend_without_constraints.d_K_J_R(angles=angles, variable=variable)
            finite_difference(fd_func, d_K_J_R)


def test_d_rotational_energy(pend_without_constraints):
    eye = np.eye(3)
    for angles, d_angles in zip([None, np.random.rand(3)], [None, np.random.rand(3)]):
        for k, variable in enumerate(["alpha", "beta", "gamma"]):
            if angles is None:
                fd_func = [
                    lambda var: pend_without_constraints.rotational_energy(
                        angles=pend_without_constraints.angles + var * eye[:, k],
                        d_angles=None,
                    ),
                    lambda var: pend_without_constraints.rotational_energy(
                        angles=None,
                        d_angles=pend_without_constraints.d_angles + var * eye[:, k],
                    ),
                ]
            else:
                fd_func = [
                    lambda var: pend_without_constraints.rotational_energy(
                        angles=angles + var * eye[:, k],
                        d_angles=d_angles,
                    ),
                    lambda var: pend_without_constraints.rotational_energy(
                        angles=angles, d_angles=d_angles + var * eye[:, k]
                    ),
                ]

            for j, prefix in enumerate(["", "d_"]):
                variable_full = prefix + variable

                d_rot_energy = pend_without_constraints.d_rotational_energy(
                    angles=angles, d_angles=d_angles, variable=variable_full
                )
                finite_difference(fd_func[j], d_rot_energy)


def test_d_rotational_energy_dqdot(pend_without_constraints):
    for angles, d_angles in zip([None, np.random.rand(3)], [None, np.random.rand(3)]):
        d_energy_dqdot = pend_without_constraints.d_rotational_energy_dqdot(
            angles=angles, d_angles=d_angles
        )
        assert d_energy_dqdot.shape == (3,)
        for k, variable in enumerate(["d_alpha", "d_beta", "d_gamma"]):
            d_energy_d_var = pend_without_constraints.d_rotational_energy(
                angles, d_angles, variable=variable
            )
            assert np.allclose(d_energy_d_var, d_energy_dqdot[k])


def test_mass_rotational(pend_without_constraints):
    for angles, d_angles in zip([None, np.random.rand(3)], [None, np.random.rand(3)]):
        mass = pend_without_constraints.mass_rotational_energy(angles)
        for k, variable in enumerate(["d_alpha", "d_beta", "d_gamma"]):
            dd_energy_exp = mass[:, k]
            if d_angles is None:
                fd_func = (
                    lambda var: pend_without_constraints.d_rotational_energy_dqdot(
                        angles=angles,
                        d_angles=pend_without_constraints.d_angles
                        + var * np.eye(3)[:, k],
                    )
                )
            else:
                fd_func = (
                    lambda var: pend_without_constraints.d_rotational_energy_dqdot(
                        angles=angles, d_angles=d_angles + var * np.eye(3)[:, k]
                    )
                )
            finite_difference(fd_func, dd_energy_exp)


def test_dd_rotational_energy_dqdot_dq(pend_without_constraints):
    for angles, d_angles in zip([None, np.random.rand(3)], [None, np.random.rand(3)]):
        dd_energy = pend_without_constraints.dd_rotational_energy_dqdot_dq(
            angles=angles, d_angles=d_angles
        )

        for k, variable in enumerate(["alpha", "beta", "gamma"]):
            if angles is None:
                fd_func = (
                    lambda var: pend_without_constraints.d_rotational_energy_dqdot(
                        angles=pend_without_constraints.angles + var * np.eye(3)[:, k],
                        d_angles=d_angles,
                    )
                )
            else:
                fd_func = (
                    lambda var: pend_without_constraints.d_rotational_energy_dqdot(
                        angles=angles + var * np.eye(3)[:, k],
                        d_angles=d_angles,
                    )
                )
            finite_difference(fd_func, dd_energy[:, k], arg=0, stepsize=1e-5)


def test_gen_damping(pend_without_constraints):
    prevs = []
    for angles in [None, np.random.rand(3)]:
        for d_angles in [None, np.random.rand(3)]:
            gen_damping = pend_without_constraints.generalized_damping_force(
                angles=angles, d_angles=d_angles
            )
            for prev in prevs:
                assert not np.allclose(
                    gen_damping, prev
                ), "Generalized damping force should differ for different angles and velocities."
            prevs.append(gen_damping)


if __name__ == "__main__":

    vec_angles = np.random.rand(3)
    vec_d_angles = np.random.rand(3)
    pend = SpatialPendulumWithoutConstraints(
        0, vec_angles, vec_d_angles, [1, 1, 1], 0.1, 1, 1, 1, 1, stability_method=None
    )
    test_dd_rotational_energy_dqdot_dq(pend_without_constraints=pend)
