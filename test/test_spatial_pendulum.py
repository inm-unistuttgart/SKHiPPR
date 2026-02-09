import numpy as np
import pytest
from skhippr.odes.spatialpendulum import *


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
def pend():
    vec_angles = np.random.rand(3)
    return SpatialPendulumWithoutConstraints(
        0, vec_angles, np.zeros(3), [1, 1, 1], 0.1, 1, 1, 1, 1, stability_method=None
    )


def test_angle_extraction(pend):
    assert np.linalg.norm(pend.d_angles) == 0
    assert np.linalg.norm(pend.angles - pend.x[:3]) == 0
    try:
        pend.angles = np.array([0.1, 0.2, 0.3])
        raise AssertionError("Should not be able to set angles directly.")
    except AttributeError as AE:
        pass  # expected behavior

    try:
        pend.d_angles = np.array([0.1, 0.2, 0.3])
        raise AssertionError("Should not be able to set angle derivatives directly.")
    except AttributeError as AE:
        pass  # expected behavior


def test_A_KI(pend):
    range_angles = np.arange(0, 2 * np.pi, 13)
    for alpha in range_angles:
        for beta in range_angles:
            for gamma in range_angles:
                angles = np.array([alpha, beta, gamma])
                res_A_IK = pend.A_IK(angles)
                res_prod = pend.A_I1(angles) @ pend.A_12(angles) @ pend.A_2K(angles)
                res_exp = (
                    trafo_around_z(alpha) @ trafo_around_x(beta) @ trafo_around_z(gamma)
                )
                assert np.allclose(res_A_IK, res_exp)
                assert np.allclose(res_prod, res_exp)


def test_fd():
    func = lambda x: np.array([np.sin(x), np.cos(x), np.exp(x)])
    res_expected = np.array([np.cos(1), -np.sin(1), np.exp(1)])
    finite_difference(func, res_expected, arg=1)


def finite_difference(func, res_expected, arg=0, stepsize=1e-5):
    val_0 = func(arg)
    val_1 = func(arg + stepsize)
    res = (val_1 - val_0) / stepsize
    assert np.allclose(
        res, res_expected, atol=1e-2
    ), f"Finite difference error: {np.linalg.norm(res - res_expected)}"
