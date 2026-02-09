"""Dynamics of a vertically excited spatial pendulum, with and without constraints."""

from abc import abstractmethod
from typing import override
from skhippr.odes.AbstractODE import AbstractDAE
import numpy as np


class AbstractSpatialPendulum(AbstractDAE):
    """Abstract class for spatial pendulum, encoding everything that is the same with and without constraints."""

    def __init__(
        self,
        t,
        shape_cuboid,
        density,
        delta,
        epsilon,
        omega,
        damping,
        n_dof,
        invertible,
        stability_method=None,
    ):
        super().__init__(
            n_dof=n_dof,
            autonomous=False,
            stability_method=stability_method,
            M_is_constant=False,
            invertible=invertible,
        )
        self.a = shape_cuboid[0]
        self.b = shape_cuboid[1]
        self.c = shape_cuboid[2]
        self.density = density
        self.total_mass = self.a * self.b * self.c * self.density
        self.K_inertia = (
            1
            / (12 * self.total_mass)
            * np.diag(
                [self.b**2 + self.c**2, self.a**2 + self.c**2, self.a**2 + self.b**2]
            )
        )
        self.t = t
        self.damping = damping
        self.delta = delta
        self.epsilon = epsilon
        self.omega = omega
        self.K_r_SP = 0.5 * np.array(shape_cuboid)
        self.I_normal = self.K_r_SP / np.linalg.norm(self.K_r_SP)

        self.I_gravity = -9.81 * self.I_normal

    """ Extract angles and their derivatives from x"""

    @property
    def angles(self):
        return self.x[:3]

    @property
    def d_angles(self):
        return self.x[3:]

    """ Euler angle transformation matrices and their derivatives. """

    def A_I1(self, angles=None):
        if angles is None:
            angles = self.angles
        alpha = angles[0]
        return trafo_around_z(alpha)

    def d_A_I1(self, angles=None, d_angles=None, variable="alpha"):
        match variable:
            case "alpha":
                if angles is None:
                    angles = self.angles
                alpha = angles[0]
                return d_trafo_around_z(alpha)
            case "t":
                if d_angles is None:
                    d_angles = self.d_angles
                d_alpha = d_angles[0]
                return d_alpha * self.d_A_I1(angles=angles, variable="alpha")
            case _:
                return np.zeros((3, 3))

    def A_12(self, angles=None):
        if angles is None:
            angles = self.angles
        beta = angles[1]
        return trafo_around_x(beta)

    def d_A_12(self, angles=None, d_angles=None, variable="beta"):
        match variable:
            case "beta":
                if angles is None:
                    angles = self.angles
                beta = angles[1]
                return d_trafo_around_x(beta)
            case "t":
                if d_angles is None:
                    d_angles = self.d_angles
                d_beta = d_angles[1]
                return d_beta * self.d_A_12(angles=angles, variable="beta")
            case _:
                return np.zeros((3, 3))

    def A_2K(self, angles=None):
        if angles is None:
            angles = self.angles
        gamma = angles[2]
        return trafo_around_z(gamma)

    def d_A_2K(self, angles=None, d_angles=None, variable="gamma"):
        match variable:
            case "gamma":
                if angles is None:
                    angles = self.angles
                gamma = angles[2]
                return d_trafo_around_z(gamma)
            case "t":
                if d_angles is None:
                    d_angles = self.d_angles
                d_gamma = d_angles[2]
                return d_gamma * self.d_A_2K(angles=angles, variable="gamma")
            case _:
                return np.zeros((3, 3))

    def A_IK(self, angles=None):
        return (
            self.A_I1(angles=angles)
            @ self.A_12(angles=angles)
            @ self.A_2K(angles=angles)
        )

    def d_A_IK(self, angles=None, d_angles=None, variable="t"):
        return (
            self.d_A_I1(angles=angles, d_angles=d_angles, variable=variable)
            @ self.A_12(angles=angles)
            @ self.A_2K(angles=angles)
            + self.A_I1(angles=angles)
            @ self.d_A_12(angles=angles, d_angles=d_angles, variable=variable)
            @ self.A_2K(angles=angles)
            + self.A_I1(angles=angles)
            @ self.A_12(angles=angles)
            @ self.d_A_2K(angles=angles, d_angles=d_angles, variable=variable)
        )

    def A_KI(self, angles=None):
        return self.A_IK(angles=angles).T

    def d_A_KI(self, angles=None, d_angles=None, variable="t"):
        return self.d_A_IK(angles=angles, d_angles=d_angles, variable=variable).T

    """ Rotational quantities. """

    def K_J_R(self, angles=None):
        """Rotational Jacobian in body-fixed frame."""
        if angles is None:
            angles = self.angles
        _, beta, gamma = angles
        return np.array(
            [
                [-np.sin(gamma) * np.sin(beta), np.cos(gamma), 0],
                [np.sin(beta) * np.cos(gamma), np.sin(gamma), 0],
                [np.cos(beta), 0, 1],
            ]
        )

    def d_K_J_R(self, angles=None, d_angles=None, variable="t"):
        """Time derivative of the rotational Jacobian in body-fixed frame."""

        match variable:
            case "beta":
                if angles is None:
                    angles = self.angles
                _, beta, gamma = angles
                return np.array(
                    [
                        [-np.sin(gamma) * np.cos(beta), 0, 0],
                        [np.cos(beta) * np.cos(gamma), 0, 0],
                        [-np.sin(beta), 0, 0],
                    ]
                )
            case "gamma":
                if angles is None:
                    angles = self.angles
                _, beta, gamma = angles
                return np.array(
                    [
                        [-np.cos(gamma) * np.sin(beta), -np.sin(gamma), 0],
                        [-np.sin(beta) * np.sin(gamma), np.cos(gamma), 0],
                        [0, 0, 0],
                    ]
                )
            case "t":
                if d_angles is None:
                    d_angles = self.d_angles
                _, d_beta, d_gamma = d_angles
                return d_beta * self.d_K_J_R(
                    angles=angles, variable="beta"
                ) + d_gamma * self.d_K_J_R(angles=angles, variable="gamma")
            case _:
                return np.zeros((3, 3))

    def K_Omega(self, angles=None, d_angles=None):
        """Angular velocity of the rigid body in body-fixed frame."""
        if d_angles is None:
            d_angles = self.d_angles
        return self.K_J_R(angles=angles) @ d_angles

    def d_K_Omega(self, angles=None, d_angles=None, variable="t"):
        if d_angles is None:
            d_angles = self.d_angles
        match variable:
            case "d_alpha":
                return self.K_J_R(angles=angles)[0, :]
            case "d_beta":
                return self.K_J_R(angles=angles)[1, :]
            case "d_gamma":
                return self.K_J_R(angles=angles)[2, :]

            case ["alpha", "beta", "gamma"]:
                return self.d_K_J_R(angles=angles, variable=variable) @ d_angles

            case "t":
                raise ValueError(
                    "Time derivative of K_Omega requires 2nd derivatives of the angles and cannot be computed in this method. Use self.Mass_secondorder @ ddot_angles instead."
                )
            case _:
                return np.zeros(3)

    def generalized_damping_force(self, angles=None, d_angles=None):
        """Damping torque in body-fixed frame."""
        K_momentum = -self.damping * self.K_Omega(angles=angles, d_angles=d_angles)
        return (
            self.K_J_R.T @ K_momentum
        )  # generalized damping force in angle coordinates

    def rotational_energy(self, angles=None, d_angles=None):
        """Rotational kinetic energy."""
        K_Omega = self.K_Omega(angles=angles, d_angles=d_angles)
        return 0.5 * K_Omega.T @ self.K_inertia @ K_Omega

    def d_rotational_energy(self, angles=None, d_angles=None, variable="t"):
        """Derivative of the rotational kinetic energy."""
        K_Omega = self.K_Omega(angles=angles, d_angles=d_angles)
        d_K_Omega = self.d_K_Omega(angles=angles, d_angles=d_angles, variable=variable)
        return K_Omega.T @ self.K_inertia @ d_K_Omega

    def ddt_d_rotational_energy_dqdot_without_qdot(self, angles=None, d_angles=None):
        """Every component of d/dt(dT/dqdot) except M(q) @ dd_qdot."""
        result = np.zeros(3)
        for angle in ["alpha, beta, gamma"]:
            result += (
                self.d_K_J_R(angles=angles, d_angles=d_angles, variable=angle).T
                @ self.K_inertia
                @ self.K_Omega(angles=angles, d_angles=d_angles)
            )
            result += (
                self.K_J_R(angles=angles).T
                @ self.K_inertia
                @ self.d_K_Omega(angles=angles, d_angles=d_angles, variable=angle)
            )

    def mass_secondorder(self, angles=None):
        """Mass matrix for the second-order form of the equations of motion."""
        return self.K_J_R(angles=angles).T @ self.K_inertia @ self.K_J_R(angles=angles)

    def rhs_spin(self, angles=None, d_angles=None):
        """Contribution to the RHS of the angle-rows of the equations of motion due to balance of rotational momentum."""
        rhs = -self.ddt_d_rotational_energy_dqdot_without_qdot(angles, d_angles)
        for k, variable in enumerate(["alpha", "beta", "gamma"]):
            rhs[k] += self.d_rotational_energy(
                angles=angles, d_angles=d_angles, variable=variable
            )
        rhs += self.generalized_damping_force(angles=angles, d_angles=d_angles)
        return rhs

    def potential_energy(self, angles=None):
        """Potential energy."""
        if angles is None:
            angles = self.angles
        return -self.total_mass * np.inner(self.I_r_OS(angles=angles), self.I_gravity)

    @abstractmethod
    def I_r_OS(self, angles=None):
        """Position of the center of mass in inertial frame."""
        ...

    """ Kinematic quantities related to the suspension point. """

    def I_r_OP(self, t=None):
        """Position suspension point in inertial frame."""
        if t is None:
            t = self.t
        return (self.delta + self.epsilon * np.cos(self.omega * t)) * self.I_normal

    def I_v_P(self, t=None):
        """Velocity of the suspension point in inertial frame."""
        if t is None:
            t = self.t
        return -self.epsilon * self.omega * np.sin(self.omega * t) * self.I_normal


class SpatialPendulumWithConstraints(AbstractSpatialPendulum):
    pass


class SpatialPendulumWithoutConstraints(AbstractSpatialPendulum):
    def __init__(
        self,
        t,
        angles,
        d_angles,
        shape_cuboid,
        density,
        delta,
        epsilon,
        omega,
        damping,
        stability_method=None,
    ):
        super().__init__(
            t=t,
            shape_cuboid=shape_cuboid,
            density=density,
            delta=delta,
            epsilon=epsilon,
            omega=omega,
            damping=damping,
            n_dof=6,
            invertible=True,
            stability_method=stability_method,
        )
        self.x = np.concatenate([angles, d_angles])

    def I_r_OS(self, t=None, angles=None):
        """Position of the center of mass in inertial frame."""
        return self.I_r_OP(t=t) + self.A_IK(angles=angles) @ self.K_r_SP

    def I_v_S(self, t=None, angles=None, d_angles=None):
        """Velocity of the center of mass in inertial frame."""
        return self.I_v_P(t=t) + self.A_IK(angles=angles) @ (
            np.cross(self.K_Omega(angles=angles, d_angles=d_angles), self.K_r_SP)
        )

    def d_I_r_OS(self, t=None, angles=None, d_angles=None, variable="alpha"):
        """derivative of the center of mass in inertial frame."""
        match variable:
            case "t":
                return self.I_v_S(t=t, angles=angles, d_angles=d_angles)
            case ["alpha", "beta", "gamma"]:
                return (
                    self.d_A_IK(angles=angles, d_angles=d_angles, variable=variable)
                    @ self.K_r_SP
                )
            case _:
                return np.zeros(3)

        return (
            self.I_v_P(t=t)
            + self.d_A_IK(angles=angles, d_angles=d_angles) @ self.K_r_SP
        )

    def d_potential_energy(self, angles=None, variable="alpha"):
        """derivative of Potential energy w.r.t. alpha, beta, gamma."""
        return np.inner(
            -self.I_gravity, self.d_I_r_OS(angles=angles, variable=variable)
        )

    def translational_energy(self, t=None, angles=None, d_angles=None):
        """Translational kinetic energy."""
        v_S = self.I_v_S(t=t, angles=angles, d_angles=d_angles)
        return 0.5 * self.total_mass * np.inner(v_S, v_S)

    def d_translational_energy(
        self, t=None, angles=None, d_angles=None, variable="alpha"
    ):
        """Derivative of the translational kinetic energy."""
        v_S = self.I_v_S(t=t, angles=angles, d_angles=d_angles)
        d_v_S = self.d_I_r_OS(t=t, angles=angles, d_angles=d_angles, variable=variable)
        return self.total_mass * np.inner(v_S, d_v_S)

    @override
    def mass_secondorder(self, angles=None):
        """There is an additional contribution to the mass matrix from the translational kinetic energy."""
        mass = super().mass_secondorder(angles)
        mass -= (
            self.total_mass
            * self.A_IK(angles=angles)
            @ tilde_operator(self.K_r_SP)
            @ self.K_J_R(angles=angles)
        )  # TODO check whether this thing is symmetric!

    def M_small(self, t=None, x=None):
        return None

    def dynamics(self, t=None, x=None):
        return None


def trafo_around_x(angle):
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )


def d_trafo_around_x(angle):
    return np.array(
        [
            [0, 0, 0],
            [0, -np.sin(angle), -np.cos(angle)],
            [0, np.cos(angle), -np.sin(angle)],
        ]
    )


def trafo_around_z(angle):
    return np.array(
        [
            [np.cos(angle), np.sin(angle), 0],
            [-np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )


def d_trafo_around_z(angle):
    return np.array(
        [
            [-np.sin(angle), np.cos(angle), 0],
            [-np.cos(angle), -np.sin(angle), 0],
            [0, 0, 0],
        ]
    )


def tilde_operator(vector):
    """Skew-symmetric matrix for cross product."""
    return np.array(
        [
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0],
        ]
    )
