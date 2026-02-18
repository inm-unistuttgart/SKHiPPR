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
        spring,
        damping,
        q_all=["alpha", "beta", "gamma"],
        stability_method=None,
    ):
        self.num_constraints = len(q_all) - 3
        invertible = self.num_constraints == 0
        super().__init__(
            n_dof=2 * len(q_all) + self.num_constraints,
            autonomous=False,
            stability_method=stability_method,
            M_is_constant=False,
            invertible=invertible,
        )

        if q_all[:3] != ["alpha", "beta", "gamma"]:
            raise ValueError(
                "The first three generalized coordinates must be the angles alpha, beta, gamma."
            )
        self.q_all = q_all
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
        self.spring = spring
        self.damping = damping
        self.delta = delta
        self.epsilon = epsilon
        self.omega = omega
        self.K_r_SP = 0.5 * np.array(shape_cuboid)
        self.I_normal = np.array([0, 0, 1])

        self.I_gravity = -9.81 * np.array([0, 0, 1])

    """ Extract angles and their derivatives from x"""

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        if not len(value.shape) == 1:
            raise ValueError("x must be a 1-D array.")
        if not len(value) == self.n_dof:
            raise ValueError(f"x must have length {self.n_dof}.")
        self._x = value

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        if not np.isscalar(value):
            raise ValueError("t must be a scalar.")
        self._t = value

    @property
    def angles(self):
        return self.x[:3]

    @property
    def d_angles(self):
        return self.x[len(self.q_all) : len(self.q_all) + 3]

    @property
    def lam(self):
        return self.x[2 * len(self.q_all) :]

    """ Kardan angle transformation matrices and their derivatives. """

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
        return trafo_around_y(beta)

    def d_A_12(self, angles=None, d_angles=None, variable="beta"):
        match variable:
            case "beta":
                if angles is None:
                    angles = self.angles
                beta = angles[1]
                return d_trafo_around_y(beta)
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
        return trafo_around_x(gamma)

    def d_A_2K(self, angles=None, d_angles=None, variable="gamma"):
        match variable:
            case "gamma":
                if angles is None:
                    angles = self.angles
                gamma = angles[2]
                return d_trafo_around_x(gamma)
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

    """ Rotational quantities -- same in both formulations. """

    def K_J_R(self, angles=None):
        """Rotational Jacobian in body-fixed frame."""
        if angles is None:
            angles = self.angles
        _, beta, gamma = angles
        return np.array(
            [
                [-np.sin(beta), 0, 1],
                [np.cos(beta) * np.sin(gamma), np.cos(gamma), 0],
                [np.cos(beta) * np.cos(gamma), -np.sin(gamma), 0],
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
                        [-np.cos(beta), 0, 0],
                        [-np.sin(beta) * np.sin(gamma), 0, 0],
                        [-np.sin(beta) * np.cos(gamma), 0, 0],
                    ]
                )
            case "gamma":
                if angles is None:
                    angles = self.angles
                _, beta, gamma = angles
                return np.array(
                    [
                        [0, 0, 0],
                        [np.cos(beta) * np.cos(gamma), -np.sin(gamma), 0],
                        [-np.cos(beta) * np.sin(gamma), -np.cos(gamma), 0],
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
                return self.K_J_R(angles=angles)[:, 0]
            case "d_beta":
                return self.K_J_R(angles=angles)[:, 1]
            case "d_gamma":
                return self.K_J_R(angles=angles)[:, 2]

            case "alpha" | "beta" | "gamma":
                return self.d_K_J_R(angles=angles, variable=variable) @ d_angles

            case "t":
                raise ValueError(
                    "Time derivative of K_Omega requires 2nd derivatives of the angles and cannot be computed in this method. Use self.Mass_secondorder @ ddot_angles instead."
                )
            case _:
                return np.zeros(3)

    def generalized_damping_force(self, angles=None, d_angles=None):
        """Damping and spring torque in body-fixed frame."""
        K_damping = -self.damping * self.K_Omega(angles=angles, d_angles=d_angles)
        proj_matrix = np.vstack(
            [self.K_J_R(angles=angles).T, np.zeros((len(self.q_all) - 3, 3))]
        )
        # Spring force for rotation
        gen_spring_force = np.zeros(len(self.q_all))
        gen_spring_force[0] = (
            -self.spring * self.angles[0]
        )  # to break rotational symmetry
        return gen_spring_force + proj_matrix @ K_damping

    def rotational_energy(self, angles=None, d_angles=None):
        """Rotational kinetic energy."""
        K_Omega = self.K_Omega(angles=angles, d_angles=d_angles)
        return 0.5 * K_Omega.T @ self.K_inertia @ K_Omega

    def d_rotational_energy(self, angles=None, d_angles=None, variable="alpha"):
        """Derivative of the rotational kinetic energy."""
        K_Omega = self.K_Omega(angles=angles, d_angles=d_angles)
        d_K_Omega = self.d_K_Omega(angles=angles, d_angles=d_angles, variable=variable)
        return K_Omega.T @ self.K_inertia @ d_K_Omega

    def d_rotational_energy_dqdot(self, angles=None, d_angles=None):
        """Derivative of the rotational kinetic energy w.r.t. [d_alpha, d_beta, d_gamma]."""
        return (
            self.K_Omega(angles, d_angles).T
            @ self.K_inertia
            @ self.K_J_R(angles=angles)
        )

    def dd_rotational_energy_dqdot_dq(self, angles=None, d_angles=None):
        """Every component of d/dq(dT/dqdot)."""
        result = np.zeros((3, 3))
        for k, angle in enumerate(["alpha", "beta", "gamma"]):
            result[:, k] = (
                self.d_K_J_R(angles=angles, d_angles=d_angles, variable=angle).T
                @ self.K_inertia
                @ self.K_Omega(angles=angles, d_angles=d_angles)
            )
            result[:, k] += (
                self.K_J_R(angles=angles).T
                @ self.K_inertia
                @ self.d_K_Omega(angles=angles, d_angles=d_angles, variable=angle)
            )
        return result

    def mass_rotational_energy(self, angles=None):
        """Contributions to mass matrix from the rotational kinetic energy."""
        K_J_R = self.K_J_R(angles=angles)
        mass_rot = K_J_R.T @ self.K_inertia @ K_J_R  # acting on rotational DOFs
        mass = np.zeros((len(self.q_all), len(self.q_all)))
        mass[:3, :3] = mass_rot
        return mass

    def h_rotational_energy(self, angles=None, d_angles=None):
        """Contributions to h vector for angle coordinates from the rotational kinetic energy.
        h vector in the form d/dt(dT/dqdot) - dT/dq = M @ ddot_q - h = 0.
        h vector therefore includes - d/dq(dT/dqdot)*qdot + dT/dq."""

        if d_angles is None:
            d_angles = self.d_angles

        h = np.zeros(len(self.q_all))

        h[:3] = (
            -self.dd_rotational_energy_dqdot_dq(angles=angles, d_angles=d_angles)
            @ d_angles
        )
        for k, angle in enumerate(["alpha", "beta", "gamma"]):
            h[k] += self.d_rotational_energy(
                angles=angles, d_angles=d_angles, variable=angle
            )
        return h

    """ Potential energy and its derivative. """

    def potential_energy(self, I_r_OS=None):
        """Potential energy."""
        if I_r_OS is None:
            I_r_OS = self.I_r_OS()
        return -self.total_mass * np.inner(I_r_OS, self.I_gravity)

    def d_potential_energy(self, variable="alpha", **kwargs):
        """Derivative of the potential energy."""
        return -self.total_mass * np.inner(
            self.I_gravity, self.d_I_r_OS(variable=variable, **kwargs)
        )

    def h_potential_energy(self, **kwargs):
        """Contributions to h vector for angle coordinates from the potential energy.
        - dV/dq"""
        h = np.zeros(len(self.q_all))
        for k, var in enumerate(self.q_all):
            h[k] = -self.d_potential_energy(variable=var, **kwargs)
        return h

    """ Translational kinematic quantities """

    @abstractmethod
    def I_r_OS(self, **kwargs):
        """Position of the center of mass in inertial frame."""
        ...

    @abstractmethod
    def d_I_r_OS(self, variable, **kwargs):
        """Derivative of the position of the center of mass in inertial frame."""
        ...

    @abstractmethod
    def I_v_S(self, **kwargs):
        """Velocity of the center of mass in inertial frame."""
        ...

    @abstractmethod
    def d_I_v_S(self, variable, **kwargs): ...

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

    def I_a_P(self, t=None):
        """Acceleration of the suspension point in inertial frame."""
        if t is None:
            t = self.t
        return -self.epsilon * self.omega**2 * np.cos(self.omega * t) * self.I_normal

    """ Translational kinetic energy and its derivatives. """

    def translational_energy(self, **kwargs):
        v_S = self.I_v_S(**kwargs)
        return 0.5 * self.total_mass * np.inner(v_S, v_S)

    def d_translational_energy(self, variable, **kwargs):
        v_S = self.I_v_S(**kwargs)
        d_v_S = self.d_I_v_S(variable=variable, **kwargs)
        return self.total_mass * np.inner(v_S, d_v_S)

    @abstractmethod
    def mass_translational_energy(self, **kwargs):
        """Contributions to mass matrix from the translational kinetic energy."""
        ...

    @abstractmethod
    def h_translational_energy(self, **kwargs):
        """Contributions to h vector for angle coordinates from the translational kinetic energy.
        h vector in the form d/dt(dT/dqdot) - dT/dq = M @ ddot_q - h = 0.
        h vector therefore includes - d/dq(dT/dqdot)*qdot + dT/dq."""
        ...

    def constraints(self, **kwargs):
        """Constraint equations. Should be zero when constraints are satisfied."""
        return np.zeros(len(self.q_all) - 3)

    def W_constraints(self, **kwargs):
        """Transposed constraint Jacobian. Should be zero when constraints are satisfied."""
        return np.zeros((len(self.q_all) - 3, len(self.q_all))).T

    """ Assembly of first-order dynamics."""

    def h_all(self, t, x, **kwargs):
        """Assembly of h vector for all coordinates."""
        if t is not None:
            if not np.isscalar(t):
                raise ValueError("t must be a scalar.")
            self.t = t
        if x is not None:
            if not len(x.shape) == 1:
                raise ValueError("x must be a 1-D array.")
            self.x = x

        h = (
            self.h_rotational_energy()
            + self.h_translational_energy()
            + self.generalized_damping_force()
            + self.h_potential_energy()
        )

        if self.num_constraints > 0:
            h += self.W_constraints(t=t, x=x, **kwargs) @ self.lam
        return h

    def dynamics(self, t=None, x=None):
        if t is not None:
            self.t = t
        if x is not None:
            self.x = x

        # M = self.mass_rotational_energy() + self.mass_translational_energy()
        h = self.h_all(t=t, x=x)
        f = np.zeros(self.n_dof)
        # Kinematics qdot = qdot
        f[: len(self.q_all)] = self.x[len(self.q_all) : 2 * len(self.q_all)]
        # Dynamics M @ ddot_q = h
        f[len(self.q_all) : 2 * len(self.q_all)] = h
        # Constraints
        if self.num_constraints > 0:
            f[2 * len(self.q_all) :] = self.constraints()
        return f

    def M_small(self, t=None, x=None):
        """Pseudo-mass matrix for the first-order dynamics"""
        if t is not None:
            self.t = t
        if x is not None:
            self.x = x

        # Kinematics
        M = np.eye(self.n_dof)

        # Kinetics
        M[
            len(self.q_all) : 2 * len(self.q_all), len(self.q_all) : 2 * len(self.q_all)
        ] = (self.mass_rotational_energy() + self.mass_translational_energy())

        # Constraints
        M[2 * len(self.q_all) :, len(self.q_all) :] = 0

        return M


class SpatialPendulumWithConstraints(AbstractSpatialPendulum):
    def __init__(
        self,
        t,
        angles,
        d_angles,
        I_r_OS,
        I_v_S,
        lam,
        shape_cuboid,
        density,
        delta,
        epsilon,
        omega,
        spring,
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
            spring=spring,
            damping=damping,
            q_all=["alpha", "beta", "gamma", "x", "y", "z"],
            stability_method=stability_method,
        )
        self.x = np.concatenate([angles, I_r_OS, d_angles, I_v_S, lam])

    def I_r_OS(self, I_r_OS=None, **kwargs):
        if I_r_OS is None:
            return self.x[3:6]
        else:
            return I_r_OS

    def I_v_S(self, I_v_S=None, **kwargs):
        if I_v_S is None:
            return self.x[len(self.q_all) + 3 : len(self.q_all) + 6]
        else:
            return I_v_S

    def d_I_r_OS(self, variable, I_v_S=None, **kwargs):
        match variable:
            case "t":
                return self.I_v_S(I_v_S=I_v_S)
            case "x":
                return np.eye(3)[:, 0]
            case "y":
                return np.eye(3)[:, 1]
            case "z":
                return np.eye(3)[:, 2]
            case _:
                return np.zeros(3)

    def d_I_v_S(self, variable, **kwargs):
        match variable:
            case "t":
                raise ValueError(
                    "Time derivative of I_v_S requires 2nd derivatives of the state and cannot be computed in this method. Use self.Mass_secondorder @ ddot_angles instead."
                )
            case "d_x":
                return np.eye(3)[:, 0]
            case "d_y":
                return np.eye(3)[:, 1]
            case "d_z":
                return np.eye(3)[:, 2]
            case _:
                return np.zeros(3)

    def h_translational_energy(self, **kwargs):
        return np.zeros(len(self.q_all))

    def mass_translational_energy(self, **kwargs):
        return (
            np.block(
                [[np.zeros((3, 3)), np.zeros((3, 3))], [np.zeros((3, 3)), np.eye(3)]]
            )
            * self.total_mass,
        )

    def constraints(self, t=None, I_r_OS=None, angles=None, **kwargs):
        I_r_OP = (
            self.I_r_OS(I_r_OS=I_r_OS, **kwargs)
            + self.A_IK(angles=angles) @ self.K_r_SP
        )
        return I_r_OP - self.I_r_OP(t=t, **kwargs)

    def W_constraints(self, t=None, angles=None, I_r_OS=None, **kwargs):
        dg_dal = np.zeros((3, 3))
        for k, angle in enumerate(["alpha", "beta", "gamma"]):
            dg_dal[:, k] = self.d_A_IK(angles=angles, variable=angle) @ self.K_r_SP
        return np.vstack((dg_dal.T, np.eye(3)))


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
        spring,
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
            spring=spring,
            q_all=["alpha", "beta", "gamma"],
            stability_method=stability_method,
        )
        self.x = np.concatenate([angles, d_angles])

    def I_r_OS(self, t=None, angles=None):
        """Position of the center of mass in inertial frame."""
        K_r_PS = -self.K_r_SP
        I_r_PS = self.A_IK(angles=angles) @ K_r_PS
        return self.I_r_OP(t=t) + I_r_PS

    def I_v_S(self, t=None, angles=None, d_angles=None):
        """Velocity of the center of mass in inertial frame."""
        A_IK = self.A_IK(angles=angles)
        K_Omega = self.K_Omega(angles=angles, d_angles=d_angles)
        K_r_PS = -self.K_r_SP
        return self.I_v_P(t=t) + A_IK @ np.cross(K_Omega, K_r_PS)

    def d_I_r_OS(self, t=None, angles=None, d_angles=None, variable="alpha"):
        """derivative of the center of mass in inertial frame."""
        match variable:
            case "t":
                # return self.I_v_S(t=t, angles=angles, d_angles=d_angles)
                return (
                    self.I_v_P(t=t)
                    - self.d_A_IK(angles=angles, d_angles=d_angles, variable="t")
                    @ self.K_r_SP
                )
            case "alpha" | "beta" | "gamma":
                return -self.d_A_IK(angles=angles, variable=variable) @ self.K_r_SP
            case _:
                return np.zeros(3)

    def d_I_v_S(self, t=None, angles=None, d_angles=None, variable="alpha"):
        """derivative of the velocity of the center of mass in inertial frame.
        v_S = v_P + A_IK @ tilde(K_r_SP) @ K_J_R @ d_angles

        """
        match variable:
            case "t":
                raise ValueError(
                    "Time derivative of I_v_S requires 2nd derivatives of the angles and cannot be computed in this method. "
                )
            case "alpha" | "beta" | "gamma":
                r_tilde = tilde_operator(self.K_r_SP)
                A_IK = self.A_IK(angles=angles)
                K_J_R = self.K_J_R(angles=angles)
                if d_angles is None:
                    d_angles = self.d_angles

                result = (
                    self.d_A_IK(angles=angles, variable=variable)
                    @ r_tilde
                    @ K_J_R
                    @ d_angles
                )
                result += (
                    A_IK
                    @ r_tilde
                    @ self.d_K_J_R(angles=angles, variable=variable)
                    @ d_angles
                )
                return result
            case "d_alpha" | "d_beta" | "d_gamma":
                dv_dqdot = (
                    self.A_IK(angles=angles)
                    @ tilde_operator(self.K_r_SP)
                    @ self.K_J_R(angles=angles)
                )
                for k, angle in enumerate(["d_alpha", "d_beta", "d_gamma"]):
                    if variable == angle:
                        return dv_dqdot[:, k]
            case _:
                return np.zeros(3)

    def d_transl_dqdot(self, t=None, angles=None, d_angles=None):
        """Derivative of the translational kinetic energy w.r.t. [d_alpha, d_beta, d_gamma]."""
        K_J_R = self.K_J_R(angles=angles)
        A_KI = self.A_KI(angles=angles)
        r_tilde = tilde_operator(self.K_r_SP)
        v = self.I_v_S(t=t, angles=angles, d_angles=d_angles)
        return -self.total_mass * K_J_R.T @ r_tilde @ A_KI @ v

    def h_translational_energy(self, **kwargs):
        """Contributions to h vector for angle coordinates from the translational kinetic energy.
        h vector in the form d/dt(dT/dqdot) - dT/dq = M @ ddot_q - h = 0.
        h vector therefore includes -partial_t(dT/dqdot) -d/dq(dT/dqdot)*qdot + dT/dq.
        """
        angles = kwargs.get("angles", self.angles)
        d_angles = kwargs.get("d_angles", self.d_angles)
        t = kwargs.get("t", self.t)

        K_J_R = self.K_J_R(angles=angles)
        A_KI = self.A_KI(angles=angles)
        r_tilde = tilde_operator(self.K_r_SP)
        v = self.I_v_S(t=t, angles=angles, d_angles=d_angles)

        # partial time derivative
        h = self.total_mass * K_J_R.T @ r_tilde @ A_KI @ self.I_a_P(t=t)
        for k, var in enumerate(["alpha", "beta", "gamma"]):
            # dT/dq term
            h[k] += self.d_translational_energy(
                variable=var, angles=angles, d_angles=d_angles, **kwargs
            )

            # d/dq (dT/dqdot) qdot terms by product rule
            h += (
                d_angles[k]
                * self.total_mass
                * self.d_K_J_R(angles=angles, variable=var).T
                @ r_tilde
                @ A_KI
                @ v
            )
            h += (
                d_angles[k]
                * self.total_mass
                * K_J_R.T
                @ r_tilde
                @ self.d_A_KI(angles=angles, variable=var)
                @ v
            )
            h += (
                d_angles[k]
                * self.total_mass
                * K_J_R.T
                @ r_tilde
                @ A_KI
                @ self.d_I_v_S(t=t, angles=angles, d_angles=d_angles, variable=var)
            )
        return h

    def mass_translational_energy(self, angles=None, **kwargs):
        """Contributions to mass matrix from the translational kinetic energy."""
        K_J_R = self.K_J_R(angles=angles)
        r_tilde = tilde_operator(self.K_r_SP)
        mass = -self.total_mass * K_J_R.T @ r_tilde @ r_tilde @ K_J_R
        return mass


class SpatialPendulumWithMassInverted(SpatialPendulumWithoutConstraints):
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
        spring,
        damping,
        stability_method=None,
    ):
        super().__init__(
            t,
            angles,
            d_angles,
            shape_cuboid,
            density,
            delta,
            epsilon,
            omega,
            spring,
            damping,
            stability_method,
        )
        self.M_is_constant = True
        self.M_invertible = True

    def M_small(self, t=None, x=None):
        return np.eye(self.n_dof)

    def dynamics(self, t=None, x=None):
        f = super().dynamics(t=t, x=x)
        M = super().M_small(t=t, x=x)
        return np.linalg.solve(M, f)


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


def trafo_around_y(angle):
    return np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )


def d_trafo_around_y(angle):
    return np.array(
        [
            [-np.sin(angle), 0, np.cos(angle)],
            [0, 0, 0],
            [-np.cos(angle), 0, -np.sin(angle)],
        ]
    )


def trafo_around_z(angle):
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )


def d_trafo_around_z(angle):
    return np.array(
        [
            [-np.sin(angle), -np.cos(angle), 0],
            [np.cos(angle), -np.sin(angle), 0],
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


def trafo_for_plot(a=2, b=1.5, c=1):
    """Transformation matrix from body-fixed to inertial frame for plotting."""

    # y axis of I frame points upwards and along r_SP
    K_ey_I = np.array([a, b, c])

    # x axis of I frame is chosen orthogonal with z component zero (arbitrary choice)
    K_ex_I = np.array([-b, a, 0])

    # z axis of I frame to complete right-handed system
    K_ez_I = np.cross(K_ex_I, K_ey_I)

    # Normalize to get orthonormal basis
    K_ex_I = K_ex_I / np.linalg.norm(K_ex_I)
    K_ey_I = K_ey_I / np.linalg.norm(K_ey_I)
    K_ez_I = K_ez_I / np.linalg.norm(K_ez_I)

    # Construct A_{IK} = A_{KI}.T from the basis vectors
    A_IK = np.vstack((K_ex_I, K_ey_I, K_ez_I))
    print(f"a={a}, b={b}, c={c}")
    print("A_IK for plotting:")
    print(A_IK)


if __name__ == "__main__":
    trafo_for_plot()
