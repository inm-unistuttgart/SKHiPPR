import numpy as np
from typing import override
from skhippr.odes.AbstractODE import AbstractODE  # , SecondOrderODE


class Duffing(AbstractODE):
    """
    Non-autonomous Duffing oscillator as concrete subclass of :py:class:`~skhippr.odes.AbstractODE.AbstractODE`. ::

        dx[0]/dt = x[1]
        dx[1]/dt = -alpha * x[0] - delta * x[1] - beta * [0]**3 + F * cos(omega * t)

    """

    def __init__(
        self,
        t: float,
        x: np.ndarray,
        omega: float,
        alpha: float,
        beta: float,
        F: float,
        delta: float,
        exponent=3,
    ):
        super().__init__(autonomous=False, n_dof=2)
        self.t = t
        self.x = x
        self.alpha = alpha
        self.beta = beta
        self.F = F
        self.omega = omega
        self.delta = delta
        self.exponent = exponent

    @override
    def dynamics(self, t=None, x=None) -> np.ndarray:

        if t is None:
            t = self.t
        if x is None:
            x = self.x
        self.check_dimensions(t, x)

        f = np.zeros_like(x)
        f[0, ...] = x[1, ...]
        f[1, ...] = (
            -self.alpha * x[0, ...]
            - self.delta * x[1, ...]
            - self.beta * x[0, ...] ** self.exponent
            + self.F * np.cos(self.omega * t)
        )

        return f

    @override
    def closed_form_derivative(self, variable, t=None, x=None):

        if t is None:
            t = self.t
        if x is None:
            x = self.x

        self.check_dimensions(t, x)

        match variable:
            case "x":
                return self.df_dx(t, x)
            case "omega":
                return self.df_dom(t, x)
            case "F":
                return self.df_dF(t, x)
            case "alpha":
                return self.df_dalpha(t, x)
            case "beta":
                return self.df_dbeta(t, x)
            case "delta":
                return self.df_ddelta(t, x)
            case _:
                raise NotImplementedError(
                    f"Derivative w.r.t {variable} not implemented in closed form."
                )

    def df_dx(self, t=None, x=None):

        if t is None:
            t = self.t
        if x is None:
            x = self.x

        df_dx = np.zeros((2, *x.shape), dtype=x.dtype)
        df_dx[0, 1, ...] = 1
        df_dx[1, 0, ...] = -self.alpha - self.exponent * self.beta * x[0, ...] ** (
            self.exponent - 1
        )
        df_dx[1, 1, ...] = -self.delta

        return df_dx

    def df_dalpha(self, t=None, x=None):

        if x is None:
            x = self.x

        df_dal = np.zeros_like(x)
        df_dal[1, ...] = -x[0, ...]

        return df_dal[:, np.newaxis, ...]

    def df_dbeta(self, t=None, x=None):

        if x is None:
            x = self.x

        df_dbe = np.zeros_like(x)
        df_dbe[1, ...] = -x[0, ...] ** self.exponent
        return df_dbe[:, np.newaxis, ...]

    def df_ddelta(self, t=None, x=None):

        if x is None:
            x = self.x
        df_ddel = np.zeros_like(x)
        df_ddel[1, ...] = -x[1, ...]

        return df_ddel[:, np.newaxis, ...]

    def df_dF(self, t=None, x=None):

        if t is None:
            t = self.t
        if x is None:
            x = self.x

        df_dF = np.zeros_like(x)
        df_dF[1, ...] = np.cos(self.omega * t)
        return df_dF[:, np.newaxis, ...]

    def df_dom(self, t=None, x=None):

        if t is None:
            t = self.t
        if x is None:
            x = self.x

        df_dom = np.zeros_like(x)
        df_dom[1, ...] = -self.F * t * np.sin(self.omega * t)
        return df_dom[:, np.newaxis, ...]


class CubicQuadratic(AbstractODE):
    """
    Non-autonomous oscillator with quadratic and cubic nonlinearity as concrete subclass of :py:class:`~skhippr.odes.AbstractODE.AbstractODE`. ::

        dx[0]/dt = x[1]
        dx[1]/dt = -alpha * x[0] - delta * x[1] - beta * x[0]**2  - gamma*x[0]**3 + F * cos(omega * t)

    """

    def __init__(
        self,
        t: float,
        x: np.ndarray,
        omega: float,
        alpha: float,
        beta: float,
        gamma: float,
        F: float,
        delta: float,
    ):
        super().__init__(autonomous=False, n_dof=2)
        self.t = t
        self.x = x
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.F = F
        self.omega = omega
        self.delta = delta

    @override
    def dynamics(self, t=None, x=None) -> np.ndarray:

        if t is None:
            t = self.t
        if x is None:
            x = self.x
        self.check_dimensions(t, x)

        f = np.zeros_like(x)
        f[0, ...] = x[1, ...]
        f[1, ...] = (
            -self.alpha * x[0, ...]
            - self.delta * x[1, ...]
            - self.beta * x[0, ...] ** 2
            - self.gamma * x[0, ...] ** 3
            + self.F * np.cos(self.omega * t)
        )

        return f

    @override
    def closed_form_derivative(self, variable, t=None, x=None):

        if t is None:
            t = self.t
        if x is None:
            x = self.x

        self.check_dimensions(t, x)

        match variable:
            case "x":
                return self.df_dx(t, x)
            case "omega":
                return self.df_dom(t, x)
            case "F":
                return self.df_dF(t, x)
            case "alpha":
                return self.df_dalpha(t, x)
            case "beta":
                return self.df_dbeta(t, x)
            case "gamma":
                return self.df_dgamma(t, x)
            case "delta":
                return self.df_ddelta(t, x)
            case _:
                raise NotImplementedError(
                    f"Derivative w.r.t {variable} not implemented in closed form."
                )

    def df_dx(self, t=None, x=None):

        if t is None:
            t = self.t
        if x is None:
            x = self.x

        df_dx = np.zeros((2, *x.shape), dtype=x.dtype)
        df_dx[0, 1, ...] = 1
        df_dx[1, 0, ...] = (
            -self.alpha - 2 * self.beta * x[0, ...] - 3 * self.gamma * x[0, ...] ** 2
        )
        df_dx[1, 1, ...] = -self.delta

        return df_dx

    def df_dalpha(self, t=None, x=None):

        if x is None:
            x = self.x

        df_dal = np.zeros_like(x)
        df_dal[1, ...] = -x[0, ...]

        return df_dal[:, np.newaxis, ...]

    def df_dbeta(self, t=None, x=None):

        if x is None:
            x = self.x

        df_dbe = np.zeros_like(x)
        df_dbe[1, ...] = -x[0, ...] ** 2
        return df_dbe[:, np.newaxis, ...]

    def df_dgamma(self, t=None, x=None):

        if x is None:
            x = self.x

        df_dbe = np.zeros_like(x)
        df_dbe[1, ...] = -x[0, ...] ** 3
        return df_dbe[:, np.newaxis, ...]

    def df_ddelta(self, t=None, x=None):

        if x is None:
            x = self.x
        df_ddel = np.zeros_like(x)
        df_ddel[1, ...] = -x[1, ...]

        return df_ddel[:, np.newaxis, ...]

    def df_dF(self, t=None, x=None):

        if t is None:
            t = self.t
        if x is None:
            x = self.x

        df_dF = np.zeros_like(x)
        df_dF[1, ...] = np.cos(self.omega * t)
        return df_dF[:, np.newaxis, ...]

    def df_dom(self, t=None, x=None):

        if t is None:
            t = self.t
        if x is None:
            x = self.x

        df_dom = np.zeros_like(x)
        df_dom[1, ...] = -self.F * t * np.sin(self.omega * t)
        return df_dom[:, np.newaxis, ...]


class Quadratic(CubicQuadratic):
    """
    Non-autonomous nonlinear oscillator with quadratic nonlinearity. ::

        dx[0]/dt = x[1]
        dx[1]/dt = -alpha * x[0] - delta * x[1] - beta * [0]**2 + F * cos(omega * t)

    """

    def __init__(
        self,
        t: float,
        x: np.ndarray,
        omega: float,
        alpha: float,
        beta: float,
        F: float,
        delta: float,
    ):
        super().__init__(
            t=t, x=x, omega=omega, alpha=alpha, beta=beta, gamma=0, F=F, delta=delta
        )


# class NLTVA(SecondOrderODE):
#     def __init__(self, t: float, q, dq, omega, m1, m2, c1, c2, k1, k2, k_nl1, k_nl2, F):
#         M = np.diag((m1, m2))
#         D = np.array([[c1 + c2, -c2], [-c2, c2]])
#         K = np.array([[k1 + k2, -k2], [-k2, k2]])
#         super().__init__(t, q, dq, M, D, K, stability_method=None)
#         self.omega = omega
#         self.k_nl1 = k_nl1
#         self.k_nl2 = k_nl2
#         self.F = F

#     @override
#     def f_nonlin(self, t=None, q=None, dq=None) -> np.ndarray:
#         if q is None:
#             q = self.x[: self.n_dof]
#         if t is None:
#             t = self.t

#         f = np.zeros_like(q)
#         f[0, ...] = (
#             self.k_nl1 * q[0, ...] ** 3 + self.k_nl2 * (q[0, ...] - q[1, ...]) ** 3
#         )
#         f[0, ...] -= self.F * np.cos(self.omega * t)
#         f[1, ...] = self.k_nl2 * (q[1, ...] - q[0, ...]) ** 3
#         return f

#     def derivative_f_nonlin(self, variable, t, q=None, dq=None):
#         if q is None:
#             q = self.x[: self.n_dof]
#         match variable:
#             case "q":
#                 df_dq = np.zeros((q.shape[0], *q.shape))
#                 df_dq[0, 0, ...] = (
#                     3 * self.k_nl1 * q[0, ...] ** 2
#                     + 3 * self.k_nl2 * (q[0, ...] - q[1, ...]) ** 2
#                 )
#                 df_dq[0, 1, ...] = -3 * self.k_nl2 * (q[0, ...] - q[1, ...]) ** 2
#                 df_dq[1, 0, ...] = -3 * self.k_nl2 * (q[0, ...] - q[1, ...]) ** 2
#                 df_dq[1, 1, ...] = 3 * self.k_nl2 * (q[0, ...] - q[1, ...]) ** 2
#                 return df_dq
#             case "F":
#                 return np.array([-np.cos(self.omega * t), 0.0])[:, np.newaxis, ...]
#             case "k_nl1 | k_nl2":
#                 raise NotImplementedError
#             case "dq":
#                 return np.zeros((q.shape[0], *q.shape))
#             case _:
#                 return np.zeros_like(q)


class NLTVA_FO(AbstractODE):
    """Nonlinear Tuned Vibration Absorber in first-order form."""

    def __init__(self, t, x, k, c, m, k_nl, F, omega):
        """
        Initialize the NLTVA in first-order form.

        Parameters
        ----------
        t : float
            Time variable.
        x : np.ndarray
            State vector.
        k : float
            Linear stiffness.
        c : float
            Damping coefficient.
        m : float
            Mass.
        k_nl : float
            Nonlinear stiffness coefficient.
        F : float
            Forcing amplitude.
        omega : float
            Forcing frequency.
        """
        super().__init__(False, 4)
        self.knl = k_nl
        self.k = k
        self.c = c
        self.m = m
        self.t = t
        self.x = x
        self.F = F
        self.omega = omega

    def dynamics(self, t=None, x=None):
        if x is None:
            x = self.x
        if t is None:
            t = self.t
        self.check_dimensions(t, x)

        q, dq = np.split(x, 2, axis=0)
        q_rel = q[0, ...] - q[1, ...]
        v_rel = dq[0, ...] - dq[1, ...]
        rhs = np.zeros_like(q)
        rhs[0, ...] = self.c[0] * dq[0] + self.c[1] * v_rel
        rhs[0, ...] += self.k[0] * q[0] + self.k[1] * q_rel
        rhs[0, ...] += self.knl[0] * q[0] ** 3 + self.knl[1] * q_rel**3
        rhs[0, ...] -= self.F * np.cos(self.omega * t)
        rhs[0, ...] /= -self.m[0]

        rhs[1, ...] = -self.c[1] * v_rel
        rhs[1, ...] -= self.k[1] * q_rel
        rhs[1, ...] -= self.knl[1] * q_rel**3
        rhs[1, ...] /= -self.m[1]

        return np.squeeze(np.vstack((dq[:, np.newaxis, ...], rhs[:, np.newaxis, ...])))

    def closed_form_derivative(self, variable, t=None, x=None):

        if x is None:
            x = self.x
        if t is None:
            t = self.t
        self.check_dimensions(t, x)

        match variable:
            case "x":
                q, _ = np.split(x, 2, axis=0)
                q_rel = q[0] - q[1]

                df_dx = np.zeros((x.shape[0], *x.shape), dtype=x.dtype)
                df_dx[0, 2, ...] = 1
                df_dx[1, 3, ...] = 1

                df_dx[2, 0, ...] = (
                    -self.k[0]
                    - self.k[1]
                    - 3 * self.knl[0] * q[0] ** 2
                    - 3 * self.knl[1] * q_rel**2
                ) / self.m[0]
                df_dx[2, 1, ...] = (self.k[1] + 3 * self.knl[1] * q_rel**2) / self.m[0]
                df_dx[2, 2, ...] = (-self.c[0] - self.c[1]) / self.m[0]
                df_dx[2, 3, ...] = self.c[1] / self.m[0]

                df_dx[3, 0, ...] = (self.k[1] + 3 * self.knl[1] * q_rel**2) / self.m[1]
                df_dx[3, 1, ...] = (-self.k[1] - 3 * self.knl[1] * q_rel**2) / self.m[1]
                df_dx[3, 2, ...] = self.c[1] / self.m[1]
                df_dx[3, 3, ...] = -self.c[1] / self.m[1]

                return df_dx
            case "F":
                df_dF = np.zeros_like(x)
                df_dF[2, ...] = -np.cos(self.omega * t) / self.m[1]
                return df_dF[:, np.newaxis, ...]
            case "omega":
                return np.zeros_like(x)
            case _:
                raise NotImplementedError(
                    f"Closed-form derivative w.r.t {variable} not implemented."
                )
