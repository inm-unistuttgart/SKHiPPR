class BlockOnBelt(AbstractODE):
    """
    Smoothed Block-on-belt system as a subclass of :py:class:`~skhippr.odes.AbstractODE.AbstractODE`. A block with mass ``m`` is placed on a belt with constant velocity ``vdr``. The block is subject to a spring force ``Fs`` and smoothed coulomb friction with the belt. The equations of motion are ::

    dx[0]/dt = x[1]
    dx[1]/dt = -k/m * x[0] + Fs/m * 2/pi * arctan(epsilon * (x[1] - vdr)) / (1 + delta * abs(x[1] - vdr))

    """

    def __init__(
        self,
        x: np.ndarray,
        epsilon: float,
        k: float,
        m: float,
        Fs: float,
        vdr: float,
        delta: float,
    ):
        super().__init__(True, 2)
        self.x = x
        self.epsilon = epsilon
        self.k = k
        self.m = m
        self.Fs = Fs
        self.vdr = vdr
        self.delta = delta

        self.check_dimensions(t=None, x=x)

    def dynamics(self, t=None, x=None):
        if x is None:
            x = self.x

        self.check_dimensions(t=t, x=x)

        gamma_T = x[1, ...] - self.vdr
        F_T = (
            -self.Fs
            / (1 + self.delta * np.abs(gamma_T))
            * 2
            / np.pi
            * np.arctan(self.epsilon * gamma_T)
        )

        f = np.zeros_like(x)
        f[0, ...] = x[1, ...]
        f[1, ...] = -self.k / self.m * x[0, ...] + F_T / self.m
        return f

    def closed_form_derivative(self, variable, t=None, x=None):
        if x is None:
            x = self.x

        self.check_dimensions(t=t, x=x)

        if variable == "x":
            gamma_T = x[1, ...] - self.vdr
            df_dx = np.zeros((2, 2, *x.shape[1:]), dtype=x.dtype)
            df_dx[0, 1, ...] = 1
            df_dx[1, 0, ...] = -self.k / self.m
            df_dx[1, 1, ...] = (self.Fs / self.m) * (
                (
                    np.arctan(self.epsilon * gamma_T)
                    / (1 + self.delta * np.abs(gamma_T)) ** 2
                )
                * (self.delta * np.sign(gamma_T) * 2 / np.pi)
                - (1 / (1 + self.delta * np.abs(gamma_T)))
                * (2 / np.pi)
                / (1 + (self.epsilon * gamma_T) ** 2)
                * self.epsilon
            )
            return df_dx
        else:
            raise NotImplementedError(
                f"Derivative w.r.t {variable} not implemented in closed form."
            )
