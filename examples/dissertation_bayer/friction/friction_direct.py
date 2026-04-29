import numpy as np
import matplotlib.pyplot as plt

from skhippr.solvers.newton import ScipyRootSolver, NewtonSolver
from skhippr.equations.AbstractEquation import AbstractEquation
from skhippr.Fourier import Fourier
from skhippr.odes.daes import FrictionOscillator, SmoothedFrictionOscillator


def solve_friction(
    oscillator: FrictionOscillator,
    fourier: Fourier,
    omega,
    smoothing,
    solver: NewtonSolver = None,
    initial_guess=None,
):
    """Solve substituted friction problem and return a FrictionDirect object.

    This function constructs a :class:`FrictionDirect` instance for the given
    oscillator and Fourier settings, then solves the reduced problem for the
    Lagrange multiplier coefficients ``Lambda``.

    In the nonsmooth case, if no initial guess is provided, the function
    performs a warm-start by first solving a smoothed version of the problem.

    Parameters
    ----------
    oscillator : FrictionOscillator or SmoothedFrictionOscillator
        Oscillator object providing mechanical parameters (must expose
        attributes like ``stiffnesses``, ``dampings``, ``masses``, ``mu``,
        ``forcing_amplitudes``, ``forcing_phases``, and optionally
        ``prox_parameter``).
    fourier : Fourier
        Fourier helper object used for HBM/DFT settings (``N_HBM``,
        ``L_DFT``, ``real_formulation``).
    omega : float
        Excitation frequency.
    smoothing : float
        Smoothing parameter; if ``np.inf`` a non-smooth proximal constraint
        is used (requires ``prox_parameter`` on the oscillator), otherwise a
        smooth ``tanh``-based constraint is used.
    solver : NewtonSolver, optional
        Solver instance used to solve the reduced system. If
        ``None``, a default :class:`ScipyRootSolver` is constructed.
    initial_guess : ndarray or None, optional
        Initial guess for the full ``Lambda`` coefficients. If ``None``, a
        zero vector of appropriate length is used.

    Returns
    -------
    FrictionDirect
        A solved :class:`FrictionDirect` instance containing the converged
        ``Lambda_odd`` coefficients and allowing retrieval of inferred states
        (via methods such as :meth:`x_time`).
    """

    if solver is None:
        solver = ScipyRootSolver(
            tolerance=1e-8,
            max_iterations=1000000,
            verbose=True,
            use_fprime=False,
            method="lm",
        )

    try:
        prox_parameter = oscillator.prox_parameter
    except AttributeError:
        if smoothing == np.inf:
            raise ValueError("Must specify prox_parameter for non-smooth oscillator.")
        prox_parameter = 0

    g = oscillator.lam_crit / (oscillator.mu * oscillator.masses[-1])

    warmstart = smoothing == np.inf and initial_guess is None
    if initial_guess is None:
        initial_guess = np.zeros(2 * fourier.N_HBM + 1)

    equ = FrictionDirect(
        N_HBM=fourier.N_HBM,
        L_DFT=fourier.L_DFT,
        real_formulation=fourier.real_formulation,
        stiffnesses=oscillator.stiffnesses,
        dampings=oscillator.dampings,
        masses=oscillator.masses,
        g=g,
        mu=oscillator.mu,
        forcing_amplitudes=oscillator.forcing_amplitudes,
        forcing_phases=oscillator.forcing_phases,
        omega=omega,
        smoothing=smoothing,
        prox_parameter=prox_parameter,
        initial_guess=initial_guess,
    )

    if warmstart:  # warm-start with smoothed solution
        if solver.verbose:
            print("Solving smoothed lambda problem for warm-start...")

        equ.smoothing = 40
        solver.solve_equation(equ, unknown="Lambda_odd")
        equ.smoothing = smoothing

    if solver.verbose:
        print(
            f"Solving lambda problem (smoothing = {equ.smoothing}). Residual before: {np.linalg.norm(equ.residual(update=True))}..."
        )

    solver.solve_equation(equ, unknown="Lambda_odd")

    return equ


class FrictionDirect(AbstractEquation):
    """Substituted formulation of the friction oscillator using Harmonic Balance Method.

    This class solves the tanh-smoothed or nonsmooth frictional oscillator HBM problem by treating the
    odd Fourier coefficients of the friction force  (Lambda) as the primary unknowns and inferring the remaining
    state variables through a direct substitution method.

    Parameters
    ----------
    N_HBM : int, optional
        Number of Fourier harmonics in the harmonic balance approximation. Default is 60.
    L_DFT : int, optional
        Length of the time domain grid for Discrete Fourier Transform. Default is 4096.
    real_formulation : bool, optional
        If True, uses real-valued Fourier series representation. Default is True.
    stiffnesses : float or array-like, optional
        Stiffness coefficients for the system. Default is 1.
    dampings : float or array-like, optional
        Damping coefficients for the system. Default is 0.1.
    masses : tuple or array-like, optional
        Mass values for each degree of freedom. Default is (1, 1).
    g : float, optional
        Gravitational acceleration (or normalized vertical load). Default is 9.81.
    mu : float, optional
        Coefficient of friction. Default is 1.
    forcing_amplitudes : tuple or array-like, optional
        Amplitudes of harmonic forcing on each DOF. Default is (1, 0).
    forcing_phases : tuple or array-like, optional
        Phase angles of forcing on each DOF. Default is (0, 0).
    omega : float, optional
        Excitation frequency. Default is 1.
    smoothing : float, optional
        Smoothing parameter for tanh-smoothing. If np.inf, uses non-smooth
        proximal formulation. Default is np.inf.
    prox_parameter : float, optional
        Prox parameter used for non-smooth constraint. Default is 1.
    initial_guess : ndarray, optional
        Initial guess for Lambda coefficients. If None, defaults to zeros.

    Attributes
    ----------
    masses : ndarray
        Mass values for each DOF.
    n_dof : int
        Number of degrees of freedom. (Caution: Really degrees of freedom (i.e. number of masses), not number of states!)
    g : float
        Gravitational/normal force parameter.
    mu : float
        Coefficient of friction.
    forcing_amplitudes : ndarray
        Forcing amplitudes for each DOF.
    forcing_phases : ndarray
        Forcing phase angles for each DOF - k-th mass forced by sin(omega*t + phases[k])
    smoothing : float
        Current smoothing parameter.
    prox_parameter : float
        Proximal parameter for constraints.
    fourier : Fourier
        Fourier series object for HBM approximations.
    omega : float
        Excitation frequency.
    M : ndarray
        Mass matrix in Fourier coefficient space.
    C : ndarray
        Damping matrix in Fourier coefficient space.
    K : ndarray
        Stiffness matrix in Fourier coefficient space.
    Z : ndarray
        Combined impedance matrix (M,C,K) in freq. domain.
    F : ndarray
        Forcing vector in Fourier coefficient space.
    W : ndarray
        generalized force direction in freq. domain
    lam_crit : float
        Critical friction force value.
    N_odd : int
        Number of odd harmonics in the approximation.
    Lambda : ndarray
        Full Lambda vector, internally reconstructed from odd harmonics.
    Lambda_odd : ndarray
        Odd harmonic coefficients of the Lagrange multiplier.
    """

    def __init__(
        self,
        N_HBM=60,
        L_DFT=4096,
        real_formulation=True,
        stiffnesses=1,
        dampings=0.1,
        masses=(1, 1),
        g=9.81,
        mu=1,
        forcing_amplitudes=(1, 0),
        forcing_phases=(0, 0),
        omega=1,
        smoothing=np.inf,
        prox_parameter=1,
        initial_guess=None,
    ):
        """Initialize a FrictionDirect solver instance."""
        super().__init__(stability_method=None)

        # passed arguments
        self.masses = np.atleast_1d(masses)
        self.n_dof = len(self.masses)
        self.g = g
        self.mu = mu
        self.forcing_amplitudes = np.atleast_1d(forcing_amplitudes)
        self.forcing_phases = np.atleast_1d(forcing_phases)
        self.smoothing = smoothing
        self.prox_parameter = prox_parameter
        self.omega = omega

        # Fourier object and derivative operator
        self.fourier = Fourier(
            N_HBM=N_HBM, L_DFT=L_DFT, n_dof=1, real_formulation=real_formulation
        )
        D = self.fourier.derivative_matrix

        # initial guess
        if initial_guess is None:
            initial_guess = np.zeros((2 * self.fourier.N_HBM + 1))
        self.Lambda = initial_guess

        # assemble system matrices in time domain
        M = np.diag(self.masses)
        stiffnesses = np.append(stiffnesses, 0)
        dampings = np.append(dampings, 0)

        K = np.zeros((len(masses), len(masses)))
        C = np.zeros(K.shape)
        for i in range(len(masses)):

            K[i, i] = stiffnesses[i + 1] + stiffnesses[i]
            if i > 0:
                K[i, i - 1] = -stiffnesses[i]
            if i < len(masses) - 1:
                K[i, i + 1] = -stiffnesses[i + 1]

            C[i, i] = dampings[i + 1] + dampings[i]
            if i > 0:
                C[i, i - 1] = -dampings[i]
            if i < len(masses) - 1:
                C[i, i + 1] = -dampings[i + 1]

        # transform system matrices to Freq domain
        self.M = np.kron(self.omega**2 * D @ D, M)
        self.C = np.kron(self.omega * D, C)
        self.K = np.kron(np.eye(2 * self.fourier.N_HBM + 1), K)
        self.Z = self.M + self.C + self.K

        # evaluate Fourier coeffs of forcing
        t = self.fourier.time_samples(self.omega)
        arguments = self.forcing_phases[:, np.newaxis] + self.omega * t[np.newaxis, :]
        forcing = self.forcing_amplitudes[:, np.newaxis] * np.sin(arguments)

        forcing_FC = np.zeros((self.n_dof, 2 * self.fourier.N_HBM + 1), dtype=complex)
        for i in range(self.n_dof):
            forcing_FC[i, :] = self.fourier.DFT(np.atleast_2d(forcing[i, :]))

        self.F = forcing_FC.flatten(order="F")
        W_time = np.zeros(self.n_dof)
        W_time[-1] = 1
        self.W = np.kron(np.eye(2 * self.fourier.N_HBM + 1), W_time[:, np.newaxis])

        self.lam_crit = self.mu * self.masses[-1] * self.g
        self.N_odd = int(np.ceil(self.fourier.N_HBM / 2))

    @property
    def Lambda(self):
        """Reconstruct full Lambda vector from odd harmonic coefficients.

        The Lambda property reconstructs the complete set of Fourier coefficients
        (constant, cosine, and sine terms) from the stored odd harmonic components.
        Even harmonics are zero due to symmetry.

        Returns
        -------
        ndarray
            Full Lambda vector of length (2*N_HBM + 1) with even harmonics set to zero.
        """
        Lambda_cos_odd = self.Lambda_odd[: self.N_odd]
        Lambda_sin_odd = self.Lambda_odd[self.N_odd :]

        Lambda_const = np.array([0])

        Lambda_cos = np.vstack((Lambda_cos_odd, np.zeros_like(Lambda_cos_odd)))
        Lambda_cos = np.reshape(Lambda_cos, (-1), order="F")
        if self.fourier.N_HBM % 2 != 0:
            Lambda_cos = Lambda_cos[:-1]

        Lambda_sin = np.vstack((Lambda_sin_odd, np.zeros_like(Lambda_sin_odd)))
        Lambda_sin = np.reshape(Lambda_sin, (-1), order="F")
        if self.fourier.N_HBM % 2 != 0:
            Lambda_sin = Lambda_sin[:-1]

        return np.hstack((Lambda_const, Lambda_cos, Lambda_sin))

    @Lambda.setter
    def Lambda(self, value):
        """Set Lambda from full vector, extracting odd harmonics.

        Parameters
        ----------
        value : ndarray
            Full Lambda vector to be decomposed into odd/even harmonics.
        """
        self.Lambda_odd = self.extract_odd(value)

    def extract_odd(self, Lambda):
        """Extract odd harmonic coefficients from full Fourier coefficient vector.

        Decomposes the full FC vector into odd and even harmonic components.
        Even harmonics should theoretically be zero; any non-zero even harmonics
        are flagged but not enforced to be zero.

        Parameters
        ----------
        Lambda : ndarray
            Full Fourier coefficient vector of length (2*N_HBM + 1).

        Returns
        -------
        ndarray
            Odd harmonic coefficients concatenated as [cos_odd, sin_odd].
        """
        Lambda_const = Lambda[0]
        Lambda_cos = Lambda[1 : self.fourier.N_HBM + 1]
        Lambda_sin = Lambda[self.fourier.N_HBM + 1 :]
        Lambda_odd = np.hstack((Lambda_cos[::2], Lambda_sin[::2]))
        Lambda_even = np.hstack(((Lambda_const,), Lambda_cos[1::2], Lambda_sin[1::2]))
        if any(np.abs(Lambda_even) > 1e-14):
            # raise ValueError("Lambda_even must be zero.")
            print(f"ignored even values of magnitude {np.max(np.abs(Lambda_even))}.")
        return Lambda_odd

    def FC_X(self, Lambda=None):
        """Compute displacement Fourier coefficients.

        Solves the linear system Z @ X = W @ Lambda + F to obtain the displacement
        Fourier coefficients for a given friction force.

        Parameters
        ----------
        Lambda : ndarray, optional
            Lambda coefficients. If None, uses self.Lambda.

        Returns
        -------
        ndarray
            State (except Lambda) Fourier coefficients of shape (n_dof, 2*N_HBM + 1).
        """
        if Lambda is None:
            Lambda = self.Lambda

        X = np.linalg.solve(self.Z, self.W @ Lambda + self.F)
        X = np.reshape(X, (self.n_dof, -1), order="F")
        return X

    def FC_dX(self, Lambda=None):
        """Compute FCs of full state vector (except lambda).

        Computes both displacement and velocity (time derivative) Fourier coefficients
        by solving for X and then computing its Fourier-space derivative.

        Parameters
        ----------
        Lambda : ndarray, optional
            Lambda coefficients. If None, uses current self.Lambda.

        Returns
        -------
        tuple of ndarray
            (X, dX) where X are displacement coefficients and dX are velocity
            coefficients, each of shape (n_dof, 2*N_HBM + 1).
        """
        X = self.FC_X(Lambda)
        dX = np.zeros_like(X)
        for i in range(X.shape[0]):
            dX[i, :] = self.fourier.derivative_coeffs(X[i, :], omega=self.omega)
        return X, dX

    def x_time(self, Lambda=None):
        """Reconstruct full system state in time domain.

        Transforms displacement, velocity, and Lagrange multiplier from Fourier
        coefficient space to time domain using inverse DFT.

        Parameters
        ----------
        Lambda : ndarray, optional
            Lambda coefficients. If None, uses current self.Lambda.

        Returns
        -------
        ndarray
            Time-domain state matrix of shape (2*n_dof + 1, L_DFT) containing
            [x_1, ..., x_n, v_1, ..., v_n, lambda] as rows and time samples as columns.
        """
        if Lambda is None:
            Lambda = self.Lambda
        X, dX = self.FC_dX(Lambda)
        x_time = np.zeros((2 * X.shape[0] + 1, self.fourier.L_DFT), dtype=X.dtype)
        for k in range(X.shape[0]):
            x_time[k, :] = self.fourier.inv_DFT(X[k, :])
            x_time[k + X.shape[0], :] = self.fourier.inv_DFT(dX[k, :])
        x_time[-1, :] = self.fourier.inv_DFT(Lambda)
        return x_time

    def FC_gamma(self, Lambda=None):
        """Compute the acceleration of the friction contact DOF in Fourier space.

        Extracts the acceleration (second DOF velocity) which represents the
        constraint acceleration for the friction constraint.

        Parameters
        ----------
        Lambda : ndarray, optional
            Lambda coefficients. If None, uses current self.Lambda.

        Returns
        -------
        ndarray
            Fourier coefficients of the constraint acceleration, shape (2*N_HBM + 1,).
        """
        _, dX = self.FC_dX(Lambda)
        return dX[-1, :]

    def residual_with_argument(self, Lambda=None):
        """Compute residual of the friction constraint in odd harmonic space.

        Evaluates the friction constraint condition in time domain and transforms
        back to Fourier space, returning only the odd harmonic coefficients for the
        reduced problem.

        Parameters
        ----------
        Lambda : ndarray, optional
            Lambda coefficients. If None, uses current self.Lambda.

        Returns
        -------
        ndarray
            Odd harmonic coefficients of the residual, shape (N_odd + N_odd,).
        """
        if Lambda is None:
            Lambda = self.Lambda

        Gamma = self.FC_gamma(Lambda)
        lambdas = self.fourier.inv_DFT(Lambda)
        gammas = self.fourier.inv_DFT(Gamma)
        residual_time = self.constraint(gammas, lambdas)
        residual = self.fourier.DFT(residual_time)
        return self.extract_odd(residual)

    def residual_function(self):
        """Evaluate residual using current Lambda state.

        Wrapper method for compatibility with solver interfaces. Computes the
        residual of the friction constraint equation.

        Returns
        -------
        ndarray
            Odd harmonic residual coefficients.
        """
        return self.residual_with_argument()

    def constraint(self, gammas, lambdas):
        """Evaluate the appropriate friction constraint formulation.

        Dispatches to either the smooth (tanh-based) or non-smooth (proximal)
        constraint formulation based on the smoothing parameter.

        Parameters
        ----------
        gammas : ndarray
            Constraint acceleration in time domain, shape (L_DFT,).
        lambdas : ndarray
            Lagrange multiplier in time domain, shape (L_DFT,).

        Returns
        -------
        ndarray
            Constraint residual in time domain, shape (L_DFT,).
        """
        if self.smoothing == np.inf:
            return self.constraint_prox(gammas, lambdas)
        else:
            return self.constraint_smooth(gammas, lambdas)

    def constraint_smooth(self, gammas, lambdas):
        """Evaluate smoothed friction constraint using tanh regularization.

        The smooth constraint uses a hyperbolic tangent function to approximate
        the non-smooth complementarity condition: lambda + lam_crit * tanh(smoothing * gamma).

        Parameters
        ----------
        gammas : ndarray
            Constraint acceleration in time domain, shape (L_DFT,).
        lambdas : ndarray
            Lagrange multiplier in time domain, shape (L_DFT,).

        Returns
        -------
        ndarray
            Smoothed constraint residual, shape (L_DFT,).
        """
        return lambdas + self.lam_crit * np.tanh(self.smoothing * gammas)

    def constraint_prox(self, gammas, lambdas):
        """Evaluate non-smooth friction constraint using proximal formulation.

        The non-smooth constraint uses min/max operations to enforce the
        complementarity condition: gamma + min(0, prox*(lambda + lam_crit) - gamma)
                                      + max(0, prox*(lambda - lam_crit) - gamma).

        Parameters
        ----------
        gammas : ndarray
            Constraint acceleration in time domain, shape (L_DFT,).
        lambdas : ndarray
            Lagrange multiplier in time domain, shape (L_DFT,).

        Returns
        -------
        ndarray
            Non-smooth constraint residual, shape (L_DFT,).
        """
        return (
            gammas
            + np.minimum(
                0,
                self.prox_parameter * (lambdas + self.lam_crit) - gammas,
            )
            + np.maximum(
                0,
                self.prox_parameter * (lambdas - self.lam_crit) - gammas,
            )
        )

    def closed_form_derivative(self, variable):
        """Evaluate closed-form analytical derivative (inherited from parent).

        Parameters
        ----------
        variable : str
            Name of the variable to differentiate with respect to.

        Returns
        -------
        ndarray or None
            Analytical derivative if available, otherwise None.
        """
        return super().closed_form_derivative(variable)


def plot_solve_friction():
    """Solve friction oscillator and plot the time series.

    This demo function demonstrates the use of :func:`solve_friction`
    by building example oscillators (smoothed and non-smooth), solving the
    substituted problem and plotting the time-domain responses for the
    solution.

    Notes
    -----
    The function uses fixed example parameters and the :mod:`matplotlib`
    plotting backend to produce a 5-row subplot showing displacements,
    velocities and the Lagrange multiplier time series.
    """
    # # Legrand
    masses = [1, 1]
    stiffnesses = [1, 1]
    dampings = [0.02, 0.02]
    forcings = [20, 0]
    omega = 0.299
    phases = [0.5 * np.pi, 0]
    mu = 0.9
    smoothing = 10
    prox_parameter = 1
    normal_force = 10.5
    g = normal_force / masses[1]

    # warm-start from smoothed oscillator
    N_HBM = 40
    L_DFT = 4096
    fourier = Fourier(N_HBM=N_HBM, L_DFT=L_DFT, n_dof=5, real_formulation=True)

    dae_smooth = SmoothedFrictionOscillator(
        stiffnesses=stiffnesses,
        dampings=dampings,
        masses=masses,
        g=g,
        mu=mu,
        forcing_amplitudes=forcings,
        forcing_phases=phases,
        smoothing=smoothing,
    )

    dae_nonsmooth = FrictionOscillator(
        stiffnesses=stiffnesses,
        dampings=dampings,
        masses=masses,
        g=g,
        mu=mu,
        forcing_amplitudes=forcings,
        forcing_phases=phases,
        prox_parameter=prox_parameter,
    )

    for smoothing, dae in zip(
        (dae_smooth.smoothing, np.inf), [dae_smooth, dae_nonsmooth]
    ):

        equ = solve_friction(dae, fourier, omega, smoothing, initial_guess=None)

        x_time = equ.x_time()
        ts = equ.fourier.time_samples(equ.omega)
        _, axs = plt.subplots(5, 1)
        for k in range(5):
            axs[k].plot(ts, x_time[k, :])
            if k == 0:
                axs[k].set_title(f"Direct Friction oscillator smoothing = {smoothing}")
            axs[k].set_xlabel("time")
            axs[k].set_ylabel(f"x[{k}]")


if __name__ == "__main__":
    plot_solve_friction()
    plt.show()
