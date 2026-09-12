import warnings
import numpy as np
from scipy import fft
from scipy.linalg import block_diag


class Fourier:
    """
    Methods for (Fast) Fourier Transform (FFT) and related operations in real or complex formulation.

    Attributes:

    * :py:attr:`~skhippr.Fourier.Fourier.n_dof`: Number of states (degrees of freedom) of the considered signals.
    * :py:attr:`~skhippr.Fourier.Fourier.N_HBM`: Largest harmonic to be considered in the Fourier series.
    * :py:attr:`~skhippr.Fourier.Fourier.L_DFT`: Number of samples for the fast Fourier transform (FFT). Must be ``>= 2 * (N_HBM + 1)`` to avoid aliasing.
    * :py:attr:`~skhippr.Fourier.Fourier.real_formulation`: Whether Fourier coefficients are returned in real or complex formulation.

    Useful methods:

    * :py:func:`~skhippr.Fourier.Fourier.time_samples`
    * :py:func:`~skhippr.Fourier.Fourier.DFT`
    * :py:func:`~skhippr.Fourier.Fourier.inv_DFT`
    * :py:func:`~skhippr.Fourier.Fourier.matrix_DFT`
    * :py:func:`~skhippr.Fourier.Fourier.matrix_inv_DFT`
    * :py:func:`~skhippr.Fourier.Fourier.derivative_coeffs`
    * :py:func:`~skhippr.Fourier.Fourier.differentiate`
    * :py:func:`~skhippr.Fourier.Fourier.resize_coefficients`
    * :py:func:`~skhippr.Fourier.Fourier.__replace__`

    Notes
    -----

    This class supports both real-valued and complex-valued formulations.

    * For the complex-valued formulation, Fourier coefficients are represented in increasing frequency order (from ``-N_HBM`` to ``N_HBM``).
    * For the real-valued formulation, the 0-th Fourier coefficient is stored first, then all cosine coefficients in increasing frequency order, then all sine coefficients in increasing frequency order.

    """

    def __init__(
        self,
        N_HBM: int,
        L_DFT: int,
        n_dof: int,
        real_formulation: bool = True,
    ):
        """
        Initialize the class.

        Parameters
        ----------
        N_HBM : int
            Number of harmonics.
        L_DFT : int
            Number of points for the fast Fourier transform.
            If less than 2 * (N_HBM + 1), it will be increased to the next efficient length for FFT.
        n_dof : int
            Number of degrees of freedom of the system to be considered.
        real_formulation : bool, optional
            Whether to use the real-valued formulation (default is ``True``).

        Warns
        -----
        UserWarning
            If ``L_DFT`` is too small to resolve the given ``N_HBM`` without aliasing, a warning is issued and ``L_DFT`` is set to the next fast length.

        """
        self.n_dof = n_dof
        self.N_HBM = N_HBM
        if L_DFT < 2 * (N_HBM + 1):
            warnings.warn(
                f"L_DFT={L_DFT} too small for N_HBM={N_HBM}. Chose L_DFT={fft.next_fast_len(2 * (N_HBM + 1))} instead."
            )
            L_DFT = fft.next_fast_len(2 * (N_HBM + 1))
        self.L_DFT = L_DFT
        self.real_formulation = real_formulation

        self.T_to_cplx_from_real: np.ndarray
        self.T_to_real_from_cplx: np.ndarray
        self._init_trafo_matrices()

        self.DFT_matrix: np.ndarray
        self.iDFT_matrix: np.ndarray
        self.iDFT_matrix: np.ndarray
        self._init_DFT()

        self.derivative_matrix: np.ndarray
        self._init_derivative_matrix()

    def _init_trafo_matrices(self) -> None:
        """
        Initialize transformation matrices between complex-valued and real-valued Fourier coefficient representations.

        This method constructs two transformation matrices, which are stored as attributes:

        * ``T_to_cplx_from_real``: Converts real-valued Fourier coefficients to complex-valued coefficients.
        * ``T_to_real_from_cplx``: Converts complex-valued Fourier coefficients to real-valued coefficients.

        The transformation is based on the relationships between the real and imaginary parts of the Fourier basis:
        - Complex-valued coefficients are ordered naturally in ascending order (first the Fourier coefficients of all states with harmonic ``-N_HBM``, etc.).
        - Real-valued coefficients are ordered with all cosine terms in increasing order, followed by all sine terms.

        Notes
        -----

        The transformation uses the relationships:
            * a*cos(x) + b*sin(x) = 0.5*(a - b*i)*exp(ix) + 0.5*(a + b*i)*exp(-ix)
            * u*exp(ix) + v*exp(-ix) = (u+v)*cos(x) + i*(u-v)*sin(x)

        """

        "T_cr: Trafo from real to complex"
        T_cr = np.zeros((2 * self.N_HBM + 1, 2 * self.N_HBM + 1), dtype=complex)
        # constant term of cplx vector is in centermost row
        T_cr[self.N_HBM, 0] = 1

        for k in range(1, self.N_HBM + 1):
            # k positive: X_k = 0.5*(a_k - b_k*i)
            T_cr[self.N_HBM + k, k] = 0.5
            T_cr[self.N_HBM + k, self.N_HBM + k] = -0.5j

            # -k negative: X_{-k} = 0.5*(a_k + b_k*i)
            T_cr[self.N_HBM - k, k] = 0.5
            T_cr[self.N_HBM - k, self.N_HBM + k] = 0.5j

        "T_rc: Trafo from complex to real"
        T_rc = np.zeros((2 * self.N_HBM + 1, 2 * self.N_HBM + 1), dtype=complex)
        # constant term of real vector is in first row
        T_rc[0, self.N_HBM] = 1

        for k in range(1, self.N_HBM + 1):
            # cosine terms: a = u + v
            T_rc[k, self.N_HBM + k] = 1
            T_rc[k, self.N_HBM - k] = 1

            # sine terms: b = i*(u-v)
            T_rc[self.N_HBM + k, self.N_HBM + k] = 1j
            T_rc[self.N_HBM + k, self.N_HBM - k] = -1j

        eye = np.eye(self.n_dof)
        self.T_to_cplx_from_real = np.kron(T_cr, eye)
        self.T_to_real_from_cplx = np.kron(T_rc, eye)

    def _init_DFT(self) -> None:
        """
        Initializes the Discrete Fourier Transform (DFT) and inverse DFT (iDFT) matrices for different formulations.

        The method also prepares index arrays for extracting Fourier coefficients from FFT outputs in the complex formulation.

        Attributes set
        --------------
        time_samples_normalized : ndarray
            Array of normalized time samples in [0, 2*pi).
        DFT_matrix : ndarray
            The full DFT matrix for all degrees of freedom.
        iDFT_small : ndarray
            The small (1D) inverse DFT matrix.
        iDFT_matrix : ndarray
            The full inverse DFT matrix for all degrees of freedom.
        idx_FFT : ndarray, optional
            Index array for extracting ordered Fourier coefficients from FFT output (only in complex formulation).

        Notes
        -----

        The real-valued case uses the DFT relationships:

        * x_m = X_0 + sum_{k = 1}^N X_k_cos * cos(k*tau_m) + X_k_sin*sin(k*tau_m)
        * X_0     = 1/L * sum_{m=0}^{L-1} x(t_m)
        * X_k_cos = 2/L * sum_{m=0}^{L-1} x(t_m) * cos(k*tau_m)
        * X_k_sin = 2/L * sum_{m=0}^{L-1} x(t_m) * sin(k*tau_m)

        The complex-valued case uses the relationships:

        * x_m = sum_{k = -N}^N X_k * exp(1j*k*tau_m)
        * X_k = 1/L * sum_{m=0}^{L-1} exp(-1j*k*tau_m)*x(t_m)
        """

        self.time_samples_normalized = np.linspace(
            0, 2 * np.pi, num=self.L_DFT, endpoint=False
        )

        if self.real_formulation:
            # iDFT
            frequencies_normalized = np.arange(1, self.N_HBM + 1)
            samps = np.outer(self.time_samples_normalized, frequencies_normalized)
            iDFT_small = np.hstack(
                (np.ones((self.L_DFT, 1)), np.cos(samps), np.sin(samps))
            )

            # DFT
            DFT_small = iDFT_small.T * (2 / self.L_DFT)
            DFT_small[0, :] = 1 / self.L_DFT

        else:
            # iDFT
            frequencies_normalized = np.arange(-self.N_HBM, self.N_HBM + 1)
            samps = np.outer(self.time_samples_normalized, 1j * frequencies_normalized)
            iDFT_small = np.exp(samps)

            # DFT
            DFT_small = (1 / self.L_DFT) * (np.flip(iDFT_small, axis=1).T)

            # Index to extract [X_{-N}, ..., X_{N}] from the output of fft.fft
            # which is [X_{0}, X_{1}, ..., X_{L_DFT/2}, X_{-(L_DFT-2)/2, ..., X_{-1}}]"""
            self.idx_FFT = np.r_[
                self.L_DFT - self.N_HBM : self.L_DFT, 0 : self.N_HBM + 1
            ]

        self.iDFT_small = iDFT_small

        eye = np.eye(self.n_dof)
        self.DFT_matrix = np.kron(DFT_small, eye)
        self.iDFT_matrix = np.kron(iDFT_small, eye)

    def time_samples(self, omega: float, periods: float = 1) -> np.ndarray:
        """
        Generate a uniformly spaced vector of time samples starting at ``0`` with ``L_DFT`` samples per period.

        * If ``periods`` is integer, the last sample is ``T*periods-dt``.
        * If ``periods`` is non-integer, it is rounded down to obtain an integer number of samples that are equally spaced at the DFT sampling frequency. The last sample lies in ``[T*periods-dt, T*periods)``.

        Parameters
        ----------

        omega : float
            The angular frequency (in radians per unit time).
        periods : float, optional
            Number of periods to sample. Defaults to 1. Can be non-integer.

        Returns
        -------

        np.ndarray
            1-D array of time samples.

        """
        if periods == 1:
            return self.time_samples_normalized / omega
        else:
            # ceil ensures that the last sample is < T*periods, but >= T*periods-dt
            num = np.ceil(periods * self.L_DFT)
            T = 2 * np.pi / omega
            t_end = (num / self.L_DFT) * T
            return np.linspace(0, t_end, num=int(num), endpoint=False)

    def _DFT(self, x_samp: np.ndarray) -> np.ndarray:
        """Compute the Discrete Fourier Transform (DFT) of the input signal by matrix multiplication.

        .. deprecated:: 0.1

        This method is deprecated and has been replaced by :py:func:`~skhippr.Fourier.Fourier.DFT` which uses FFT for increased efficiency.

        Parameters
        ----------
        x_samp : np.ndarray
            Input signal. Can be a tall vector of size nL or a n x L array.

        Returns
        -------
        np.ndarray
            The Fourier coefficients of the input signal.

        """
        return self.DFT_matrix @ x_samp.flatten(order="F")

    def DFT(self, x_samp: np.ndarray) -> np.ndarray:
        """
        Compute the Fourier coefficient vector of the input signal using ``scipy.fft.fft``.

        Parameters
        ----------
        x_samp : np.ndarray
            Input signal array of shape (:py:attr:`~skhippr.Fourier.Fourier.n_dof`, ..., :py:attr:`~skhippr.Fourier.Fourier.L_DFT`)

        Returns
        -------
        np.ndarray
            Array containing the stacked Fourier coefficients with shape (:py:attr:`~skhippr.Fourier.Fourier.n_dof` * (2* :py:attr:`~skhippr.Fourier.Fourier.N_HBM` + 1 , ...).

        Notes
        -----
        If :py:attr:`~skhippr.Fourier.Fourier.real_formulation` is ``True``, the real-valued Fourier coefficients are returned; otherwise, the complex-valued Fourier coefficients.

        """

        shape = x_samp.shape
        if shape[0] != self.n_dof or shape[-1] != self.L_DFT:
            raise ValueError(
                f"x_samp shape {shape} is not {self.n_dof} x ... x {self.L_DFT}"
            )

        if self.real_formulation:
            X = self._DFT_real(x_samp)
        else:
            X = self._DFT_cplx(x_samp)

        # get coeffs in the 0th dimension
        X = X.transpose(len(shape) - 1, *range(len(shape) - 1))
        # reshape the 0th and 1th dimension C-style (last to first dimension) while keeping all others the same
        return X.reshape(self.n_dof * (2 * self.N_HBM + 1), *shape[1:-1], order="C")

    def _DFT_real(self, x_samp):
        """Real-valued (Fast) Discrete Fourier transform. x_samp must be a ... x L_DFT array.
        Return value is  ... x (2*N_HBM+1) array of Fourier coefficients of the same shape.
        """
        X_cplx = fft.rfft(x_samp, axis=-1, norm="forward", overwrite_x=False)[
            ..., : self.N_HBM + 1
        ]
        return np.concatenate(
            (
                np.real(X_cplx[..., [0]]),  # 0 is in brackets to keep the dimension
                2 * np.real(X_cplx[..., 1:]),
                -2 * np.imag(X_cplx[..., 1:]),
            ),
            axis=-1,
        )

    def _DFT_cplx(self, x_samp):
        return fft.fft(
            x_samp, n=self.L_DFT, axis=-1, norm="forward", overwrite_x=False
        )[..., self.idx_FFT]

    def _inv_DFT(self, X, imag_tol_abs: float = 1e-6, imag_tol_rel: float = 1e-3):
        """Compute the inverse Discrete Fourier Transform (iDFT) of the input signal by matrix multiplication.

        .. deprecated:: 0.1

        This method is deprecated and has been replaced by :py:func:`~skhippr.Fourier.Fourier.inv_DFT` which uses FFT for increased efficiency.

        Parameters
        ----------
        X : np.ndarray
            Fourier coefficients.

        Returns
        -------
        np.ndarray
            The samples of the signal

        """
        x_samp = self.iDFT_matrix @ X[: (2 * self.N_HBM + 1) * self.n_dof]

        if not self.real_formulation:
            # Check whether imaginary parts are small
            imag_max = max(np.abs(np.imag(x_samp)))
            real_max = max(np.abs(np.real(x_samp)))
            if imag_max < imag_tol_abs or imag_max < imag_tol_rel * real_max:
                x_samp = np.real(x_samp)

        return x_samp.reshape(self.n_dof, self.L_DFT, order="F")

    def inv_DFT(self, X: np.ndarray) -> np.ndarray:
        """Compute the inverse Discrete Fourier Transform (DFT) of the given Fourier coefficient vector.

        This method reconstructs the time-domain samples from the provided Fourier coefficients.

        Parameters
        ----------
        X : np.ndarray
            Fourier coefficient vector (as generated by :py:func:`~skhippr.Fourier.Fourier.DFT` ). First dimension must have
            :py:attr:`~skhippr.Fourier.Fourier.n_dof` * (2 * :py:attr:`~skhippr.Fourier.Fourier.N_HBM` + 1) entries. May have more dimensions.

        Returns
        -------
        np.ndarray
            Time-domain samples reconstructed from the Fourier coefficients, returned as an
            array of shape [:py:attr:`~skhippr.Fourier.Fourier.n_dof`, ..., :py:attr:`~skhippr.Fourier.Fourier.L_DFT`].

        """

        # Reshape and bring coefficient dimension to the back
        X = X[: self.n_dof * (2 * self.N_HBM + 1), ...].reshape(
            2 * self.N_HBM + 1, self.n_dof, *X.shape[1:], order="C"
        )
        X = X.transpose(*range(1, len(X.shape)), 0)

        if self.real_formulation:
            return self._inv_DFT_real(X)
        else:
            return self._inv_DFT_cplx(X)

    def _inv_DFT_real(self, X):
        """Inverse FFT.
        X is a ... x (2*N_HBM*1) array in sin/cos formulation.
        Returns a... x L_DFT array."""
        X_pos = np.concatenate(
            (
                X[..., [0]],  # [0] necessary for correct number of dimensions
                0.5 * X[..., 1 : self.N_HBM + 1] - 0.5j * X[..., self.N_HBM + 1 :],
            ),
            axis=-1,
        )
        return fft.irfft(X_pos, self.L_DFT, axis=-1, norm="forward", overwrite_x=False)

    def _inv_DFT_cplx(self, X):
        """Inverse FFT.
        X is a ... x (2*N_HBM*1) array in ascending exponential formulation.
        Returns a... x L_DFT array."""
        # Reorder and pad with zeros
        shape_padding = list(X.shape)
        shape_padding[-1] = self.L_DFT - shape_padding[-1]
        X = np.concatenate(
            (X[..., self.N_HBM :], np.zeros(shape_padding), X[..., : self.N_HBM]),
            axis=-1,
        )
        x_samp = fft.ifft(X, axis=-1, norm="forward", overwrite_x=False)
        if np.max(np.abs(x_samp.imag)) < 1e-14:
            x_samp = x_samp.real
        return x_samp

    # deprecated, only remains for comparison/testing purposes
    def _matrix_DFT(self, A: np.ndarray) -> np.ndarray:
        """
        Determine self.DFT_matrix*A_block*self.iDFT_matrix.
        A is a dense self.n_dof x self.n_dof x self.L_DFT array collecting samples A(t_m).
        A_block is a block diagonal array collecting the blocks A(t_m).
        DEPRECATED
        """

        A_block = block_diag(*[A[:, :, kk] for kk in range(self.L_DFT)])
        return (self.DFT_matrix @ A_block) @ self.iDFT_matrix

    def matrix_DFT(self, A: np.ndarray) -> np.ndarray:
        """Compute the `matrix discrete Fourier transform` of samples of a square matrix, as required for the HBM Jacobian,  using efficient scaling and FFT.

        The result of this function is equivalent to :py:attr:`~skhippr.Fourier.Fourier.DFT_matrix` @ ``A_block`` @ :py:attr:`~skhippr.Fourier.Fourier.iDFT_matrix`, where ``A_block`` is a block-diagonal matrix formed from the samples of ``A``, but the computation is carried out efficiently using ``scipy.fft.fft``.

        Parameters
        ----------

        A : np.ndarray
            Input array of shape (:py:attr:`~skhippr.Fourier.Fourier.n_dof`, :py:attr:`~skhippr.Fourier.Fourier.n_dof`, :py:attr:`~skhippr.Fourier.Fourier.L_DFT`), collecting samples of the time-periodic matrix.

        Returns
        -------

        np.ndarray
            The result of the matrix DFT operation, a square array of length :py:attr:`~skhippr.Fourier.Fourier.n_dof` * (2 * :py:attr:`~skhippr.Fourier.Fourier.N_HBM` + 1).

        Notes
        -----

        * The ``k``-th column of the result is computed as the FFT of the ``k``-th column of ``A_block @`` :py:attr:`~skhippr.Fourier.Fourier.iDFT_matrix`.
        * Due to the block-diagonal structure, this is achieved by scaling ``A[:, :, l]`` by :py:attr:`~skhippr.Fourier.Fourier.iDFT_small` ``[k, l]`` for each ``l``.
        * The resulting matrix has block-Toeplitz structure.
        """

        column_blocks = []
        for k in range(2 * self.N_HBM + 1):
            A_scaled = A * self.iDFT_small[np.newaxis, np.newaxis, :, k]
            next_col = self.DFT(A_scaled)

            column_blocks.append(next_col)

        else:
            return np.hstack(column_blocks)

    def matrix_inv_DFT(self, A: np.ndarray) -> np.ndarray:
        """Pseudo-inverse operation of :py:func:`~skhippr.Fourier.Fourier.matrix_DFT` (up to aliasing).

        Parameters
        ----------

        A : np.ndarray
            The result of a matrix DFT operation, a square array of length :py:attr:`~skhippr.Fourier.Fourier.n_dof` * (2 * :py:attr:`~skhippr.Fourier.Fourier.N_HBM` + 1) with block-Toeplitz structure.

        Returns
        -------

        np.ndarray
            array of shape (:py:attr:`~skhippr.Fourier.Fourier.n_dof`, :py:attr:`~skhippr.Fourier.Fourier.n_dof`, :py:attr:`~skhippr.Fourier.Fourier.L_DFT`), collecting samples of the corresponding time-periodic matrix.

        """
        # index where self.iDFT_small has a row of ones
        if self.real_formulation:
            idx_1 = 0
        else:
            idx_1 = self.N_HBM

        return self.inv_DFT(A[:, self.n_dof * idx_1 : self.n_dof * (idx_1 + 1)])

    def _init_derivative_matrix(self) -> None:
        """
        Initialize derivative matrix self.derivative_matrix,
        such that x_dot_samp = self.inv_DFT(self.omega*self.derivative_matrix @ self.DFT(X))
        But compute it more efficiently element-wise.
        """
        if self.real_formulation:
            self.factors_derivative = np.kron(
                np.hstack(
                    (0, np.arange(1, self.N_HBM + 1), -np.arange(1, self.N_HBM + 1))
                ),
                np.ones(self.n_dof),
            )
            idx_blocks = self.n_dof * np.array([1, self.N_HBM + 1, 2 * self.N_HBM + 1])
            self.idx_derivative = np.hstack(
                (
                    np.arange(0, idx_blocks[0]),
                    np.arange(idx_blocks[1], idx_blocks[2]),
                    np.arange(idx_blocks[0], idx_blocks[1]),
                )
            )
        else:
            self.factors_derivative = 1j * np.kron(
                np.arange(-self.N_HBM, self.N_HBM + 1), np.ones(self.n_dof)
            )
            self.idx_derivative = np.arange(self.n_dof * (2 * self.N_HBM + 1))

        if self.real_formulation:
            Omega = np.diag(np.arange(1, self.N_HBM + 1))
            D_small = block_diag([0], np.kron(np.array([[0, 1], [-1, 0]]), Omega))
        else:
            D_small = np.diag(1j * np.arange(-self.N_HBM, self.N_HBM + 1))

        self.derivative_matrix = np.kron(D_small, np.eye(self.n_dof))

    def derivative_coeffs(self, X: np.ndarray, omega: float = 1) -> np.ndarray:
        """
        Compute the Fourier coefficients of the derivative of a signal which is represented by Fourier coefficients.

        Parameters
        ----------
        X : np.ndarray
            A 1-D array of shape :py:attr:`~skhippr.Fourier.Fourier.n_dof` * (2 * :py:attr:`~skhippr.Fourier.Fourier.N_HBM` + 1), containing the Fourier coefficients of the original signal.
        omega : float, optional
            The angular frequency to scale the derivative coefficients. Default is 1.

        Returns
        -------
        np.ndarray
            A 1-D array of the same shape as ``X``, containing the Fourier coefficients of the derivative
            of the signal.
        """
        return self.factors_derivative * X[self.idx_derivative] * omega

    def differentiate(self, x_samp: np.ndarray, omega: float) -> np.ndarray:
        """
        Compute the the derivative of a periodic signal in the frequency domain.

        Parameters
        ----------
        x_samp : np.ndarray
            A 2-D array of shape (:py:attr:`~skhippr.Fourier.Fourier.n_dof`,  :py:attr:`~skhippr.Fourier.Fourier.L_DFT`), containing the samples of the original signal.
        omega : float, optional
            The angular frequency of the signal. Default is 1.

        Returns
        -------
        np.ndarray
            A 1-D array of the same shape as ``x_samp``, containing the dfferentiated signal.
        """
        return self.inv_DFT(self.derivative_coeffs(self.DFT(x_samp), omega))

    def __str__(self):
        if self.real_formulation:
            label_formulation = "real"
        else:
            label_formulation = "complex"

        return f"DFT helper N_HBM = {self.N_HBM}, L_DFT = {self.L_DFT}, {label_formulation}"

    def __replace__(self, /, **changes):
        """
        Return a new instance of the Fourier class with updated attributes.

        Parameters
        ----------
        **changes : dict
            Keyword arguments (arguments of :py:class:`~skhippr.Fourier.Fourier`)
            specifying attribute values to replace in the new instance.
            If an attribute is not provided in ``changes``, its value from the current instance is used.

        Returns
        -------
        Fourier
            A new instance of the Fourier class with the specified changes applied.

        """

        if "N_HBM" not in changes:
            changes["N_HBM"] = self.N_HBM
        if "L_DFT" not in changes:
            changes["L_DFT"] = self.L_DFT
        if "n_dof" not in changes:
            changes["n_dof"] = self.n_dof
        if "real_formulation" not in changes:
            changes["real_formulation"] = self.real_formulation

        return Fourier(**changes)

    def resize_coefficients(self, X: np.ndarray) -> np.ndarray:
        """Resize Fourier coefficients to match the current harmonic truncation.

        The input coefficients may correspond to a different harmonic truncation
        ``N_other`` than :attr:`N_HBM`. If ``N_other`` is larger than the current
        truncation, the highest-frequency coefficients are discarded. If
        ``N_other`` is smaller, the missing coefficients are padded with zeros.

        The method accepts either a flattened coefficient vector of length
        ``n_dof * (2 * N_other + 1)`` or a 2-D array with ``n_dof`` rows and
        ``2 * N_other + 1`` columns and correspondingly returns either a flattened or a 2-D result.
        The coefficient ordering is preserved in the
        current formulation:

        * real formulation: ``[const, cos_1, ..., cos_N, sin_1, ..., sin_N]``
        * complex formulation: ``[X_{-N}, ..., X_0, ..., X_N]``

        Parameters
        ----------
        X : np.ndarray
            Fourier coefficients to be resized. The first dimension must match
            :attr:`n_dof` if ``X`` is 2-D, or the total length must be divisible
            by :attr:`n_dof` if ``X`` is 1-D.

        Returns
        -------
        np.ndarray
            Fourier coefficients resized to the current :attr:`N_HBM`. The
            returned array has the same dimensionality as the input ``X``.

        Raises
        ------
        ValueError
            If ``X`` does not have a compatible shape or does not contain an
            odd number of coefficients per degree of freedom.
        """

        X = np.asarray(X)
        input_is_vector = X.ndim == 1

        if input_is_vector:
            if X.size % self.n_dof != 0:
                raise ValueError(
                    f"Input length {X.size} is not divisible by n_dof={self.n_dof}."
                )
            X = np.reshape(X, (self.n_dof, -1), order="F")
        elif X.ndim == 2:
            if X.shape[0] != self.n_dof:
                raise ValueError(
                    f"Expected X to have {self.n_dof} rows, got shape {X.shape}."
                )
        else:
            raise ValueError("X must be a 1-D or 2-D numpy array.")

        n_coeffs = X.shape[1]
        if n_coeffs % 2 != 1:
            raise ValueError(
                f"Expected an odd number of coefficients (2*N_HBM_old + 1) per DOF, got {n_coeffs}."
            )

        N_other = (n_coeffs - 1) // 2
        X_resized = np.zeros((self.n_dof, 2 * self.N_HBM + 1), dtype=X.dtype)

        n_keep = min(self.N_HBM, N_other)

        if self.real_formulation:
            # Constant term.
            X_resized[:, 0] = X[:, 0]

            # Cosine coefficients.
            X_resized[:, 1 : n_keep + 1] = X[:, 1 : n_keep + 1]

            # Sine coefficients.
            X_resized[:, self.N_HBM + 1 : self.N_HBM + n_keep + 1] = X[
                :, N_other + 1 : N_other + n_keep + 1
            ]
        else:
            # Complex formulation stores frequencies symmetrically around zero.
            start_old = N_other - n_keep
            end_old = N_other + n_keep + 1
            start_new = self.N_HBM - n_keep
            end_new = self.N_HBM + n_keep + 1
            X_resized[:, start_new:end_new] = X[:, start_old:end_old]

        if input_is_vector:
            return X_resized.reshape(-1, order="F")
        return X_resized


def round_to_significant_digits(value, significant_digits):
    # reference: https://gist.github.com/ttamg/3f65227fd580b3d8dc8ba91e01507280
    if abs(value) == 0:
        return value

    round_digits = -int(np.floor(np.log10(np.abs(value)))) + significant_digits - 1
    return np.round(value, round_digits)
