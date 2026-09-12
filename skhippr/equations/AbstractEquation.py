from abc import ABC, abstractmethod
import numpy as np


class AbstractEquation(ABC):
    """
    Abstract base class for algebraic equations in the SKHiPPR framework.

    Must provide a residual. The residual may depend on arbitrarily many attributes of the object and must return a numpy array. The :py:class:`~skhippr.solvers.newton.NewtonSolver` attempts to modify the object's attributes such that the residual becomes zero.

    Subclasses may provide closed-form expressions for the derivatives of the residual with respect to the attributes. Otherwise, finite differences are used.

    The class also offers functionality for stability analysis if a :py:class:`~skhippr.stability.AbstractStabilityMethod.AbstractStabilityMethod` object is passed.
    """

    def __init__(self, stability_method=None):

        super().__init__()
        self._derivative_dict = {}
        self.residual_value = None
        self.stability_method = stability_method
        self.stable = None
        self.eigenvalues = None

    # @property
    # def residual_value(self):
    #     return self._residual_value

    # @residual_value.setter
    # def residual_value(self, value):
    #     if (
    #         self.residual_value is not None
    #         and value.shape[0] != self._residual_value.shape[0]
    #     ):
    #         raise ValueError(
    #             f"Shape of residual value cannot be changed once set. Expected shape {self._residual_value.shape}, got {value.shape}."
    #         )
    #     self._residual_value = value

    def residual(self, update=False):
        if update:
            # compute the residual using the attributes
            self.residual_value = self.residual_function()
            if self.residual_value.ndim > 1:
                raise ValueError(
                    f"Residual must be a 1-D numpy array but has shape {self.residual_value.shape}"
                )
        elif self.residual_value is None:
            # raise RuntimeError("Residual has not been computed yet!")
            self.residual(update=True)
        return self.residual_value

    @abstractmethod
    def residual_function(self):  # -> ndarray:
        """Compute the residual function based on attributes of the object. This method must be implemented in subclasses."""
        residual: np.ndarray = ...
        return residual

    def visualize(self):
        pass

    def derivative(self, variable: str, update=False, h_fd=1e-4) -> np.ndarray:
        """
        Compute the derivative of the residual with respect to a given variable.
        This method should be called whenever a derivative is desired.

        If ``update`` is ``False``, a previously computed derivative is returned if available. Otherwise, the derivative is computed anew and is cached for future use.

        If closed-form derivatives are available, they are used. Otherwise, if :py:func:`~skhippr.equations.AbstractEquationSystem.AbstractEquationSystem.closed_form_derivative` raises a ``NotImplementedError``, :py:func:`~skhippr.equations.AbstractEquationSystem.AbstractEquationSystem.finite_difference_derivative` is used to compute the derivative.

        Parameters
        ----------

        variable : str
            The name of the variable (attribute) with respect to which the derivative is computed.
        update : bool, optional
            If ``True``, updates the cached derivative with the newly computed value. Default is ``False``.
        h_fd : float, optional
            Step size for finite difference approximation. Default is ``1e-4``.

        Returns
        -------

        np.ndarray
            The partial derivative of  the residual with respect to the specified variable.

        """

        # use cached derivative?
        if not update and variable in self._derivative_dict:
            return self._derivative_dict[variable]

        ########### DEBUGGING always use finite differences
        # if True:  # variable in ("X", "omega"):
        #     derivative = self.finite_difference_derivative(variable, h_step=h_fd)
        #     # Check sizes
        #     cols_expected = np.atleast_1d(getattr(self, variable)).shape[0]
        #     rows_expected = self.residual(update=False).shape[0]
        #     others_expected = self.residual(update=False).shape[1:]
        #     if derivative.shape != (rows_expected, cols_expected, *others_expected):
        #         raise ValueError(
        #             f"Size mismatch in derivative w.r.t. '{variable}': Expected {(rows_expected, cols_expected, *others_expected)}, got {derivative.shape[:2]}"
        #         )

        #     self._derivative_dict[variable] = derivative
        #     print(
        #         f"Caution overrode '{variable}' closed form derivative in AbstractSystems.py for debugging reasons"
        #       )
        #     warnings.warn(
        #         f"Caution overrode '{variable}' closed form derivative in AbstractSystems.py for debugging reasons"
        #       )

        #     return derivative
        ###########

        try:
            derivative = self.closed_form_derivative(variable)
        except NotImplementedError:
            # Fall back on finite differences.
            derivative = self.finite_difference_derivative(variable, h_step=h_fd)

        # Check sizes
        cols_expected = np.atleast_1d(getattr(self, variable)).shape[0]
        rows_expected = self.residual(update=False).shape[0]
        others_expected = self.residual(update=False).shape[1:]
        if derivative.shape != (rows_expected, cols_expected, *others_expected):
            raise ValueError(
                f"Size mismatch in derivative w.r.t. '{variable}': Expected {(rows_expected, cols_expected, *others_expected)}, got {derivative.shape[:2]}"
            )

        self._derivative_dict[variable] = derivative

        return derivative

    def closed_form_derivative(self, variable: str) -> np.ndarray:  # -> Any:
        """
        Compute the closed-form derivative of the residual with respect to a given variable.  To be implemented in subclasses. Must raise a ``NotImplementedError`` if closed-form derivative is nonzero and not available analytically.

        Returns
        -------

        np.ndarray
            The closed-form derivative of the residual with respect to the specified variable as a2-D numpy array. Must be 2-D even if the variable or the residual is a scalar.

        Raises
        ------
        NotImplementedError
            If the derivative is not available in closed form and should be determined using finite differences.
        """
        raise NotImplementedError(
            f"Closed-form derivative of residual w.r.t {variable} not implemented."
        )

    def finite_difference_derivative(self, variable, h_step=1e-4) -> np.ndarray:
        """Compute the finite difference derivative of the residual with respect to a given variable.

        Parameters
        ----------

        variable : str
            The name of the variable (attribute) with respect to which the derivative is computed.
        h_step : float, optional
            Step size for finite difference approximation. Default is ``1e-4``.

        Returns
        -------

        np.ndarray
            The finite difference derivative of the residual with respect to the specified variable as a 2-D numpy array.


        """

        x_orig = getattr(self, variable)
        x = np.atleast_1d(x_orig)
        if x.ndim == 1:
            x = x[:, np.newaxis]
        n = x.shape[0]

        try:
            f = self.residual(update=True)
            vectorized = True
        except Exception as e:
            if len(x.shape) > 2:
                raise RuntimeError(
                    f"Error computing vectorized finite difference derivative w.r.t. '{variable}' at index {k}. This may be due to a mismatch in the expected shape of the residual. Vectorized version required if dim(x) = {x.shape} > 2. Original error message: {e}"
                ) from e
            vectorized = False
            f = np.zeros_like(x)
            for i in range(x.shape[1]):
                setattr(self, variable, np.squeeze(x[:, i]))
                f[:, i] = self.residual(update=True)

        delta = h_step * np.eye(n)
        derivative = np.zeros((f.shape[0], n, *f.shape[1:]), dtype=f.dtype)

        for k in range(n):
            if vectorized:
                setattr(self, variable, np.squeeze(x + delta[:, [k]]))
                derivative[:, k, ...] = (self.residual_function() - f) / h_step
            else:
                for i in range(f.shape[1]):
                    setattr(self, variable, np.squeeze(x[:, i] + delta[:, k]))
                    derivative[:, k, i] = (self.residual_function() - f[:, i]) / h_step

        setattr(self, variable, x_orig)
        return derivative

    def determine_stability(self, update=False):
        """Determine the stability of the equation using the stability method.

        If the stability method is not set, an ``AttributeError`` is raised.
        If ``update`` is ``True``, the eigenvalues are computed using the stability method. Otherwise, the cached values is used.
        The stability is determined based on :py:func:`~skhippr.equations.AbstractEquationSystem.AbstractEquationSystem.stability_criterion`.

        Parameters
        ----------
        update : bool, optional
            If ``True``, updates the eigenvalues and stability status. Default is ``False``.
        """
        if self.stability_method is None:
            raise AttributeError("Stability method not available")

        if update:
            eigenvalues = self.stability_method.determine_eigenvalues(self)
            self.eigenvalues = eigenvalues
            stable = self.stability_criterion(eigenvalues)
            self.stable = stable

        return self.stable, self.eigenvalues

    def stability_criterion(self, eigenvalues):
        """Determine stability based on stability-defining eigenvalues computed by the stability method."""
        if self.stability_method is None:
            raise ValueError("No stability method available!")
        else:
            raise NotImplementedError(
                "To be implemented in concrete subclasses if needed"
            )
