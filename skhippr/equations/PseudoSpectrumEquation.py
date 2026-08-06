"""This module provides methods to determine the epsilon-pseudospectrum of a matrix.

If matrix B is a perturbation of matrix A, i.e., ||A-B|| <= epsilon,
then the eigenvalues of B must lie within the epsilon-pseudospectrum of A. (Trefethen, 1997).
The epsilon-pseudospectrum of A is defined as

    Lambda_epsilon(A) = {z: sigma_min(z*eye - A) <= epsilon}

i.e. all complex numbers for which the minimum singular value of (z*eye - A)
is smaller than epsilon.

Reference:
----------
Trefethen, L. N. (1997), 'Pseudospectra of Linear Operators', SIAM Review, 383--406
doi: 10.1137/s0036144595295284
"""

import numpy as np
import matplotlib.pyplot as plt

from skhippr.equations.AbstractEquation import AbstractEquation
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.solvers.newton import NewtonSolver
from skhippr.solvers.continuation import pseudo_arclength_continuator


class PseudoSpectrumEquation(AbstractEquation):
    def __init__(self, A, epsilon, z=None):
        super().__init__(stability_method=None)
        self.A = A
        self.epsilon = epsilon
        self.eye = np.eye(A.shape[0])

        if z is None:
            eigenvalues, _ = np.linalg.eig(A)
            z = eigenvalues[0]

        self.z = z

    @property
    def z(self):
        return self.re_z + 1j * self.im_z

    @z.setter
    def z(self, value):
        self.re_z = np.real(value)
        self.im_z = np.imag(value)

    def residual_function(self):
        R = self.A - self.z * self.eye
        _, s, _ = np.linalg.svd(R)
        sigma_min = np.min(s)
        return np.atleast_1d(sigma_min - self.epsilon)

    def closed_form_derivative(self, variable):
        if variable == "epsilon":
            return np.atleast_2d(-1)
        if variable == "z":
            raise ValueError(
                "Derivative of real-valued residual w.r.t complex variable 'z' problematic. "
                "Consider 're_z' (real part) and 'im_z' (imaginary part) as two independent, real-valued unknowns. "
            )

        return super().closed_form_derivative(variable)


def compute_pseudospectrum(
    A: np.array = None,
    epsilon: float = 0.3,
    z_init: complex = None,
    tolerance: float = 1e-8,
    verbose: bool = True,
    max_step=0.1,
) -> np.ndarray:
    """
    Computes a boundary of the pseudospectrum of a given matrix A around an
    initial value z_0 using pseudo-arclength continuation.

    Eigenvalues of A + dA, where ||dA|| < ``epsilon``, lie within the pseudospectrum.

    Parameters
    ----------

    A : array-like or None
        The input matrix for which the pseudospectrum is to be computed.
    epsilon : float, optional
        The perturbation parameter defining the pseudospectrum boundary (default is 0.3).
    z_init : complex or None, optional
        Initial guess for the complex variable z (default is None, then ``A``'s first eigenvalue is used).
    tolerance : float, optional
        Tolerance for the Newton solver (default is 1e-8).
    verbose : bool, optional
        If True, enables verbose output during computation (default is True).

    Returns
    -------

    pseudo_spectrum : np.ndarray
        List of complex numbers representing the computed pseudospectrum boundary.
    """
    eq = PseudoSpectrumEquation(A=A, epsilon=epsilon, z=z_init)
    sys = EquationSystem(equations=[eq], unknowns=["re_z", "im_z"])
    solver = NewtonSolver(verbose=False, tolerance=tolerance)

    pseudo_spectrum = []

    for bp in pseudo_arclength_continuator(
        initial_system=sys,
        solver=solver,
        verbose=verbose,
        num_steps=50000,
        stepsize_range=(0.01 * max_step, max_step),
    ):
        pseudo_spectrum.append(bp.re_z + +1j * bp.im_z)
        if len(pseudo_spectrum) > 1 and bp.im_z > np.imag(pseudo_spectrum[0]) > np.imag(
            pseudo_spectrum[-2]
        ):
            break

    return np.array(pseudo_spectrum)


def does_pseudospectrum_include(A: np.array, epsilon: float, z_vals: np.array) -> bool:
    """Determine whether the pseudospectrum of A includes any of the points in z_vals."""
    equ = PseudoSpectrumEquation(A, epsilon, z_vals[0])
    for z in z_vals:
        equ.z = z
        if equ.residual(update=True) < 0:
            return True

    return False


def plot_pseudospectrum(A, epsilon=0.3, ax=None):
    """
    Plots the pseudospectrum of a given matrix A.

    Parameters
    ----------

    A : array-like or np.ndarray
        The input square matrix whose pseudospectrum is to be plotted.
    epsilon : float, optional
        The perturbation parameter for the pseudospectrum calculation (default is 0.3).
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot. If None, a new figure and axes are created.

    Notes
    -----

    The function plots the eigenvalues of A as red 'x' markers and the pseudospectrum points
    around each eigenvalue as dots. The aspect ratio of the plot is set to 'equal'.
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    eigs, _ = np.linalg.eig(A)
    ax.plot(np.real(eigs), np.imag(eigs), "rx")

    for z in eigs:
        p_spectrum = compute_pseudospectrum(A, epsilon, z)
        ax.plot(np.real(p_spectrum), np.imag(p_spectrum), ".")

    ax.set_aspect("equal")
    ax.set_ylabel("Im")
    ax.set_xlabel("Re")


def examples_pseudospectrum():
    # normal matrix: pseudospectrum is a circle
    eigs = [1 + 1j, 2]
    A = np.diag(eigs)
    plot_pseudospectrum(A)

    # non-normal matrix
    A = np.array([[1, 50], [0, 0.4]])
    plot_pseudospectrum(A, epsilon=0.002)


if __name__ == "__main__":

    examples_pseudospectrum()
    plt.show()
