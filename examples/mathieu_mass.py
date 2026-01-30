import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

from skhippr.odes.ltp import HillODE, HillWithMass, HillWithMassInverted
from skhippr.cycles.hbm import HBMEquation, HBMEquationDAE
from skhippr.Fourier import Fourier
from skhippr.solvers.newton import NewtonSolver

from skhippr.stability.KoopmanHillProjection import (
    KoopmanHillProjection,
    KoopmanHillDAE,
)

from skhippr.visualization.cycles import (
    plot_phase,
    plot_period,
    plot_floquet_multipliers,
)

import warnings
from numpy.exceptions import ComplexWarning  # Corrected import

# Raise an error on ComplexWarning
warnings.filterwarnings("error", category=ComplexWarning)


def main():
    # ---- Parameters ----
    params = dict(omega=1.0, damping=0.05, forcing=1)
    g_fun = lambda t: 1 / (1 + 0.9 * np.cos(t))
    fourier_ref = Fourier(N_HBM=150, L_DFT=1024, n_dof=2, real_formulation=True)
    fourier = fourier_ref.__replace__(N_HBM=140)
    x0 = np.zeros((2, fourier.L_DFT))

    # ---- ODEs ----
    ode = HillODE(t=0, x=x0, g_fcn=g_fun, **params)
    ode_inv = HillWithMassInverted(t=0, x=x0, g_fun=g_fun, **params)
    ode_mass = HillWithMass(t=0, x=x0, g_fun=g_fun, **params)

    # ---- Solutions ----
    hbm_ref = solve_hbm(equ=ode, fourier=fourier_ref)
    axes = plot_everything(hbm=hbm_ref, label="ref")

    hbm_dir = solve_hbm(equ=ode, fourier=fourier)
    axes = plot_everything(hbm_dir, hbm_ref, axes, label="no mass", linestyle="--")

    hbm_inv = solve_hbm(equ=ode_inv, fourier=fourier)
    plot_everything(hbm=hbm_inv, hbm_ref=hbm_ref, axes=axes, label="inv", linestyle=":")

    hbm_mass = solve_hbm(equ=ode_mass, fourier=fourier, dae=True)
    plot_everything(hbm_mass, hbm_ref, axes, label="with mass", linestyle="-.")

    for ax in axes:
        ax.legend()

    # ---- g functions + spectra ----
    g, g_inv = plot_g_functions(ode, ode_mass, fourier)
    fs = hbm_mass.ode.dynamics(t=fourier.time_samples(ode.omega), x=hbm_mass.x_time())
    # plot_fourier_coeffs(fourier_ref.N_HBM, g, g_inv, hbm_ref.x_time(), fs)
    plot_fourier_coeffs(fourier_ref.N_HBM, g, g_inv)

    # ---- Hill matrices ----
    hill_matrix_ref = hbm_dir.hill_matrix(real_formulation=False, update=True)
    hill_matrix_inv = hbm_inv.hill_matrix(real_formulation=False, update=True)

    plot_matrix_block_norm(
        matrix=hill_matrix_ref - hill_matrix_inv,
        block_size=hbm_dir.fourier.n_dof,
        ax="error inv before",
        logscale=True,
    )

    M = hbm_mass.M()
    M = hbm_mass.fourier.T_to_cplx_from_real @ M @ hbm_mass.fourier.T_to_real_from_cplx
    hill_matrix_mass = np.linalg.solve(
        M, hbm_mass.hill_matrix(real_formulation=False, update=True)
    )

    plot_matrix_block_norm(
        matrix=hill_matrix_ref - hill_matrix_mass,
        block_size=hbm_dir.fourier.n_dof,
        index=range(-hbm_dir.fourier.N_HBM, hbm_dir.fourier.N_HBM + 1),
        ax="error inv after",
        logscale=True,
    )

    # ---- Prints ----

    if fourier.N_HBM <= 20:
        print("Reference Hill matrix blocks:")
        print_Toeplitz_blocks(clean_matrix(hill_matrix_ref), hbm_dir.fourier.n_dof)

        print("Inverted Mass Hill matrix blocks:")
        print_Toeplitz_blocks(clean_matrix(hill_matrix_inv), hbm_dir.fourier.n_dof)

        print("With Mass Hill matrix blocks:")
        print_Toeplitz_blocks(clean_matrix(hill_matrix_mass), hbm_dir.fourier.n_dof)


def clean_matrix(matrix, tol=1e-10):
    """Set real and imaginary parts of a matrix to zero if smaller than tolerance.

    Parameters
    ----------
    matrix : np.ndarray
        Complex or real matrix to clean.
    tol : float, optional
        Tolerance threshold. Default is 1e-10.

    Returns
    -------
    np.ndarray
        Matrix with small real/imaginary parts zeroed out.
    """
    matrix = matrix.copy()

    # Zero out small real parts
    matrix[np.abs(np.real(matrix)) < tol] = 1j * np.imag(
        matrix[np.abs(np.real(matrix)) < tol]
    )

    # Zero out small imaginary parts
    matrix[np.abs(np.imag(matrix)) < tol] = np.real(
        matrix[np.abs(np.imag(matrix)) < tol]
    )

    return matrix


def print_Toeplitz_blocks(matrix, block_size):
    """Print the blocks of a given square matrix, traversing diagonally."""
    n_blocks = matrix.shape[0] // block_size
    row_start = n_blocks - 1
    col_start = 0
    diag_number = n_blocks - 1
    while col_start < n_blocks and row_start >= 0:
        row = row_start
        col = col_start
        print(f"Diagonal {diag_number}")
        while row < n_blocks and col < n_blocks:
            block = matrix[
                row * block_size : (row + 1) * block_size,
                col * block_size : (col + 1) * block_size,
            ]
            print(block)
            row += 1
            col += 1
        if row_start > 0:
            row_start -= 1
        else:
            col_start += 1

        diag_number -= 1
        print("\n")


def plot_g_functions(ode: HillODE, ode_mass: HillWithMass, fourier: Fourier):
    """Plot g and 1/g functions in time and frequency domain."""
    t_samples = fourier.time_samples(ode.omega)

    # Time domain plot
    _, ax_cos = plt.subplots()
    g = ode.g_fcn(t_samples)
    ax_cos.plot(t_samples, g, "-", label="g")

    g_inv = ode_mass.g_inv(t_samples)
    ax_cos.plot(t_samples, g_inv, "--", label="1/g")
    ax_cos.plot(t_samples, g * g_inv, "-.", label="g*1/g")
    ax_cos.set_xlabel("t")
    ax_cos.legend()

    return g, g_inv


def plot_fourier_coeffs(N_HBM=30, *funs):
    """Frequency domain plot"""
    _, ax = plt.subplots()

    bar_width = (1 - 0.1) / len(funs)
    indices = np.arange(-N_HBM, N_HBM + 1)

    for i, fun in enumerate(funs):
        if len(fun.shape) == 1:
            fun = fun[np.newaxis, :]
        fourier = Fourier(
            N_HBM=N_HBM, L_DFT=fun.shape[1], n_dof=fun.shape[0], real_formulation=False
        )
        coeffs = fourier.DFT(fun)
        coeffs = np.reshape(coeffs, (fun.shape[0], -1), order="F")
        x = indices + (i - (len(funs) - 1) / 2) * bar_width
        ax.bar(
            x,
            np.linalg.norm(coeffs, axis=0),
            width=bar_width,
        )

    ax.set_xlabel("Harmonic index")
    ax.set_yscale("log")
    return ax


def solve_hbm(
    equ,
    fourier,
    dae=False,
):
    if dae:
        hbm = HBMEquationDAE(
            dae=equ,
            omega=equ.omega,
            fourier=fourier,
            stability_method=KoopmanHillDAE(fourier),
        )
    else:
        hbm = HBMEquation(
            ode=equ,
            omega=equ.omega,
            fourier=fourier,
            stability_method=KoopmanHillProjection(fourier),
        )

    solver = NewtonSolver(verbose=True)
    solver.solve_equation(hbm, "X")
    return hbm


def plot_everything(hbm, hbm_ref=None, axes=(None, None, None, None), **kwargs_plot):
    axes = list(axes)
    axes[0] = plot_phase(hbm, ax=axes[0], **kwargs_plot)
    axes[1] = plot_period(hbm, ax=axes[1], **kwargs_plot)
    axes[2] = plot_floquet_multipliers(hbm, ax=axes[2], **kwargs_plot)

    if hbm_ref is not None:
        if axes[-1] is None:
            _, axes[-1] = plt.subplots()
            axes[-1].set_yscale("log")
            axes[-1].set_xlabel("t")
            axes[-1].set_ylabel("Error norm")

        t_samples = hbm.fourier.time_samples(hbm.omega)
        error_dir = np.linalg.norm(hbm.x_time() - hbm_ref.x_time(), axis=0)
        axes[-1].plot(t_samples, error_dir, **kwargs_plot)

    return axes


def plot_matrix_block_norm(
    matrix,
    block_size,
    ax=None,
    index=None,
    logscale=False,
    vmax=None,
    vmin=None,
    **scatter_kwargs,
):
    """
    Plot a given square matrix as a grid of blocks, colored by their 2-norm.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to be plotted.
    block_size : int
        The size of each block (assumed square).
    ax : matplotlib.axes.Axes, optional
        The :py:class:`~matplotlib.axes.Axes` object on which to plot. If ``None``, a new :py:class:`~matplotlib.axes.Axes` instance will be created.
    cmap : str, optional
        The colormap to use for coloring the dots by block norm. Default is 'viridis'.
    **scatter_kwargs
        Additional keyword arguments passed to ``ax.scatter()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The :py:class:`~matplotlib.axes.Axes` object with the plotted matrix blocks.
    sc : matplotlib.collections.PathCollection
        The scatter plot collection object (for accessing colorbar, etc.).
    """

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Input matrix must be 2-D and square but has shape {matrix.shape}"
        )

    if index is None:
        index = range(matrix.shape[0] // block_size)

    num_blocks = len(index)

    if num_blocks * block_size != matrix.shape[0]:
        raise ValueError(
            f"Matrix size ({matrix.shape[0]}) not given by block size ({block_size}) times number of blocks ({num_blocks})"
        )

    x_positions = []
    y_positions = []
    norm_values = []

    for i in range(num_blocks):
        for j in range(num_blocks):
            block = matrix[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size,
            ]
            norm = np.linalg.norm(block, ord=2)
            x_positions.append(index[j])
            y_positions.append(index[i])
            norm_values.append(norm)

    if isinstance(ax, str):
        title = ax
    else:
        title = ""

    if ax is None or isinstance(ax, str):
        _, ax = plt.subplots()
        ax.set_xlabel("Column index")
        ax.set_ylabel("Row index")
        ax.set_title(title)
        ax.invert_yaxis()

    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)
    norm_values = np.array(norm_values)

    scatter_defaults = {"cmap": "viridis", "s": 100, "alpha": 0.8}
    if logscale:
        scatter_defaults["norm"] = matplotlib.colors.LogNorm(
            vmax=vmax, vmin=vmin, clip=False
        )
    scatter_defaults.update(scatter_kwargs)

    sc = ax.scatter(
        x_positions,
        y_positions,
        c=norm_values,
        **scatter_defaults,
    )

    cbar = plt.colorbar(sc, ax=ax)
    if logscale:
        cbar.formatter = plt.matplotlib.ticker.LogFormatterMathtext(base=10)
        cbar.update_ticks()
    cbar.set_label("2-norm of block")

    return ax


def plot_hill_matrix_blocks(
    hbm: HBMEquation, real_formulation=False, ax=None, **scatter_kwargs
):
    """
    Plot the Hill matrix as a grid of blocks, colored by their 2-norm.

    This function computes the Hill matrix of a solved :py:class:`~skhippr.cycles.hbm.HBMEquation`,
    segments it into n_dof x n_dof blocks, and creates a scatter plot where each block is represented
    as a dot. The color of each dot is determined by the 2-norm (spectral norm) of the corresponding block.

    Parameters
    ----------
    hbm : HBMEquation or HBMSystem
        The equation or equation system containing the solution. If it is of type :py:class:`~skhippr.cycles.hbm.HBMSystem`, the first valid :py:class:`~skhippr.cycles.hbm.HBMEquation` instance contained is used.
    ax : matplotlib.axes.Axes, optional
        The :py:class:`~matplotlib.axes.Axes` object on which to plot. If ``None``, a new :py:class:`~matplotlib.axes.Axes` instance will be created.
    cmap : str, optional
        The colormap to use for coloring the dots by block norm. Default is 'viridis'.
    **scatter_kwargs
        Additional keyword arguments passed to ``ax.scatter()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The :py:class:`~matplotlib.axes.Axes` object with the plotted Hill matrix blocks.

    Notes
    -----
    The Hill matrix is partitioned into blocks of size n_dof x n_dof, arranged in a 2D grid.
    Each block's position in the plot corresponds to its position in the Hill matrix, and its
    color represents the spectral norm (2-norm) of that block.
    """

    # Compute Hill matrix
    H = hbm.hill_matrix(real_formulation=real_formulation, update=True)
    n_dof = hbm.fourier.n_dof

    if real_formulation:
        index = range(2 * hbm.fourier.N_HBM + 1)
    else:
        index = range(-hbm.fourier.N_HBM, hbm.fourier.N_HBM + 1)
    ax = plot_matrix_block_norm(H, n_dof, ax=ax, index=index, **scatter_kwargs)
    return ax


if __name__ == "__main__":
    main()
    plt.show()
