import pytest
import numpy as np
from skhippr.Fourier import Fourier


def test_dimensions(fourier_small: Fourier):
    """Test the dimensions of the DFT matrix."""

    assert fourier_small.DFT_matrix.shape == (
        fourier_small.n_dof * (2 * fourier_small.N_HBM + 1),
        fourier_small.n_dof * fourier_small.L_DFT,
    )

    assert fourier_small.iDFT_matrix.shape == (
        fourier_small.n_dof * fourier_small.L_DFT,
        fourier_small.n_dof * (2 * fourier_small.N_HBM + 1),
    )


def test_time_samples(fourier_small: Fourier, verbose=False):
    if verbose:
        print(fourier_small, end=": ")
        print("Test time samples:", end=" ")

    omega = np.random.rand()
    ts = fourier_small.time_samples(omega)
    assert len(ts) == fourier_small.L_DFT

    T = 2 * np.pi / omega
    dt = T / fourier_small.L_DFT
    assert np.allclose(ts, np.arange(0, T, dt), atol=2e-14)

    if verbose:
        print("success")


def test_time_samples_multiple_periods(fourier_small: Fourier, verbose=False):
    if verbose:
        print(fourier_small, end=": ")
        print("Test time samples (multiple periods):", end=" ")

    omega = np.random.rand()
    T = 2 * np.pi / omega
    dt = T / fourier_small.L_DFT

    for periods in (3, np.sqrt(2)):
        ts = fourier_small.time_samples(omega, periods=periods)
        assert ts[0] == 0.0
        assert np.allclose(ts[1:] - ts[:-1], dt, atol=1e-14)
        assert ts[-1] >= (T * periods - dt) - 1e-14
        assert ts[-1] < T * periods

    if verbose:
        print("success")


def test_reconstruct_X(fourier_small: Fourier, verbose=False):
    if verbose:
        print(fourier_small, end=": ")
        print("Generate and reconstruct a vector of Fourier coefficients:", end=" ")

    X = np.random.rand(fourier_small.n_dof * (2 * fourier_small.N_HBM + 1))
    x_samp = fourier_small.inv_DFT(X)
    x_samp_deprecated = fourier_small._inv_DFT(X)
    assert np.allclose(x_samp, x_samp_deprecated)

    X_reconstructed = fourier_small.DFT(x_samp)
    X_reconstructed_deprecated = fourier_small._DFT(x_samp)

    assert np.allclose(X, X_reconstructed)
    assert np.allclose(X, X_reconstructed_deprecated)
    if verbose:
        print("success")


def generate_x_samp(DFT, omega):
    ts = DFT.time_samples(omega)
    assert DFT.N_HBM >= 3  # x_samp(t) has 3 harmonics

    x_samp = np.vstack(
        (np.sin(omega * ts) + 4 * np.cos(3 * omega * ts), np.ones_like(ts))
    )

    x_derivative = np.vstack(
        (
            omega * np.cos(omega * ts) - 12 * omega * np.sin(3 * omega * ts),
            np.zeros_like(ts),
        )
    )

    return x_samp, x_derivative


def test_reconstruct_x_samp(fourier_small: Fourier, verbose=False):
    if verbose:
        print(fourier_small, end=": ")
        print(
            "Generate and reconstruct a vector of time samples within Nyquist freq:",
            end=" ",
        )
    omega = np.random.rand()
    assert fourier_small.N_HBM >= 3  # Prevent aliasing

    omega = np.random.rand()
    x_samp, _ = generate_x_samp(fourier_small, omega)

    # Transform to frequency domain
    X = fourier_small.DFT(x_samp)
    X_deprecated = fourier_small._DFT(x_samp)
    assert np.allclose(X, X_deprecated)

    # back to freq domain
    x_samp_reconstructed = fourier_small.inv_DFT(X)
    x_samp_reconstructed_deprecated = fourier_small._inv_DFT(X)

    assert np.allclose(x_samp, x_samp_reconstructed)
    assert np.allclose(x_samp, x_samp_reconstructed_deprecated)
    if verbose:
        print("success")


def test_derivative(fourier_small: Fourier, verbose=False):
    if verbose:
        print(fourier_small, end=": ")
        print("Test derivative matrix:", end=" ")

    omega = np.random.rand()
    x_samp, x_deriv = generate_x_samp(fourier_small, omega)

    # Derivative function
    assert np.allclose(x_deriv, fourier_small.differentiate(x_samp, omega))

    # Derivative matrix
    X = fourier_small.DFT(x_samp)
    X_deriv = omega * fourier_small.derivative_matrix @ X
    x_deriv_reconstructed = fourier_small.inv_DFT(X_deriv)
    assert np.allclose(x_deriv, x_deriv_reconstructed)

    print("success")


def test_limit_frequency(fourier_small: Fourier, verbose=False):
    if verbose:
        print(fourier_small, end=": ")
        print("Add a frequency at N_HBM:", end=" ")

    omega = np.random.rand()
    ts = fourier_small.time_samples(omega)
    x_samp, _ = generate_x_samp(fourier_small, omega)
    x_samp[1, :] += 2 * np.sin(fourier_small.N_HBM * omega * ts + np.pi / 3)
    X = fourier_small.DFT(x_samp)
    x_samp_reconstructed = fourier_small.inv_DFT(X)
    assert np.allclose(x_samp, x_samp_reconstructed)
    if verbose:
        print("success")


def test_too_large_frequency(fourier_small: Fourier, verbose=False):
    if verbose:
        print(fourier_small, end=": ")
        print("Add a frequency larger than N_HBM:", end=" ")

    omega = np.random.rand()
    ts = fourier_small.time_samples(omega)
    x_samp, _ = generate_x_samp(fourier_small, omega)
    x_samp[1, :] += 2 * np.sin((fourier_small.N_HBM + 1) * omega * ts + np.pi / 3)
    X = fourier_small.DFT(x_samp)
    x_samp_reconstructed = fourier_small.inv_DFT(X)
    assert not np.allclose(x_samp, x_samp_reconstructed)
    if verbose:
        print("success (expected results to differ)")


def test_matrix_DFT_and_iDFT(fourier):
    omega = np.random.rand()
    phase_3 = np.random.rand()

    # Check minimum required frequency
    assert fourier.N_HBM >= 5
    ts = fourier.time_samples(omega)[np.newaxis, np.newaxis, :]

    # Construct test case
    A_samples = np.vstack(
        (
            np.hstack(
                (
                    3 * np.cos(omega * ts + 3),
                    -np.sin(omega * ts) + 0.5 * np.cos(5 * omega * ts + phase_3),
                )
            ),
            np.hstack(
                (-4 * np.sin(2 * omega * ts) + 3, np.ones((1, 1, fourier.L_DFT)))
            ),
        )
    )

    # Compare transformations
    A = fourier.matrix_DFT(A_samples)
    A_old = fourier._matrix_DFT(A_samples)

    print(f"Difference: {np.max(np.abs(A - A_old))}")

    assert np.allclose(A, A_old, atol=1e-13, rtol=1e-13)

    # Compare inverse transformations
    A_samples_fft = fourier.matrix_inv_DFT(A)
    assert np.allclose(A_samples_fft, A_samples, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("real_formulation", [True, False])
@pytest.mark.parametrize("input_shape", ["vector", "matrix"])
@pytest.mark.parametrize("N_new", [2, 5, 8])
def test_resize_coefficients(real_formulation, input_shape, N_new):
    """Test resizing Fourier coefficients across different N_HBM values.

    The test compares ``resize_coefficients`` against the reference procedure
    of iDFT with the old Fourier object and then DFT with the new Fourier object.
    """

    n_dof = 3
    N_old = 5
    L_DFT = 128

    fourier_old = Fourier(
        N_HBM=N_old,
        L_DFT=L_DFT,
        n_dof=n_dof,
        real_formulation=real_formulation,
    )
    fourier_new = Fourier(
        N_HBM=N_new,
        L_DFT=L_DFT,
        n_dof=n_dof,
        real_formulation=real_formulation,
    )

    coeff_count_old = 2 * N_old + 1
    if real_formulation:
        X_old = np.random.randn(n_dof * coeff_count_old)
    else:
        X_old = np.random.randn(n_dof * coeff_count_old) + 1j * np.random.randn(
            n_dof * coeff_count_old
        )

    if input_shape == "matrix":
        X_old = np.reshape(X_old, (n_dof, coeff_count_old), order="F")

    X_resized = fourier_new.resize_coefficients(X_old)
    X_old_vector = (
        X_old if input_shape == "vector" else np.reshape(X_old, -1, order="F")
    )
    x_time = fourier_old.inv_DFT(X_old_vector)

    X_expected = fourier_new.DFT(x_time)

    if input_shape == "matrix":
        X_expected = np.reshape(X_expected, (n_dof, -1), order="F")

    assert X_resized.shape == X_expected.shape
    assert np.allclose(X_resized, X_expected, atol=1e-14, rtol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__])
