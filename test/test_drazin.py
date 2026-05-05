"""
Comprehensive unit tests for Drazin inverse implementations in KoopmanHillProjection.py
Tests cover:
  - Example matrix from Soleymani2013 paper
  - Call signature uniformity across all implementations
  - Nonsingular matrices (index 0)
  - Singular matrices with various algebraic indices
"""

import pytest
import numpy as np
import inspect
from skhippr.stability.KoopmanHillProjection import (
    drazin,
    drazin_schur_2,
    drazin_ord2,
    drazin_ord9,
    compute_index,
)

# ============================================================================
# Example matrix fixtures (from Soleymani2013 paper)
# ============================================================================


def drazin_example_matrix():
    """Example matrix for Drazin inversion, taken from Soleymani2013, Eq. (52)."""
    A = np.zeros((12, 12))

    A[0, 0] = 2
    A[0, 1] = 0.4
    A[1, 0] = -2
    A[1, 1] = 0.4

    A[2, 0] = -1
    A[2, 1] = -1
    A[2, 2] = 1
    A[2, 3] = -1
    A[2, 8] = -1

    A[3, 0] = -1
    A[3, 1] = -1
    A[3, 2] = -1
    A[3, 3] = 1

    A[4, 4] = 1
    A[4, 5] = 1
    A[4, 6] = -1
    A[4, 7] = -1
    A[4, 10] = -1

    A[5, 4] = 1
    A[5, 5] = 1
    A[5, 6] = -1
    A[5, 7] = -1

    A[6, 3] = -1
    A[6, 4] = -2
    A[6, 5] = 0.4

    A[7, 4] = 2
    A[7, 5] = 0.4

    A[8, 1] = -1
    A[8, 8] = 1
    A[8, 9] = -1
    A[8, 10] = -1
    A[8, 11] = -1

    A[9, 8] = -1
    A[9, 9] = 1
    A[9, 10] = -1
    A[9, 11] = -1

    A[10, 10] = 0.4
    A[10, 11] = -2
    A[11, 10] = 0.4
    A[11, 11] = 2

    return A


def drazin_example_result():
    """Drazin inversion result for example matrix (Soleymani2013, Eq. (53))."""
    A = np.zeros((12, 12))

    A[0, 0] = 0.25
    A[0, 1] = -0.25

    A[1, 0] = 1.25
    A[1, 1] = 1.25

    A[2, 0] = -1.66406
    A[2, 1] = -0.992187
    A[2, 2] = 0.25
    A[2, 3] = -0.25
    A[2, 8] = -0.0625
    A[2, 9] = -0.0625
    A[2, 11] = 0.15625

    A[3, 0] = -1.19531
    A[3, 1] = -0.679687
    A[3, 2] = -0.25
    A[3, 3] = 0.25
    A[3, 8] = -0.0625
    A[3, 9] = 0.1875
    A[3, 10] = 0.6875
    A[3, 11] = 1.34375

    A[4, 0] = -2.76367
    A[4, 1] = -1.04492
    A[4, 2] = -1.875
    A[4, 3] = -1.25
    A[4, 4] = -1.25
    A[4, 5] = 1.25
    A[4, 6] = 1.25
    A[4, 7] = 1.25
    A[4, 8] = 1.48438
    A[4, 9] = 2.57813
    A[4, 10] = 3.32031
    A[4, 11] = 6.64063

    A[5, 0] = -2.76367
    A[5, 1] = -1.04492
    A[5, 2] = -1.875
    A[5, 3] = -1.25
    A[5, 4] = -1.25
    A[5, 5] = 1.25
    A[5, 6] = 1.25
    A[5, 7] = 1.25
    A[5, 8] = 1.48438
    A[5, 9] = 2.57813
    A[5, 10] = 4.57031
    A[5, 11] = 8.51563

    A[6, 0] = 14.1094
    A[6, 1] = 6.30078
    A[6, 2] = 6.625
    A[6, 3] = 3.375
    A[6, 4] = 5
    A[6, 5] = -3
    A[6, 6] = -5
    A[6, 7] = -5
    A[6, 8] = -4.1875
    A[6, 9] = -8.5
    A[6, 10] = -10.5078
    A[6, 11] = -22.4609

    A[7, 0] = -19.3242
    A[7, 1] = -8.50781
    A[7, 2] = -9.75
    A[7, 3] = -5.25
    A[7, 4] = -7.5
    A[7, 5] = 4.5
    A[7, 6] = 7.5
    A[7, 7] = 7.5
    A[7, 8] = 6.375
    A[7, 9] = 12.5625
    A[7, 10] = 15.9766
    A[7, 11] = 33.7891

    A[8, 0] = -0.625
    A[8, 1] = -0.3125
    A[8, 8] = 0.25
    A[8, 9] = -0.25
    A[8, 10] = -0.875
    A[8, 11] = -1.625

    A[9, 0] = -1.25
    A[9, 1] = -0.9375
    A[9, 8] = -0.25
    A[9, 9] = 0.25
    A[9, 10] = -0.875
    A[9, 11] = -1.625

    A[10, 10] = 1.25
    A[10, 11] = 1.25
    A[11, 10] = -0.25
    A[11, 11] = 0.25

    return A


# ============================================================================
# Helper functions for Drazin inverse criteria verification
# ============================================================================


def verify_drazin_criteria(A, A_D, k, tol=1e-6):
    """
    Verify the three Drazin inverse criteria:
    1. A @ A^k @ A_D == A^k
    2. A_D @ A @ A_D == A_D
    3. A @ A_D == A_D @ A (commutativity)

    Parameters
    ----------
    A : np.ndarray
        Original matrix
    A_D : np.ndarray
        Drazin inverse
    k : int
        Algebraic index
    tol : float
        Tolerance for verification

    Returns
    -------
    dict
        Results of each criterion test
    """
    A_k = np.linalg.matrix_power(A, k)

    # Criterion 1: A @ A^k @ A_D == A^k
    crit1_lhs = A @ A_k @ A_D
    crit1_rhs = A_k
    err1 = np.linalg.norm(crit1_lhs - crit1_rhs, np.inf)
    crit1_pass = err1 < tol

    # Criterion 2: A_D @ A @ A_D == A_D
    crit2_lhs = A_D @ A @ A_D
    crit2_rhs = A_D
    err2 = np.linalg.norm(crit2_lhs - crit2_rhs, np.inf)
    crit2_pass = err2 < tol

    # Criterion 3: A @ A_D == A_D @ A (commutativity)
    crit3_lhs = A @ A_D
    crit3_rhs = A_D @ A
    err3 = np.linalg.norm(crit3_lhs - crit3_rhs, np.inf)
    crit3_pass = err3 < tol

    return {
        "crit1_error": err1,
        "crit1_pass": crit1_pass,
        "crit2_error": err2,
        "crit2_pass": crit2_pass,
        "crit3_error": err3,
        "crit3_pass": crit3_pass,
        "all_pass": crit1_pass and crit2_pass and crit3_pass,
    }


# ============================================================================
# Test: Call signature uniformity
# ============================================================================


class TestCallSignature:
    """Test that all Drazin implementations have the same call signature."""

    def test_drazin_signatures_uniform(self):
        """Verify all Drazin functions accept (A, tol, ...) signature."""
        implementations = [drazin, drazin_schur_2, drazin_ord2, drazin_ord9]

        # Each should accept A (required) and tol (optional, keyword/positional)
        for impl in implementations:
            sig = inspect.signature(impl)
            params = list(sig.parameters.keys())

            # First parameter should be A (or matrix-like)
            assert (
                params[0] == "A"
            ), f"{impl.__name__} first param is {params[0]}, not 'A'"

            # Second parameter should be tol (or similar tolerance parameter)
            assert (
                params[1] == "tol"
            ), f"{impl.__name__} second param is {params[1]}, not 'tol'"

            # tol should have a default value
            assert (
                sig.parameters["tol"].default is not inspect.Parameter.empty
            ), f"{impl.__name__} tol parameter has no default"

    def test_drazin_returns_tuple(self):
        """Verify all Drazin functions return (A_D, k) tuple."""
        A = drazin_example_matrix()
        implementations = [drazin, drazin_schur_2, drazin_ord2, drazin_ord9]

        for impl in implementations:
            result = impl(A, tol=1e-6)
            assert isinstance(result, tuple), f"{impl.__name__} did not return a tuple"
            assert (
                len(result) == 2
            ), f"{impl.__name__} returned {len(result)}-tuple, expected 2-tuple"
            A_D, k = result
            assert isinstance(
                A_D, np.ndarray
            ), f"{impl.__name__} first return value is not ndarray"
            assert isinstance(
                k, (int, float, np.integer)
            ), f"{impl.__name__} second return value is not numeric"


# ============================================================================
# Test: Example matrix from Soleymani2013
# ============================================================================


class TestExampleMatrix:
    """Test all Drazin implementations on the Soleymani2013 example matrix."""

    @pytest.fixture
    def example_data(self):
        A = drazin_example_matrix()
        A_D_expected = drazin_example_result()
        return A, A_D_expected

    @pytest.mark.parametrize(
        "impl_name,impl_func",
        [
            ("drazin", drazin),
            ("drazin_schur_2", drazin_schur_2),
            ("drazin_ord2", drazin_ord2),
            ("drazin_ord9", drazin_ord9),
        ],
    )
    def test_example_matrix_criteria(self, impl_name, impl_func, example_data):
        """Test Drazin criteria on example matrix."""
        A, _ = example_data
        A_D, k = impl_func(A, tol=1e-8)

        results = verify_drazin_criteria(A, A_D, int(k), tol=1e-5)
        assert results["all_pass"], (
            f"{impl_name} failed criteria:\n"
            f"  crit1: {results['crit1_error']:.2e}\n"
            f"  crit2: {results['crit2_error']:.2e}\n"
            f"  crit3: {results['crit3_error']:.2e}"
        )

    @pytest.mark.parametrize(
        "impl_name,impl_func",
        [
            ("drazin", drazin),
            ("drazin_schur_2", drazin_schur_2),
            ("drazin_ord2", drazin_ord2),
            ("drazin_ord9", drazin_ord9),
        ],
    )
    def test_example_matrix_matches_reference(self, impl_name, impl_func, example_data):
        """Test that computed Drazin inverse matches Soleymani2013 reference."""
        A, A_D_expected = example_data
        A_D, k = impl_func(A, tol=1e-8)

        error = np.linalg.norm(A_D - A_D_expected, np.inf)
        assert error < 1e-4, f"{impl_name} result differs from reference by {error:.2e}"


# ============================================================================
# Test: Nonsingular matrix (index 0)
# ============================================================================


class TestNonsingularMatrix:
    """Test Drazin on a nonsingular matrix (should satisfy A_D @ A = I)."""

    @pytest.fixture
    def nonsingular_matrix(self):
        """Create a well-conditioned nonsingular matrix."""
        np.random.seed(42)
        A = np.random.randn(20, 20)
        # Ensure it's invertible by making it diagonally dominant
        A = A + 30 * np.eye(20)
        return A

    @pytest.mark.parametrize(
        "impl_name,impl_func",
        [
            ("drazin", drazin),
            ("drazin_schur_2", drazin_schur_2),
            ("drazin_ord9", drazin_ord9),
        ],
    )
    def test_nonsingular_index_zero(self, impl_name, impl_func, nonsingular_matrix):
        """Test that drazin_ord9 correctly identifies index 0."""
        A = nonsingular_matrix
        A_D, k = impl_func(A, tol=1e-10)

        # Index should be 0 for nonsingular matrix
        assert k == 0, f"{impl_name} computed index {k} for nonsingular matrix"

    @pytest.mark.parametrize(
        "impl_name,impl_func",
        [
            ("drazin", drazin),
            ("drazin_schur_2", drazin_schur_2),
            ("drazin_ord9", drazin_ord9),
        ],
    )
    def test_nonsingular_satisfies_inverse(
        self, impl_name, impl_func, nonsingular_matrix
    ):
        """Test that for nonsingular matrix: A_D @ A = I."""
        A = nonsingular_matrix
        A_D, k = impl_func(A, tol=1e-10)

        product = A_D @ A
        I = np.eye(A.shape[0])
        error = np.linalg.norm(product - I, np.inf)
        assert error < 1e-8, f"{impl_name}: A_D @ A != I, error = {error:.2e}"

    @pytest.mark.parametrize(
        "impl_name,impl_func",
        [
            ("drazin", drazin),
            ("drazin_schur_2", drazin_schur_2),
            ("drazin_ord9", drazin_ord9),
        ],
    )
    def test_nonsingular_criteria(self, impl_name, impl_func, nonsingular_matrix):
        """Test Drazin criteria on nonsingular matrix."""
        A = nonsingular_matrix
        A_D, k = impl_func(A, tol=1e-10)

        results = verify_drazin_criteria(A, A_D, int(k), tol=1e-8)
        assert results["all_pass"], (
            f"{impl_name} failed criteria on nonsingular matrix:\n"
            f"  crit1: {results['crit1_error']:.2e}\n"
            f"  crit2: {results['crit2_error']:.2e}\n"
            f"  crit3: {results['crit3_error']:.2e}"
        )


# ============================================================================
# Test: Singular matrices with various algebraic indices
# ============================================================================


class TestSingularMatrices:
    """Test Drazin on singular matrices with controlled algebraic indices."""

    @staticmethod
    def create_jordan_block(eigenvalue, size):
        """Create a Jordan block with given eigenvalue and size."""
        J = np.eye(size) * eigenvalue
        for i in range(size - 1):
            J[i, i + 1] = 1
        return J

    @staticmethod
    def create_singular_matrix_with_index(n, index):
        """
        Create an n×n singular matrix with algebraic index equal to 'index'.

        Constructed by placing a Jordan block with eigenvalue 0 of size (index)
        in the top-left, and nonzero eigenvalues elsewhere.

        Parameters
        ----------
        n : int
            Matrix dimension
        index : int
            Desired algebraic index (size of largest zero Jordan block)

        Returns
        -------
        A : np.ndarray
            Singular matrix with the specified index
        """
        assert index <= n, "Index cannot exceed matrix dimension"

        # Create Jordan normal form with a nilpotent block of size 'index'
        J_nil = TestSingularMatrices.create_jordan_block(0, index)

        # Fill remaining diagonal with nonzero eigenvalues
        remaining_eigs = np.linspace(1, 5, n - index)
        J_nz = np.diag(remaining_eigs)

        # Combine into full Jordan form
        J = np.zeros((n, n))
        J[:index, :index] = J_nil
        J[index:, index:] = J_nz

        # Apply random similarity transformation
        np.random.seed(42 + index)  # Reproducible but different for each index
        P = np.random.randn(n, n)

        # Ensure P is invertible
        while np.linalg.cond(P) > 1e10:
            P = np.random.randn(n, n)

        # A = P @ J @ P^{-1}
        A = P @ J @ np.linalg.inv(P)

        return A

    @pytest.mark.parametrize("index", [1, 2, 5, 10, 25])
    def test_singular_index_detection(self, index):
        """Test that drazin_ord9 correctly detects algebraic index."""
        n = 51
        A = self.create_singular_matrix_with_index(n, index)

        A_D, k_computed = drazin_ord9(A, tol=1e-8)

        assert (
            int(k_computed) == index
        ), f"Expected index {index}, but got {int(k_computed)}"

    @pytest.mark.parametrize("index", [1, 2, 5, 10, 25])
    def test_singular_criteria_all_indices(self, index):
        """Test Drazin criteria on singular matrices with various indices."""
        n = 51
        A = self.create_singular_matrix_with_index(n, index)

        A_D, k = drazin_ord9(A, tol=1e-8)

        results = verify_drazin_criteria(A, A_D, int(k), tol=1e-6)
        assert results["all_pass"], (
            f"Index {index}: Drazin criteria failed:\n"
            f"  crit1: {results['crit1_error']:.2e}\n"
            f"  crit2: {results['crit2_error']:.2e}\n"
            f"  crit3: {results['crit3_error']:.2e}"
        )

    @pytest.mark.parametrize("index", [1, 2, 5, 10, 25])
    def test_singular_schur_criteria(self, index):
        """Test drazin_schur_2 on singular matrices (should also work)."""
        n = 51
        A = self.create_singular_matrix_with_index(n, index)

        A_D, k = drazin_schur_2(A, tol=1e-8)

        results = verify_drazin_criteria(A, A_D, int(k), tol=1e-6)
        assert results["all_pass"], (
            f"Index {index} (drazin_schur_2): Drazin criteria failed:\n"
            f"  crit1: {results['crit1_error']:.2e}\n"
            f"  crit2: {results['crit2_error']:.2e}\n"
            f"  crit3: {results['crit3_error']:.2e}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
