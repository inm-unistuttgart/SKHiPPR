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
    drazin_rothblum_custom,
    compute_index,
    row_reduced_echelon_form,
    back_substitution,
)

implementations = [drazin_rothblum_custom, drazin_schur_2]

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


def verify_drazin_criteria(A, A_D, k, tol=1e-8):
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


class TestGaussJordan:
    """Tests for row-reduced echelon form and back-substitution algorithms."""

    def is_rref(self, A, tol=0):
        """Check whether A is in reduced row-echelon form.

        Conditions checked:
        - Each nonzero row has a leading 1.
        - Leading 1s move strictly to the right in lower rows.
        """
        A = A.copy()
        n_rows, _ = A.shape
        first_prev = -1
        for r in range(n_rows):
            row = A[r]
            nz = np.where(np.abs(row) > tol)[0]
            if nz.size == 0:
                continue
            first = nz[0]
            # leading value must be 1
            if A[r, first] != 1.0:
                return False
            # leading positions strictly increasing
            if first <= first_prev:
                return False
            first_prev = first
        return True

    def test_rref_and_backsub_nonsingular(self):
        """For nonsingular A, rref -> identity and back_substitution yields identity; B becomes inverse."""
        np.random.seed(928574)
        n = 8
        A = np.random.randn(n, n)
        # ensure invertible
        while np.linalg.cond(A) > 1e6:
            A = np.random.randn(n, n)

        B = np.eye(n)

        A_rref, B_trans = row_reduced_echelon_form(A, B, in_place=False)
        assert self.is_rref(A_rref, tol=1e-8)

        # After full Gauss-Jordan on a nonsingular square matrix the result should be identity
        A_final, B_final = back_substitution(A_rref, B_trans, in_place=False)

        assert np.allclose(A_final, np.eye(n), atol=1e-14)

        # B_final should be the inverse of original A
        assert np.allclose(B_final @ A, np.eye(n), atol=1e-7)

    def test_rref_singular(self):
        """RREF should work for singular matrices and maintain operations on B."""
        # Construct rank-deficient matrix
        n = 7
        A = np.random.randn(n, n)

        # make one row a linear combination of other rows
        factors = np.random.randn(n - 1)
        A[-1, :] = 0
        for i in range(n - 1):
            A[-1, :] += factors[i] * A[i, :]

        # put the linearly dependent row somewhere
        pos = np.random.randint(0, n)
        A[[pos, -1], :] = A[[-1, pos], :]

        A_rref, B_rref = row_reduced_echelon_form(A, A, in_place=False)

        assert self.is_rref(A_rref, tol=0)
        assert np.allclose(
            B_rref, A_rref, atol=1e-14
        ), f" discrepancy between A and B is {np.linalg.norm(B_rref - A_rref, np.inf)}"

        # Back substitution should not change rref property and should keep same row ops on B
        A_bs, B_bs = back_substitution(A_rref, B_rref, in_place=False)
        assert self.is_rref(A_bs, tol=0)
        assert np.allclose(A_bs, B_bs, atol=1e-14, rtol=1e-14)

        # check that last row of A is zero
        assert np.all(np.abs(A_rref[-1, :]) == 0)
        assert np.all(np.abs(A_bs[-1, :]) == 0)

    def test_in_place_behavior(self):
        """If in_place=True mutate inputs; if in_place=False keep inputs unchanged."""
        np.random.seed(2026)
        n = 6
        A0 = np.random.randn(n, n)
        while np.linalg.cond(A0) > 1e6:
            A0 = np.random.randn(n, n)
        B0 = np.eye(n)

        # in_place=False: original A and B must remain unchanged
        A_false = A0.copy()
        B_false = B0.copy()
        A_rref_false, B_rref_false = row_reduced_echelon_form(
            A_false, B_false, in_place=False
        )
        assert np.allclose(A_false, A0)
        assert np.allclose(B_false, B0)
        A_bs_false, B_bs_false = back_substitution(
            A_rref_false, B_rref_false, in_place=False
        )
        assert np.allclose(A_rref_false, A_rref_false.copy())
        assert np.allclose(B_rref_false, B_rref_false.copy())

        # in_place=True: passed arrays should be overwritten
        A_true = A0.copy()
        B_true = B0.copy()
        A_rref_true, B_rref_true = row_reduced_echelon_form(
            A_true, B_true, in_place=True
        )
        assert np.allclose(A_true, A_rref_true)
        assert np.allclose(B_true, B_rref_true)
        assert not np.allclose(A_true, A0)
        assert not np.allclose(B_true, B0)

        A_before_bs = A_true.copy()
        B_before_bs = B_true.copy()
        A_bs_true, B_bs_true = back_substitution(A_true, B_true, in_place=True)
        assert np.allclose(A_true, A_bs_true)
        assert np.allclose(B_true, B_bs_true)
        assert not np.allclose(A_true, A_before_bs)
        assert not np.allclose(B_true, B_before_bs)


# ============================================================================
# Test: Call signature uniformity
# ============================================================================


class TestCallSignature:
    """Test that all Drazin implementations have the same call signature."""

    @pytest.mark.parametrize(
        "impl",
        implementations,
    )
    def test_drazin_signatures_uniform(self, impl):
        """Verify Drazin functions accept (A, tol, ...) signature."""

        # Each should accept A (required) and tol (optional, keyword/positional)
        sig = inspect.signature(impl)
        params = list(sig.parameters.keys())

        # First parameter should be A (or matrix-like)
        assert params[0] == "A", f"{impl.__name__} first param is {params[0]}, not 'A'"

        # Second parameter should be tol (or similar tolerance parameter)
        assert (
            params[1] == "tol"
        ), f"{impl.__name__} second param is {params[1]}, not 'tol'"

        # tol should have a default value
        assert (
            sig.parameters["tol"].default is not inspect.Parameter.empty
        ), f"{impl.__name__} tol parameter has no default"

    @pytest.mark.parametrize(
        "impl",
        implementations,
    )
    def test_drazin_returns_tuple(self, impl):
        """Verify all Drazin functions return (A_D, k) tuple."""
        A = drazin_example_matrix()

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
        """
        ============================================================================
        Example matrix fixtures (from Soleymani2013 paper)
        ============================================================================
        """
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

        A_D = np.zeros((12, 12))

        A_D[0, 0] = 0.25
        A_D[0, 1] = -0.25

        A_D[1, 0] = 1.25
        A_D[1, 1] = 1.25

        A_D[2, 0] = -1.66406
        A_D[2, 1] = -0.992187
        A_D[2, 2] = 0.25
        A_D[2, 3] = -0.25
        A_D[2, 8] = -0.0625
        A_D[2, 9] = -0.0625
        A_D[2, 11] = 0.15625

        A_D[3, 0] = -1.19531
        A_D[3, 1] = -0.679687
        A_D[3, 2] = -0.25
        A_D[3, 3] = 0.25
        A_D[3, 8] = -0.0625
        A_D[3, 9] = 0.1875
        A_D[3, 10] = 0.6875
        A_D[3, 11] = 1.34375

        A_D[4, 0] = -2.76367
        A_D[4, 1] = -1.04492
        A_D[4, 2] = -1.875
        A_D[4, 3] = -1.25
        A_D[4, 4] = -1.25
        A_D[4, 5] = 1.25
        A_D[4, 6] = 1.25
        A_D[4, 7] = 1.25
        A_D[4, 8] = 1.48438
        A_D[4, 9] = 2.57813
        A_D[4, 10] = 3.32031
        A_D[4, 11] = 6.64063

        A_D[5, 0] = -2.76367
        A_D[5, 1] = -1.04492
        A_D[5, 2] = -1.875
        A_D[5, 3] = -1.25
        A_D[5, 4] = -1.25
        A_D[5, 5] = 1.25
        A_D[5, 6] = 1.25
        A_D[5, 7] = 1.25
        A_D[5, 8] = 1.48438
        A_D[5, 9] = 2.57813
        A_D[5, 10] = 4.57031
        A_D[5, 11] = 8.51563

        A_D[6, 0] = 14.1094
        A_D[6, 1] = 6.30078
        A_D[6, 2] = 6.625
        A_D[6, 3] = 3.375
        A_D[6, 4] = 5
        A_D[6, 5] = -3
        A_D[6, 6] = -5
        A_D[6, 7] = -5
        A_D[6, 8] = -4.1875
        A_D[6, 9] = -8.5
        A_D[6, 10] = -10.5078
        A_D[6, 11] = -22.4609

        A_D[7, 0] = -19.3242
        A_D[7, 1] = -8.50781
        A_D[7, 2] = -9.75
        A_D[7, 3] = -5.25
        A_D[7, 4] = -7.5
        A_D[7, 5] = 4.5
        A_D[7, 6] = 7.5
        A_D[7, 7] = 7.5
        A_D[7, 8] = 6.375
        A_D[7, 9] = 12.5625
        A_D[7, 10] = 15.9766
        A_D[7, 11] = 33.7891

        A_D[8, 0] = -0.625
        A_D[8, 1] = -0.3125
        A_D[8, 8] = 0.25
        A_D[8, 9] = -0.25
        A_D[8, 10] = -0.875
        A_D[8, 11] = -1.625

        A_D[9, 0] = -1.25
        A_D[9, 1] = -0.9375
        A_D[9, 8] = -0.25
        A_D[9, 9] = 0.25
        A_D[9, 10] = -0.875
        A_D[9, 11] = -1.625

        A_D[10, 10] = 1.25
        A_D[10, 11] = 1.25
        A_D[11, 10] = -0.25
        A_D[11, 11] = 0.25

        return A, A_D

    def test_example_matrix_integrity(self, example_data):
        """Check that example matrix and expected result is indeed the Drazin inverse."""
        A, A_D_expected = example_data
        tol = 1e-2  # High tolerance due to printed values in paper
        results = verify_drazin_criteria(A, A_D_expected, 3, tol=tol)
        assert results[
            "all_pass"
        ], f"Example matrix does not satisfy Drazin criteria, errors are {results}"

    @pytest.mark.parametrize("impl_func", implementations)
    def test_drazin_criteria(self, impl_func, example_data):
        tol = 1e-8
        """Test Drazin criteria on example matrix."""
        A, _ = example_data
        A_D, k = impl_func(A, tol=tol)

        results = verify_drazin_criteria(A, A_D, int(k), tol=10 * tol)
        assert results[
            "all_pass"
        ], f"{impl_func.__name__} failed criteria ({results['crit1_error']:.2e}, {results['crit2_error']:.2e}, {results['crit3_error']:.2e})\n"

    @pytest.mark.parametrize("impl_func", implementations)
    def test_result_matches_reference(self, impl_func, example_data):
        """Test that computed Drazin inverse matches Soleymani2013 reference."""
        A, A_D_expected = example_data
        A_D, k = impl_func(A, tol=1e-8)

        error = np.linalg.norm(A_D - A_D_expected, np.inf)
        assert k == 3, f"{impl_func.__name__} computed index {k}, expected 3"
        assert (
            error < 1e-4
        ), f"{impl_func.__name__} result differs from reference by {error:.2e}"


# ============================================================================
# Test: Nonsingular matrix (index 0)
# ============================================================================


class TestNonsingularMatrix:
    """Test Drazin on a nonsingular matrix (should satisfy A_D @ A = I)."""

    @pytest.fixture
    def nonsingular_matrix(self):
        """Create a well-conditioned nonsingular matrix."""
        np.random.seed(1234)
        cond = np.inf
        while cond > 1e5:
            A = np.random.randn(20, 20)
            cond = np.linalg.cond(A)
        return A

    @pytest.mark.parametrize("impl_func", implementations)
    def test_nonsingular_index_zero(self, impl_func, nonsingular_matrix):
        """Test that drazin_ord9 correctly identifies index 0."""
        A = nonsingular_matrix
        A_D, k = impl_func(A, tol=1e-10)

        # Index should be 0 for nonsingular matrix
        assert k == 0, f"{impl_func.__name__} computed index {k} for nonsingular matrix"

    @pytest.mark.parametrize("impl_func", implementations)
    def test_nonsingular_satisfies_inverse(self, impl_func, nonsingular_matrix):
        """Test that for nonsingular matrix: A_D @ A = I."""
        A = nonsingular_matrix
        A_D, k = impl_func(A, tol=1e-10)

        product = A_D @ A
        I = np.eye(A.shape[0])
        error = np.linalg.norm(product - I, np.inf)
        assert error < 1e-8, f"{impl_func.__name__}: A_D @ A != I, error = {error:.2e}"

    @pytest.mark.parametrize("impl_func", implementations)
    def test_nonsingular_criteria(self, impl_func, nonsingular_matrix):
        """Test Drazin criteria on nonsingular matrix."""
        A = nonsingular_matrix
        A_D, k = impl_func(A, tol=1e-10)

        results = verify_drazin_criteria(A, A_D, int(k), tol=1e-8)
        assert results[
            "all_pass"
        ], f"{impl_func.__name__} failed criteria ({results['crit1_error']:.2e}, {results['crit2_error']:.2e}, {results['crit3_error']:.2e})\n"


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

        np.random.seed(456384652 + index)  # Reproducible but different for each index

        assert index <= n, "Index cannot exceed matrix dimension"

        # Create Jordan normal form with a nilpotent block of size 'index'
        J_nil = TestSingularMatrices.create_jordan_block(0, index)

        # Fill the remainder with a random nonsingular matrix
        J_other = np.random.randn(n - index, n - index)
        while np.linalg.cond(J_other) > 1e6:
            J_other = np.random.randn(n - index, n - index)

        # Combine into full Jordan form
        J = np.zeros((n, n))
        J[:index, :index] = J_nil
        J[index:, index:] = J_other

        # Apply random similarity transformation
        P = np.random.randn(n, n)

        # Ensure P is VERY well conditioned
        while np.linalg.cond(P) > 1e3:
            P = np.random.randn(n, n)

        # A = P @ J @ P^{-1}
        A = P @ J @ np.linalg.inv(P)

        return A

    @pytest.mark.parametrize("impl_func", implementations)
    @pytest.mark.parametrize("index", [1, 2, 5, 10, 25])
    def test_singular_index_detection(self, impl_func, index):
        """Test that drazin_ord9 correctly detects algebraic index."""
        n = 51
        A = self.create_singular_matrix_with_index(n, index)

        A_D, k_computed = impl_func(A, tol=1e-5)

        assert (
            int(k_computed) == index
        ), f"Expected index {index}, but got {int(k_computed)}"

    @pytest.mark.parametrize("impl_func", implementations)
    @pytest.mark.parametrize("index", [1, 2, 5, 10, 25])
    def test_singular_criteria_all_indices(self, impl_func, index):
        """Test Drazin criteria on singular matrices with various indices."""
        n = 51
        A = self.create_singular_matrix_with_index(n, index)

        A_D, k = impl_func(A, tol=1e-5)

        results = verify_drazin_criteria(A, A_D, int(k), tol=1e-5)
        assert results[
            "all_pass"
        ], f"{impl_func.__name__} failed criteria ({results['crit1_error']:.2e}, {results['crit2_error']:.2e}, {results['crit3_error']:.2e})\n"


if __name__ == "__main__":
    # pytest.main([__file__, "-v", "-k", "TestNonsingularMatrix"])
    # pytest.main([__file__, "-v", "-k", "TestExampleMatrix"])
    # pytest.main(
    #     [__file__ + "::TestSingularMatrices::test_singular_index_detection", "-v"]
    # )
    pytest.main([__file__, "-v"])
