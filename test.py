import pytest
import numpy as np

from stochastic_matrix_demo import rnd_stochastic_matrix, compute_eig

# Tests for rnd_stochastic_matrix

# Happy path tests


@pytest.mark.parametrize("rows, cols", [
    (2, 2),
    (5, 5),
    (10, 10)
])
def test_rnd_stochastic_matrix_happy_path(rows, cols):

    # Act
    mat = rnd_stochastic_matrix(rows, cols)

    # Assert
    assert mat.shape == (rows, cols)
    assert np.allclose(np.sum(mat, axis=1), np.ones(rows))

# Edge case tests


@pytest.mark.parametrize("rows, cols", [
    (0, 0),
    (1, 1)
])
def test_rnd_stochastic_matrix_edge_cases(rows, cols):

    # Act
    mat = rnd_stochastic_matrix(rows, cols)

    # Assert
    assert mat.shape == (rows, cols)
    assert np.allclose(np.sum(mat, axis=1), np.ones(rows))

# Error case tests


@pytest.mark.parametrize("rows, cols", [
    (-1, 5),
    (5, -1),
    (5, "a")
], ids=["Negative rows", "Negative cols", "Invalid cols type"])
def test_rnd_stochastic_matrix_error_cases(rows, cols):

    # Assert
    with pytest.raises(Exception):
        rnd_stochastic_matrix(rows, cols)


# Tests for compute_eig

# Happy path tests
@pytest.mark.parametrize("mat", [
    np.random.rand(2, 2),
    np.random.rand(5, 5),
    np.random.rand(10, 10)
])
def test_compute_eig_happy_path(mat):

    # Act
    eigenvalues, eigenvectors = compute_eig(mat)

    # Assert
    assert eigenvalues.shape == (mat.shape[0],)
    assert eigenvectors.shape == mat.shape

# Edge case test


def test_compute_eig_edge_case():

    # Arrange
    mat = np.zeros((1, 1))

    # Act
    eigenvalues, eigenvectors = compute_eig(mat)

    # Assert
    assert eigenvalues.shape == (1,)
    assert eigenvectors.shape == (1, 1)

# Error case test


@pytest.mark.parametrize("mat", [
    np.random.rand(2, 3),
    "not a matrix"
], ids=["Matrix not square", "Invalid matrix type"])
def test_compute_eig_error_cases(mat):

    # Assert
    with pytest.raises(Exception):
        compute_eig(mat)
