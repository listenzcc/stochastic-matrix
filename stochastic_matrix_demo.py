"""
File: stochastic_matrix_demo.py
Author: Chuncheng Zhang
Date: 2023-10-09
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Demo of stochastic matrix

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2023-10-09 ------------------------
# Requirements and constants
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rich import print
from pathlib import Path
from IPython.display import display

# %% ---- 2023-10-09 ------------------------
# Function and class


def rnd_stochastic_matrix(rows: int = 6, cols: int = 6) -> np.array:
    """
    Generates a random stochastic matrix with the given number of rows and columns.

    A stochastic matrix is a square matrix with nonnegative real entries where each row sums to 1.
    This function generates a random matrix and normalizes it to be stochastic.

    Args:
    rows (int): The number of rows in the matrix.
    cols (int): The number of columns in the matrix.

    Returns: 
    numpy.ndarray: The generated stochastic matrix.
    """

    mat = 1 - np.random.rand(rows, cols)

    d = np.diag(np.sum(mat, axis=1))
    d_inv = np.linalg.inv(d)

    return np.matmul(d_inv, mat)


def compute_eig(mat: np.array):
    """Computes the eigenvalues and eigenvectors of a matrix.

    This function calculates and returns the eigenvalues and eigenvectors 
    of the input matrix using numpy's linalg.eig function.

    Args:
    mat (np.array): The input matrix to compute eigenvalues and eigenvectors for.

    Returns:
    eigenvalues (np.array): The eigenvalues of the input matrix.
    eigenvectors (np.array): The eigenvectors of the input matrix.
    """

    return np.linalg.eig(mat)

# %% ---- 2023-10-09 ------------------------
# Play ground


if __name__ == '__main__':
    plt.style.use('ggplot')

    mat = rnd_stochastic_matrix()
    eig = compute_eig(mat)
    n = mat.shape[0]

    # --------------------------------------------------------------------------------
    # Eigen vectors
    eigenvectors_real = pd.DataFrame(
        eig.eigenvectors.real,
        columns=[['real' for _ in range(n)], list(range(n))]
    )

    eigenvectors_imag = pd.DataFrame(
        eig.eigenvectors.imag,
        columns=[['imag' for _ in range(n)], list(range(n))]
    )

    display(pd.concat([eigenvectors_real, eigenvectors_imag], axis=1))

    # --------------------------------------------------------------------------------
    # Eigen vectors
    eigenvalues_real = eig.eigenvalues.real
    eigenvalues_imag = eig.eigenvalues.imag
    eigenvalues_abs = (eig.eigenvalues.conjugate() * eig.eigenvalues).real

    # --------------------------------------------------------------------------------
    # Display
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    cmap = 'RdBu'

    # The real and imaginary part of the eigenvectors
    ax = axes[0][0]
    ax.set_title('Eigenvectors (real)')
    ax = sns.heatmap(eigenvectors_real, annot=True, ax=ax, cmap=cmap)
    ax.set(xlabel='Eigen vector', ylabel='')
    ax.xaxis.tick_top()

    ax = axes[0][1]
    ax.set_title('Eigenvectors (imag)')
    ax = sns.heatmap(eigenvectors_imag, annot=True, ax=ax, cmap=cmap)
    ax.set(xlabel='Eigen vector', ylabel='')
    ax.xaxis.tick_top()

    # The eigen values
    ax = axes[1][0]
    ax.set_title('Eigenvalues (abs)')
    sns.lineplot(x=range(n), y=eigenvalues_abs, color='gray', ax=ax)
    sns.scatterplot(x=range(n), y=eigenvalues_abs,
                    hue=range(n), palette='deep', ax=ax)

    ax = axes[1][1]
    ax.set_title('Eigenvalues (complex)')
    sns.scatterplot(x=eigenvalues_real,
                    y=eigenvalues_imag,
                    hue=range(n),
                    palette='deep',
                    ax=ax)

    plt.tight_layout()
    plt.show()

    output = Path('img/demo.png')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)


# %% ---- 2023-10-09 ------------------------
# Pending


# %% ---- 2023-10-09 ------------------------
# Pending
