"""
File: eigenvalues.py
Author: Chuncheng Zhang
Date: 2023-10-10
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Amazing things

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2023-10-10 ------------------------
# Requirements and constants
import opensimplex
import numpy as np
import pandas as pd

from util.tools import timing_decorator

# %% ---- 2023-10-10 ------------------------
# Function and class


class PerlinNoiseMatrix(object):
    """
    Generates a random complex matrix using Perlin noise.
    This class initializes a random seed and timestep. 

    The random_matrix() method generates a new complex matrix each call by sampling 
    2D Perlin noise for the real and imaginary values. The timestep increments each 
    call to generate new noise values.

    The matrix size is set at initialization.

    Attributes:
    seed (int): Random seed for Perlin noise generator
    t (float): Current timestep 
    t_step (float): Amount to increment timestep each call

    Methods:
    __init__(n): Initialize class with matrix size n
    random_matrix(): Generate new random complex matrix

    """

    seed = np.random.randint(0, 65536)
    t = 0
    t_step = 0.05

    def __init__(self, n: int = 5):
        opensimplex.seed(self.seed)
        self.n = n
        self.n2 = n**2

    def random_matrix(self):
        """
        Generates a new random complex matrix using Perlin noise.

        Increments the timestep and samples 2D Perlin noise to generate the real and 
        imaginary values for a new random complex matrix.

        The matrix dimensions are set during class initialization.

        Returns:
            np.array: The new randomly generated complex matrix.

        """

        self.t += self.t_step

        t = self.t
        n = self.n
        n2 = self.n2

        real = np.array([opensimplex.noise2(j, t)
                        for j in range(n2)]).reshape((n, n))

        imag = np.array([opensimplex.noise2(1000 + j, t)
                        for j in range(n2)]).reshape((n, n))

        mat = np.zeros((n, n), dtype=np.complex64)
        mat.real = real
        mat.imag = imag

        return mat


def generate_random_matrix(n: int = 5, predefined=False) -> np.array:
    """
    Generates a random complex matrix of shape (n, n).

    The real and imaginary parts are initialized with random values from a normal distribution.

    Args:
    n (int): The size of the square matrix to generate.

    Kwargs:
    predefined (bool): If True, initialize the matrix with predefined values instead of random ones.

    Returns:
    np.array: The generated random complex matrix of shape (n, n).
    """

    mat = np.zeros((n, n), dtype=np.complex64)

    if predefined:
        mat.real = [
            [0, -1, 1, 0, 0],
            [0, -1, -1, 0, 1],
            [1, 0, 1, -1, -1],
            [1, -1, 0, 0, 0],
            [0, 0, -1, 0, 0]
        ]

        mat.imag = [
            [1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0],
        ]

        return mat

    mat.real = np.random.randn(n, n)
    mat.imag = np.random.randn(n, n)

    return mat


@timing_decorator
def simulate_eigenvalues(mat, grids: int = 100, tmin: float = -10, tmax: float = 10) -> pd.DataFrame:
    """
    Simulates the eigenvalues of a complex matrix by sweeping two values.

    The function sweeps t1 and t2 across a range of values, updating two values in the input matrix for each combination. 
    It calculates the eigenvalues of the updated matrix and stores the results in a buffer.

    The buffer is converted to a Pandas DataFrame and returned.

    Args:
    mat (np.array): The complex matrix to simulate.
    grids (int): The number of values to sweep t1 and t2 over.
    tmin (float): The minimum value to sweep t1 and t2 over. 
    tmax (float): The maximum value to sweep t1 and t2 over.

    Returns:
    pd.DataFrame: A dataframe containing the simulated eigenvalue results.

    """

    buffer = []

    for t1 in np.linspace(tmin, tmax, grids):
        for t2 in np.linspace(tmin, tmax, grids):
            mat.real[1][0] = t1
            mat.real[2][1] = t2
            eig = np.linalg.eig(mat)

            buffer.extend(
                dict(real=e.real, imag=e.imag, t1=t1, t2=t2, order=i)
                for i, e in enumerate(eig[0])
            )

    data = pd.DataFrame(buffer)

    for xy, col in zip(['x', 'y'], ['real', 'imag']):
        _max = data[col].max()
        _min = data[col].min()
        data[xy] = (data[col] - _min) / (_max - _min) * 2 - 1

    data['r'] = (data['t1'] - tmin) / (tmax - tmin)
    data['g'] = 0
    data['b'] = (data['t2'] - tmin) / (tmax - tmin)
    data['a'] = 0.1

    return data


# %% ---- 2023-10-10 ------------------------
# Play ground
if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt

    from rich import print

    pnm = PerlinNoiseMatrix()
    # mat = pnm.random_matrix()
    mat = generate_random_matrix()
    data = simulate_eigenvalues(mat)

    print(mat)
    print(data)

    sns.scatterplot(data, x='real', y='imag',
                    alpha=0.1,
                    size=1,
                    hue='order',
                    edgecolor=None,
                    marker=',')
    plt.tight_layout()
    plt.show()

# %% ---- 2023-10-10 ------------------------
# Pending

# %% ---- 2023-10-10 ------------------------
# Pending

# %%

# %%

# %%
