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
from rich import print


# %% ---- 2023-10-09 ------------------------
# Function and class
def rnd_stochastic_matrix(rows=6, cols=6):
    mat = np.random.rand(rows, cols)

    d = np.diag(np.sum(mat, axis=1))
    d_inv = np.linalg.inv(d)

    return np.matmul(d_inv, mat)


# %% ---- 2023-10-09 ------------------------
# Play ground


# %% ---- 2023-10-09 ------------------------
# Pending


# %% ---- 2023-10-09 ------------------------
# Pending
