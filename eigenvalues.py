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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from rich import print
from tqdm.auto import tqdm


# %% ---- 2023-10-10 ------------------------
# Function and class


# %% ---- 2023-10-10 ------------------------
# Play ground
mat = np.zeros((5, 5), dtype=np.complex64)
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
print(mat)

buffer = []
for t1 in tqdm(np.linspace(-10, 10, 100), 'Progress'):
    for t2 in np.linspace(-10, 10, 100):
        mat.real[1][0] = t1
        mat.real[2][1] = t2
        eig = np.linalg.eig(mat)

        buffer.extend(
            dict(real=e.real, imag=e.imag, t1=t1, t2=t2, size=1, order=i)
            for i, e in enumerate(eig[0])
        )
print(mat)
print(eig)

data = pd.DataFrame(buffer)
data

# %% ---- 2023-10-10 ------------------------
# Pending
sns.scatterplot(data, x='real', y='imag',
                alpha=0.1,
                hue='order',
                edgecolor=None,
                marker='.')
plt.show()

# %% ---- 2023-10-10 ------------------------
# Pending
data['color'] = data['order'].map(str)
fig = px.scatter_3d(data, x='real', y='imag', z='t1',
                    color='color', size='size', size_max=10)
fig.update_traces(marker=dict(line=dict(width=0)))
fig.show()

# %%
