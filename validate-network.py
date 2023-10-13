"""
File: validate-network.py
Author: Chuncheng Zhang
Date: 2023-10-13
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


# %% ---- 2023-10-13 ------------------------
# Requirements and constants
import cv2
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from rich import print
from deep_network import ImageTransformer, device
from eigenvalues import generate_random_matrix, simulate_eigenvalues, PerlinNoiseMatrix

# %% ---- 2023-10-13 ------------------------
# Function and class


# %% ---- 2023-10-13 ------------------------
# Play ground

input_dim = 2048  # Dimension of the encoded image
output_dim = 3  # Number of channels in the output image
model = ImageTransformer(input_dim, output_dim)

# Restore the model and optimizer
checkpoint = torch.load('checkpoint/checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

epoch = checkpoint['epoch']
loss = checkpoint['loss']

# print(model, epoch, loss)

# %% ---- 2023-10-13 ------------------------
# Pending

pnm = PerlinNoiseMatrix(n=6)

for _ in range(200):

    mat = pnm.random_matrix()
    # mat = generate_random_matrix(n=6)
    data = simulate_eigenvalues(mat)

    x = np.zeros((2, 3, 6, 6))
    x[0][0] = mat.real
    x[0][1] = mat.imag

    x = torch.from_numpy(x).type(torch.FloatTensor)

    y = model(x.to(device))
    print(y)

    img = y.detach().cpu().numpy()
    print(img.shape)
    img /= np.max(img)
    mat = (img[0] * 255).astype(np.uint8).transpose([2, 1, 0])
    mat = cv2.resize(mat, (400, 400))

    cv2.imshow('main', mat)
    cv2.waitKey(1)

# %% ---- 2023-10-13 ------------------------
# Pending

# %%
