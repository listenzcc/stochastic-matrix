"""
File: opengl-training.py
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
import numpy as np
import OpenGL.arrays.vbo as glvbo

import torch
import torch.nn as nn

from OpenGL import GL as gl
from OpenGL import GLUT as glut

from deep_network import ImageTransformer, device
from eigenvalues import generate_random_matrix, simulate_eigenvalues, PerlinNoiseMatrix
from util.tools import timing_decorator

pnm = PerlinNoiseMatrix(n=6)
mat = pnm.random_matrix()
# mat = generate_random_matrix(n=6)
data = simulate_eigenvalues(mat)

# %% ---- 2023-10-13 ------------------------
# Function and class
input_dim = 2048  # Dimension of the encoded image
output_dim = 3  # Number of channels in the output image
model = ImageTransformer(input_dim, output_dim).to(device)

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(x, y, model=model):
    optimizer.zero_grad()
    p = model(x)
    loss = criterion(p, y)
    loss.backward()
    optimizer.step()
    loss_val = loss.item()
    print(f'Loss is {loss_val}')
    return loss_val


class TrainingBuffer():
    buffer = []
    loss = []
    epoch = 0

    def __init__(self):
        pass

    def len(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = []
        self.epoch += 1

    def get_x(self):
        return np.array([e[0] for e in self.buffer])

    def get_y(self):
        return np.array([e[1] for e in self.buffer])


training_buffer = TrainingBuffer()


@timing_decorator
def redraw():
    """
    Redraws the eigenvalue simulation visualization.

    Generates a new random complex matrix, simulates eigenvalues, 
    and renders the results as a point cloud using OpenGL.

    The vertex buffer objects are updated with the new simulated data
    and the display is redrawn.

    """

    # mat = generate_random_matrix()
    mat = pnm.random_matrix()
    data = simulate_eigenvalues(mat)

    attrs = dict(
        pos=data[['x', 'y']].to_numpy(np.float32).ravel(),
        rgba=data[['r', 'g', 'b', 'a']].to_numpy(np.float32).ravel(),
        n=len(data)
    )

    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glEnable(gl.GL_BLEND)

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    gl.glPointSize(1.0)

    vbo_pos = glvbo.VBO(attrs['pos'])
    vbo_pos.bind()
    gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
    gl.glVertexPointer(2, gl.GL_FLOAT, 0, vbo_pos)

    vbo_rgba = glvbo.VBO(attrs['rgba'])
    vbo_rgba.bind()
    gl.glEnableClientState(gl.GL_COLOR_ARRAY)
    gl.glColorPointer(4, gl.GL_FLOAT, 0, vbo_rgba)

    gl.glDrawArrays(gl.GL_POINTS, 0, attrs['n'])

    glut.glutSwapBuffers()
    # gl.glFlush()

    oo = gl.glReadPixels(0, 0, 100, 100, gl.GL_RGB, gl.GL_FLOAT)
    # oo = cv2.resize(o, (100, 100))
    # print(o.shape, np.max(o), oo.shape, np.max(oo))

    x = np.zeros((3, 6, 6))
    x[0] = mat.real
    x[1] = mat.imag
    y = oo.transpose([2, 1, 0])

    training_buffer.buffer.append((x, y))

    if training_buffer.len() > 10:
        x = np.array(training_buffer.get_x())
        y = np.array(training_buffer.get_y())
        training_buffer.clear()

        x = torch.from_numpy(x).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.FloatTensor)
        # print(x.shape, y.shape)

        training_buffer.loss.append(train(x.to(device), y.to(device)))

        if training_buffer.epoch % 100 == 0:
            epoch = training_buffer.epoch
            torch.save(
                dict(
                    epoch=epoch,
                    model_state_dict=model.state_dict(),
                    loss=training_buffer.loss,
                ),
                'checkpoint/checkpoint.pth',
            )


def display():
    redraw()
    # while True:
    #     redraw()
    #     time.sleep(0.01)


# %% ---- 2023-10-13 ------------------------
# Play ground
if __name__ == '__main__':
    glut.glutInit()

    glut.glutInitDisplayMode(glut.GLUT_SINGLE | glut.GLUT_RGBA)

    glut.glutInitWindowSize(100, 100)
    glut.glutInitWindowPosition(500, 300)

    glut.glutCreateWindow("Wnd-1")
    glut.glutDisplayFunc(display)
    glut.glutIdleFunc(display)

    glut.glutMainLoop()


# %% ---- 2023-10-13 ------------------------
# Pending


# %% ---- 2023-10-13 ------------------------
# Pending
