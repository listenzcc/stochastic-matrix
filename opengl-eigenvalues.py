"""
File: opengl-eigenvalues.py
Author: Chuncheng Zhang
Date: 2023-10-11
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


# %% ---- 2023-10-11 ------------------------
# Requirements and constants
import time
import numpy as np
import OpenGL.arrays.vbo as glvbo

from OpenGL import GL as gl
from OpenGL import GLUT as glut

from eigenvalues import generate_random_matrix, simulate_eigenvalues, PerlinNoiseMatrix
from util.tools import timing_decorator

pnm = PerlinNoiseMatrix(n=6)
mat = pnm.random_matrix()
# mat = generate_random_matrix()
data = simulate_eigenvalues(mat)

# %% ---- 2023-10-11 ------------------------
# Function and class


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

    gl.glPointSize(3.0)

    vbo_pos = glvbo.VBO(attrs['pos'])
    vbo_pos.bind()
    gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
    gl.glVertexPointer(2, gl.GL_FLOAT, 0, vbo_pos)

    vbo_rgba = glvbo.VBO(attrs['rgba'])
    vbo_rgba.bind()
    gl.glEnableClientState(gl.GL_COLOR_ARRAY)
    gl.glColorPointer(4, gl.GL_FLOAT, 0, vbo_rgba)

    gl.glDrawArrays(gl.GL_POINTS, 0, attrs['n'])

    # gl.glFlush()
    glut.glutSwapBuffers()


def display():
    while True:
        redraw()
        time.sleep(0.01)


# %% ---- 2023-10-11 ------------------------
# Play ground
if __name__ == '__main__':
    glut.glutInit()

    glut.glutInitDisplayMode(glut.GLUT_SINGLE | glut.GLUT_RGBA)

    glut.glutInitWindowSize(400, 400)
    glut.glutInitWindowPosition(500, 300)

    glut.glutCreateWindow("Wnd-1")
    glut.glutDisplayFunc(display)

    glut.glutMainLoop()


# %% ---- 2023-10-11 ------------------------
# Pending


# %% ---- 2023-10-11 ------------------------
# Pending
