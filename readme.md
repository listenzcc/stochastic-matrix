# 转移矩阵-3

由于这两天的东西比较水，所以写两个事情。

首先，前文中猜想的那个命题根本没有道理，这类矩阵也没有那么复杂，它就是典型的“右随机矩阵”。这类矩阵具有良好的性质，它最大特征值为实数 1，且对应的特征向量为全 1 向量，另外，其他特征值的模总小于 1，这导致它的连乘总收敛于全 1 向量。这种良好的性质是我之前产生误会的根源。这部分的随机值样例可见开源代码

[Right Stochastic Matrix](https://observablehq.com/@listenzcc/right-stochastic-matrix)

其次是除了说明这种现象之外，本文还尝试使用 vscode 中的 sourcery 插件进行代码分析，它可以自动理解代码、生成高质量的文档和测试用例。本文附录部分的全部内容都是由它自动生成的。我只是调整了一下格式。

[Sourcery | Automatically Improve Code Quality](https://sourcery.ai/)

也因为这个原因，本文的内容以英文为主，因为在 vim 模式中切换输入法实在是很不方便。本文的详细代码可见我的 github 仓库

[https://github.com/listenzcc/stochastic-matrix](https://github.com/listenzcc/stochastic-matrix)

---

## Math

The stochastic matrix is $A \in R^{n \times n}$

$\begin{cases} \sum_{j=1}^{n} A_{ij} &= 1\\ A_{ij} &> 0 \end{cases}$

The definition of the stochastic matrix and its properties reads in

[Stochastic matrix](https://en.wikipedia.org/wiki/Stochastic_matrix)

The stochastic matrix in the code is the right version, since

> A right stochastic matrix is a real square matrix, with each row summing to 1.
> 

Define all ones vector as the following.

$E^{n \times 1} = [\underbrace{1, 1, \dots, 1}_n]^T$

In the very long run, it follows.

$E = A \cdot E = A^N \cdot E = \lim_{N \rightarrow \infty} A^N \cdot E$

As a result, the stochastic matrix has eigenvalues.

$\lVert \lambda_i \lVert \le 1$

And the largest eigenvalue equals to $1$, and the eigenvector of the eigenvalue is $E$.
The other eigenvalues and eigenvectors follow the pattens, as below.

![demo.png](%E8%BD%AC%E7%A7%BB%E7%9F%A9%E9%98%B5-3%2081537600a4fd433cbf8e861b6e3e4335/demo.png)

The values and vectors are complex other than real numbers,
so, the inequality changes into, which is also satisfied.

$x^{*T} x \ge x^{*T} \lambda^2 x = x^{*T} A^T A x$

In which, the $x^*$ refers to the conjugate of the $x$.

$\blacksquare$

It converges as the following.

![converge.png](%E8%BD%AC%E7%A7%BB%E7%9F%A9%E9%98%B5-3%2081537600a4fd433cbf8e861b6e3e4335/converge.png)

![converge-2.png](%E8%BD%AC%E7%A7%BB%E7%9F%A9%E9%98%B5-3%2081537600a4fd433cbf8e861b6e3e4335/converge-2.png)

## Appendix: Code description (Auto generated by sourcery)

- rnd_stochastic_matrix(rows: int = 6, cols: int = 6) -> np.array:
Generates a random stochastic matrix with the given number of rows and columns.
A stochastic matrix is a square matrix with nonnegative real entries where each row sums to 1.
This function generates a random matrix and normalizes it to be stochastic.
- compute_eig(mat: np.array):
Computes the eigenvalues and eigenvectors of a matrix.
This function calculates and returns the eigenvalues and eigenvectors
of the input matrix using numpy's linalg.eig function.
- The [test.py](https://www.notion.so/test.py) script is the test instances for the functions.

```powershell
pytest test.py

# ==== test session starts ==================================
# platform win32 -- Python 3.11.4, pytest-7.4.2, pluggy-1.3.0
# rootdir: C:\\Users\\zcc\\Desktop\\eigen
# plugins: anyio-3.5.0
# collected 14 items
#
# test.py ..............
# ==== 14 passed, xx warnings in 1.48s ======================

```

## Appendix: Code explanation (Auto generated by sourcery)

What?
This Python code generates a random stochastic matrix, computes its eigenvalues and eigenvectors, and visualizes the results.

How?
rnd_stochastic_matrix generates a random matrix and normalizes it. compute_eig calculates the eigendecomposition using NumPy.

The **main** section provides a demo - generating a matrix, computing eigenvectors/values, and visualizing them with Pandas/Seaborn.

Coupling and Cohesion
The two functions are loosely coupled. The overall module has high cohesion related to stochastic matrices.

Single Responsibility Principle
Both functions follow SRP. rnd_stochastic_matrix handles generation, compute_eig handles decomposition. No extraction needed.

Unusual Things
None

Highly Suspicious
No input validation in compute_eig could lead to errors.