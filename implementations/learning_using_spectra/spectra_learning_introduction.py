"""
https://alexshtf.github.io/2025/12/16/Spectrum.html
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla


def univariate_full_spectral(A, B, xs):
    mats = A + B * xs[..., np.newaxis, np.newaxis]

    # compute the k-th eigenvalues of each matrix
    return np.linalg.eigvalsh(mats)


def univariate_spectral(A, B, k, xs):
    # support negative eigenvalue indices:
    k = k % A.shape[0]

    # create a batch of matrices, one for each entry in xs
    eigenvalue_matrix = univariate_full_spectral(A, B, xs)

    # compute the k-th eigenvalue of each matrix
    return eigenvalue_matrix[:, k]


def bivariate_full_spectral(A, B, C, xs, ys):
    mats = A + B * xs[..., np.newaxis, np.newaxis] + C * ys[..., np.newaxis, np.newaxis]

    # compute the k-th eigenvalues of each matrix
    return np.linalg.eigvalsh(mats)


def bivariate_spectral(A, B, C, k, xs, ys):
    # support negative eigenvalue indices:
    k = k % A.shape[0]

    # create a batch of matrices, one for each entry in xs
    eigenvalue_matrix = bivariate_full_spectral(A, B, C, xs, ys)

    # compute the k-th eigenvalue of each matrix
    return eigenvalue_matrix[:, :, k]


def plot_univariate_eigenfunctions(A, B, n_rows, n_cols):
    """Plots y(x) = λₖ(A + B * x) on a grid layout"""
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(3 * n_cols, 2 * n_rows), layout="constrained"
    )
    plot_xs = np.linspace(-5, 5, 1000)
    plot_ys_mat = univariate_full_spectral(A, B, plot_xs)
    for k, ax in enumerate(axs.ravel()):
        ax.plot(plot_xs, plot_ys_mat[:, k])
        ax.set_title(f"$\\lambda_{k}(A + B * x)$")
    plt.show()


def plot_bivariate_eigenfunctions_2d(A, B, C, n_rows, n_cols):
    """Plots z(x, y) = λₖ(A + B * x + C * y) on a grid layout"""
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        subplot_kw={"projection": "3d"},
        layout="constrained",
    )

    plot_xs = np.linspace(-5, 5, 50)
    plot_ys = np.linspace(-6, 6, 50)
    grid_xs, grid_ys = np.meshgrid(plot_xs, plot_ys)
    plot_zs_mat = bivariate_full_spectral(A, B, C, grid_xs, grid_ys)

    k = A.shape[0]
    for k, ax in enumerate(axs.ravel()[:k]):
        ax.plot_surface(grid_xs, grid_ys, plot_zs_mat[:, :, k], cmap="viridis")
        ax.set_title(f"$\\lambda_{1 + k}(A + B * x + C * y)$")
    plt.show()


def make_psd(B):
    eigvals, eigvecs = sla.eigh(B)
    eigvals_pos = np.maximum(0, eigvals)
    return eigvecs @ np.diag(eigvals_pos) @ eigvecs.T


if __name__ == "__main__":
    # Example one -> 3x3 matrices
    np.random.seed(42)
    A = np.random.randn(3, 3)
    B = np.random.randn(3, 3)
    plot_univariate_eigenfunctions(A, B, 1, 3)

    # Example two -> 9x9 matrices
    np.random.seed(42)
    A = np.random.randn(9, 9)
    B = np.random.randn(9, 9)
    plot_univariate_eigenfunctions(A, B, 3, 3)

    # Example three -> positive semi-definite B
    np.random.seed(42)
    A = np.random.randn(9, 9)
    B = make_psd(np.random.randn(9, 9))
    plot_univariate_eigenfunctions(A, B, 3, 3)

    # Example four -> negative semi-definite B
    np.random.seed(42)
    A = np.random.randn(9, 9)
    B = -1.0 * make_psd(np.random.randn(9, 9))
    plot_univariate_eigenfunctions(A, B, 3, 3)

    # Example five -> indefinite B
    A = np.array([[1, np.sqrt(2)], [np.sqrt(2), 2]])
    B = np.array([[-1, 0], [0, 2]])
    plot_univariate_eigenfunctions(A, B, 1, 2)

    # Turns out eigenvalue functions are monotone -
    # if we take a matrix A and add the matrix xB whose eigenvalues are all
    # non-negative, the entire spectrum of eigenvalues increases. So larger x results
    # in larger eigenvalues, and vice versa, and we obtain an increasing function.
    # The opposite happens when all eigenvalues of B are non-positive.

    # Example six -> 2D surface with 3x3 matrices
    np.random.seed(42)
    A = np.random.randn(3, 3)
    B = np.random.randn(3, 3)
    C = np.random.randn(3, 3)
    plot_bivariate_eigenfunctions_2d(A, B, C, 1, 3)

    # Example seven -> 2D surface with 9x9 matrices
    np.random.seed(42)
    A = np.random.randn(9, 9)
    B = np.random.randn(9, 9)
    C = np.random.randn(9, 9)
    plot_bivariate_eigenfunctions_2d(A, B, C, 3, 3)
