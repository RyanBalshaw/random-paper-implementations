import os

import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import make_classification, make_blobs, make_circles

from implementations.robust_optimized_weight_spectrum.paper_utils import (
    LogisticRegression,
    RegType,
)
from helper_methods.signal_processing_methods import square_envelope_spectrum, normalised_square_envelope_spectrum, fourier_spectrum, normalised_fourier_spectrum

def simple_problem():
    X, y = make_classification(n_features = 2, n_informative = 2, n_redundant=0)
    # X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=0.5)
    # X, y = make_circles(n_samples=1000,noise=0.01, random_state=0)

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

    model = LogisticRegression(
        learning_rate=0.1,
        max_iter=400,
        regulariser_type=RegType.L1,
        alpha=0.001,
        optimiser="gradient_descent",  #
        tol=1e-4
    )
    model.fit(X, y)

    plt.figure()
    plt.plot(model._loss_values)
    plt.show()

    plt.figure()
    plt.plot(np.arange(len(model._zeta[:, 0])), model._zeta[:, 0], lw=0.4)
    plt.show()

    X_grid, Y_grid = np.meshgrid(
        np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
        np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    )
    X_test = np.hstack([X_grid.ravel().reshape(-1, 1), Y_grid.ravel().reshape(-1, 1)])

    D = model.predict_proba(X_test)

    plt.figure()
    plt.contourf(X_grid, Y_grid, D.reshape(100, 100), cmap=plt.cm.jet)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

    gradient_error, hessian_error = model.check_gradients_and_hessian(X, y)

def signal_problem():
    base_dir = "../../Datasets/IMS/dataset_two/"
    file_list = sorted(os.listdir(base_dir))

    f1 = np.loadtxt(os.path.join(base_dir, file_list[200]))
    f2 = np.loadtxt(os.path.join(base_dir, file_list[700]))

    N = f1.shape[0]
    Fs = 20480
    X1 = 2 * square_envelope_spectrum(f1[:, 0])[: N // 2]
    X2 = 2 * square_envelope_spectrum(f2[:, 0])[: N // 2]
    freq = np.fft.fftfreq(N, 1 / Fs)[: N // 2]
    X = np.vstack([X1.reshape(1, -1), X2.reshape(1, -1)])
    y = np.array([0, 1])

    print(X.shape, y.shape)

    plt.figure()
    plt.plot(freq, X2)
    plt.plot(freq, X1)
    plt.show()

    model = LogisticRegression(
        learning_rate=10,
        max_iter=1000,
        regulariser_type=RegType.L1,
        alpha=0.00001,
        optimiser="gradient_descent",  # "newton-cg"
        tol=1e-4
    )
    model.fit(X, y)

    plt.figure()
    plt.plot(model._loss_values)
    plt.show()

    min_pos = np.argmin(np.abs(freq - 1000))

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(freq, model._zeta[:-1, 0], lw=0.4)
    ax[1].plot(freq[:min_pos], model._zeta[:min_pos, 0], lw=0.4)
    plt.show()

    # gradient_error, hessian_error = model.check_gradients_and_hessian(X, y)


if __name__ == "__main__":
    # check_estimator(LogisticRegression())

    # simple_problem()
    signal_problem()










