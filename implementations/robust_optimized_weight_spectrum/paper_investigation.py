import os

import numpy as np
from matplotlib import pyplot as plt

from helper_methods.signal_processing_methods import square_envelope_spectrum
from implementations.robust_optimized_weight_spectrum.paper_utils import (
    LogisticRegression,
    RegType,
)


def signal_problem():
    base_dir = "../../Datasets/IMS/dataset_two/"
    file_list = sorted(os.listdir(base_dir))

    f1 = np.loadtxt(os.path.join(base_dir, file_list[710]))
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
        tol=1e-4,
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


if __name__ == "__main__":
    signal_problem()
