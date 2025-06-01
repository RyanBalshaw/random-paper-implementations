import os
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from helper_methods.signal_processing_methods import square_envelope_spectrum
from implementations.robust_optimized_weight_spectrum.paper_utils import (
    LogisticRegression,
    RegType,
)


def load_ses_signal(
    signal_path: str, Fs: int, data_channel: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    # Get the signals
    data_matrix = np.loadtxt(signal_path)

    N = data_matrix.shape[0]

    # Get SES (one half)
    freq = np.fft.fftfreq(N, 1 / Fs)[: N // 2]
    X = 2 * square_envelope_spectrum(data_matrix[:, data_channel])[: N // 2]

    return freq, X


if __name__ == "__main__":
    # Constants
    base_dir = "../../Datasets/IMS/dataset_two/"
    Fs = 20480
    file_list = sorted(os.listdir(base_dir))

    # Define the healthy signal
    freq, X_healthy = load_ses_signal(os.path.join(base_dir, file_list[0]), Fs)

    # Define storage lists
    OSES_list = []
    iteration_list = []

    # Define initial coefficients
    init_coeff = None
    iter_range = range(1, len(file_list), 1)

    # Perform iterations
    for i in iter_range:
        print(f"Working on file {i}...")

        # Get the test signal
        _, X_test = load_ses_signal(os.path.join(base_dir, file_list[i]), Fs)

        # Construct the data and label matrices
        X = np.vstack([X_healthy.reshape(1, -1), X_test.reshape(1, -1)])
        y = np.array([0, 1])

        # Define the model
        model = LogisticRegression(
            learning_rate=10,
            max_iter=1000,
            regulariser_type=RegType.L1,
            alpha=0.00001,
            optimiser="gradient_descent",  # "newton-cg"
            initial_coeffs=init_coeff,
        )

        # Fit the data
        model.fit(X, y)

        # Store
        iteration_list.append(model._iteration_steps)
        OSES_list.append(model.get_coefficients())

    X, Y = np.meshgrid(freq, iter_range)
    Z = np.array(OSES_list)

    # Drop the list dimension of Z (constants)
    Z = Z[:, :-1]

    # Visualise!
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    ax.plot_surface(X, Y, Z, cmap=plt.cm.magma, antialiased=False, shade=False)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(iter_range, iteration_list)
    plt.show()
