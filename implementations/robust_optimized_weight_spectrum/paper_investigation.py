import copy
import logging
import os
from typing import Tuple

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt

from helper_methods.signal_processing_methods import square_envelope_spectrum
from implementations.robust_optimized_weight_spectrum.paper_utils import (
    LogisticRegression,
    RegType,
)


def load_signal(signal_path: str, data_channel: int = 0):
    # Get the signals
    data_matrix = np.loadtxt(signal_path)

    # Return the required signal
    return data_matrix[:, data_channel].flatten()

def get_ses_signal(
    x_signal: np.ndarray, Fs: int
) -> Tuple[np.ndarray, np.ndarray]:

    # Get the signal length
    N = len(x_signal)

    # Get SES (one half)
    freq = np.fft.fftfreq(N, 1 / Fs)[: N // 2]
    X = 2 * square_envelope_spectrum(x_signal)[: N // 2]

    X[0] = 0

    return freq, X

def get_training_data(
        x: np.ndarray,
        truncation_freq: int,
        Lw: int,
        Ls: int,
        Fs: int,
        label_class: int
    ) -> Tuple[np.ndarray, np.ndarray]:

    # Flatten signals
    x_flat = copy.deepcopy(x).flatten()

    # Perform hankelisation
    Lt = len(x_flat)
    Nw = (Lt - Lw) // Ls + 1

    window_list = []

    for idx in range(Nw):
        # Get windowed signal segment
        x_segment = x_flat[int(idx * Ls) : int(idx * Ls) + Lw]

        # Compute SES
        freq, x_ses = get_ses_signal(x_segment, Fs=Fs)

        # Store
        window_list.append(x_ses)

    # Create matrix
    Xf_array = np.array(window_list)

    # Truncate matrix
    trunc_index = np.argmin(np.abs(freq - truncation_freq))

    Xf_array = Xf_array[:,  :trunc_index]
    freq = freq[:trunc_index]

    # Create labels
    Yf = np.ones(Nw) * label_class

    return Xf_array, Yf, freq


if __name__ == "__main__":
    # Constants
    base_dir = "../../Datasets/IMS/dataset_two/"
    save_dir = "./results/IMS/"
    save_path = os.path.join(save_dir, "OSES_results.npy")

    Fs = 20480
    file_list = sorted(os.listdir(base_dir))[:970]

    # Useful training parameters
    trunc_freq = 5 * 300
    Lw = 5333
    Lsft = 400
    num_healthy = 30
    iter_range = range(num_healthy, 900, 1) # len(file_list)

    try:
        OSES_results = np.load(save_path, allow_pickle=True).item()
        freq = OSES_results["freq"]
        iteration_list = OSES_results["iteration_list"]
        loss_values = OSES_results["loss_values"]
        OSES_matrix = OSES_results["OSES_matrix"]

    except:
        logging.info("Could not load data. Starting training loop...")
        os.makedirs(save_dir, exist_ok=True)

        Xh_list = []
        Yh_list = []

        for i in range(num_healthy):
            # Get signal
            x_signal_i = load_signal(os.path.join(base_dir, file_list[i]))

            # Get matrices
            Xhi, Yhi, freq = get_training_data(
                x_signal_i,
                truncation_freq=trunc_freq,
                Lw=Lw,
                Ls=Lsft,
                Fs=Fs,
                label_class=0
            )
            Xh_list.append(Xhi)
            Yh_list.append(Yhi)

        # Stack
        Xh = np.vstack(Xh_list)
        Yh = np.concatenate(Yh_list)

        # Define storage lists
        OSES_list = []
        iteration_list = []
        loss_values = []

        # Define initial coefficients
        init_coeff = None

        # Perform iterations
        for i in iter_range:
            print(f"Working on file {i}...")

            # Get the test signal
            x_test_i = load_signal(os.path.join(base_dir, file_list[i]))

            # Construct the data and label matrices
            Xt, Yt, _ = get_training_data(
                x_test_i,
                truncation_freq=trunc_freq,
                Lw=Lw,
                Ls=Lsft,
                Fs=Fs,
                label_class=1
            )

            # Combine the data to get the true training data
            X = np.vstack([Xh, Xt])
            y = np.concatenate([Yh, Yt])

            # Define the model
            model = LogisticRegression(
                learning_rate=0.5,
                max_iter=4000,
                regulariser_type=RegType.L2,
                alpha=0.5,
                optimiser="gradient_descent",  # "newton-cg"
                initial_coeffs=init_coeff,
            )

            # Fit the data
            model.fit(X, y)

            # Store
            iteration_list.append(model.get_iteration_steps())
            loss_values.append(model.get_loss_values())
            OSES_list.append(model.get_coefficients().copy())

            # Update coefficients
            init_coeff = model.get_coefficients().copy()

        # Create result dict
        OSES_matrix = np.array(OSES_list)

        OSES_results = {
            "freq": freq,
            "iteration_list": iteration_list,
            "loss_values": loss_values,
            "OSES_matrix": OSES_matrix
        }

        np.save(save_path, OSES_results)

    finally:
        # logging.info(f"OSES_matrix shape: {np.shape(OSES_results["OSES_matrix"])}")
        # Remove zero values
        OSES_matrix = OSES_matrix[:, :-1]
        print(OSES_matrix.shape)

        # Define the maximum number of signals
        sig_max = 900
        # OSES_matrix = OSES_matrix[:sig_max, :]

        # Drop the list dimension of Z (constants)
        X, Y = np.meshgrid(freq, iter_range)


        # Visualise!
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"), figsize=(12, 12))
        log_norm = mcolors.LogNorm(vmin=OSES_matrix.min(), vmax=OSES_matrix.max())

        ax.plot_surface(
            X, Y, OSES_matrix, cmap=plt.cm.jet, rstride=10, cstride=1, norm=log_norm
        )

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Sample index")
        ax.set_zlabel("OSES")

        plt.show()
        fig, ax = plt.subplots()
        ax.plot(iter_range, iteration_list)
        plt.show()

        print(OSES_matrix.shape)

        plt.figure()
        plt.plot(freq, OSES_matrix[540, :], label = "540")
        plt.plot(freq, OSES_matrix[700, :], label = "700")
        plt.plot(freq, OSES_matrix[800, :], label = "800")
        plt.legend()
        plt.show()
