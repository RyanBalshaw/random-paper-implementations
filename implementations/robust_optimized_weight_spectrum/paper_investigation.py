import copy
import logging
import os
from typing import Optional, Tuple

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt

from helper_methods.signal_processing_methods import (
    fourier_spectrum,
    normalised_fourier_spectrum,
    normalised_square_envelope_spectrum,
    square_envelope_spectrum,
)
from implementations.robust_optimized_weight_spectrum.paper_utils import (
    LogisticRegression,
    RegType,
)


def load_signal(signal_path: str, data_channel: int = 0):
    # Get the signals
    data_matrix = np.loadtxt(signal_path)

    # Return the required signal
    return data_matrix[:, data_channel].flatten()


def get_freq_domain_representation(
    x_signal: np.ndarray, Fs: int, norm_flag: bool = True, se_flag: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    # Get the spectrum method
    if norm_flag and se_flag:
        spect_method = normalised_square_envelope_spectrum

    elif not norm_flag and se_flag:
        spect_method = square_envelope_spectrum

    elif norm_flag and not se_flag:
        spect_method = normalised_fourier_spectrum

    elif not norm_flag and not se_flag:
        spect_method = fourier_spectrum

    else:
        raise ValueError("Unexpected error...")

    logging.info(f"Using `{spect_method.__name__}` method")

    # Get the signal length
    N = len(x_signal)

    # Get SES (one half)
    freq = np.fft.fftfreq(N, 1 / Fs)[: N // 2]
    X = 2 * spect_method(x_signal)[: N // 2]

    return freq, X


def get_training_data(
    x: np.ndarray,
    truncation_freq: Optional[int],
    Lw: Optional[int],  # Can be None if hankel_flag is false
    Ls: Optional[int],  # Can be None if hankel_flag is false
    Fs: int,
    hankel_flag: bool,
    norm_flag: bool,
    se_flag: bool,
    label_class: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Flatten signals
    x_flat = copy.deepcopy(x).flatten()

    if hankel_flag:
        assert Lw is not None, "Lw is None but you are using hankelisation"
        assert Ls is not None, "Lw is None but you are using hankelisation"

        # Perform hankelisation
        Lt = len(x_flat)
        Nw = (Lt - Lw) // Ls + 1

        window_list = []

        for idx in range(Nw):
            # Get windowed signal segment
            x_segment = x_flat[int(idx * Ls) : int(idx * Ls) + Lw]

            # Compute SES
            freq, x_ses = get_freq_domain_representation(
                x_segment, Fs=Fs, norm_flag=norm_flag, se_flag=se_flag
            )

            # Store
            window_list.append(x_ses)

        # Create matrix
        Xf_array = np.array(window_list)

        # Create labels
        Yf = np.ones(Nw) * label_class

    else:
        freq, x_ses = get_freq_domain_representation(
            x_flat, Fs=Fs, norm_flag=norm_flag, se_flag=se_flag
        )
        Xf_array = x_ses.reshape(1, -1)
        Yf = np.ones(1) * label_class

    if truncation_freq is not None:
        # Truncate matrix
        trunc_index = np.argmin(np.abs(freq - truncation_freq))

        Xf_array = Xf_array[:, :trunc_index]
        freq = freq[:trunc_index]

    return Xf_array, Yf, freq


if __name__ == "__main__":
    # Constants
    base_dir = "../../Datasets/IMS/dataset_two/"
    save_dir = "./results/IMS/"
    experiment_id = 3
    mat_save_path = os.path.join(save_dir, f"OSES_results_exp_{experiment_id}.npy")

    Fs = 20480
    file_list = sorted(os.listdir(base_dir))[:970]

    # Useful training parameters
    trunc_freq: Optional[int] = 5 * 300
    Lw = 5333
    Lsft = 400
    num_healthy = 30
    num_unhealthy = 10
    iter_range = range(num_healthy, 900 - num_unhealthy, 1)  # len(file_list)
    hankel_flag = True
    norm_flag = True
    se_flag = True

    # Model parameters
    learning_rate = 1.0
    max_iters = 2000
    regulariser_type = RegType.L2
    regulariser_alpha = 0.0025
    optimiser = "gradient_descent"  # "newton-cg"

    # Store all parameters in a dictionary
    training_parameters = {
        "Fs": Fs,
        "trunc_freq": trunc_freq,
        "Lw": Lw,
        "Lsft": Lsft,
        "num_healthy": num_healthy,
        "num_unhealthy": num_unhealthy,
        "hankel_flag": hankel_flag,
        "norm_flag": norm_flag,
        "se_flag": se_flag,
        "learning_rate": learning_rate,
        "max_iters": max_iters,
        "regulariser_type": regulariser_type.name,
        "regulariser_alpha": regulariser_alpha,
        "optimiser": optimiser,
    }

    try:
        OSES_results = np.load(mat_save_path, allow_pickle=True).item()
        freq = OSES_results["freq"]
        iteration_list = OSES_results["iteration_list"]
        loss_values = OSES_results["loss_values"]
        OSES_matrix = OSES_results["OSES_matrix"]

    except OSError:
        logging.info("Could not load data. Starting training loop...")
        os.makedirs(save_dir, exist_ok=True)

        Xh_list = []
        Yh_list = []
        freq = None

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
                hankel_flag=hankel_flag,
                norm_flag=norm_flag,
                se_flag=se_flag,
                label_class=0,
            )
            Xh_list.append(Xhi)
            Yh_list.append(Yhi)

        # Stack
        Xh = np.vstack(Xh_list)
        Yh = np.concatenate(Yh_list)
        assert freq is not None, "Something went wrong..."

        # Define storage lists
        OSES_list = []
        iteration_list = []
        loss_values = []

        # Define initial coefficients
        init_coeff = None

        # Perform iterations
        for cnt, i in enumerate(iter_range):
            print(f"\nWorking on file {i}...")

            Xt_list = []
            Yt_list = []

            for j in range(num_unhealthy):
                # Get the test signal
                x_test_j = load_signal(os.path.join(base_dir, file_list[i + j]))

                # Construct the data and label matrices
                Xtj, Ytj, _ = get_training_data(
                    x_test_j,
                    truncation_freq=trunc_freq,
                    Lw=Lw,
                    Ls=Lsft,
                    Fs=Fs,
                    hankel_flag=hankel_flag,
                    norm_flag=norm_flag,
                    se_flag=se_flag,
                    label_class=1,
                )

                Xt_list.append(Xtj)
                Yt_list.append(Ytj)

            # Stack
            Xt = np.vstack(Xt_list)
            Yt = np.concatenate(Yt_list)

            # Combine the data to get the true training data
            X = np.vstack([Xh, Xt])
            y = np.concatenate([Yh, Yt])

            # Define the model
            model = LogisticRegression(
                learning_rate=learning_rate,
                max_iter=max_iters,
                regulariser_type=regulariser_type,
                alpha=regulariser_alpha,
                optimiser=optimiser,  # "newton-cg"
                initial_coeffs=init_coeff,
            )

            # Fit the data
            model.fit(X, y)

            # Store
            iteration_list.append(model.get_iteration_steps())
            loss_values.append(model.get_loss_values())
            OSES_list.append(model.get_coefficients().copy())

            print(f"Iterations required: {model.get_iteration_steps()}")

            # Update coefficients
            init_coeff = model.get_coefficients().copy()

            if i % 100 == 0:
                plt.figure()
                plt.title(f"Record: {i}")
                plt.plot(freq, model.get_coefficients()[:-1])
                plt.show()

        # Create result dict
        OSES_matrix = np.array(OSES_list)

        OSES_results = {
            "freq": freq,
            "iteration_list": iteration_list,
            "loss_values": loss_values,
            "OSES_matrix": OSES_matrix,
            "training_parameters": training_parameters,  # Add the parameters here
        }

        np.save(mat_save_path, OSES_results)

    finally:
        # Visualise the signal and the spectra
        x_1 = load_signal(os.path.join(base_dir, file_list[0]))
        x_2 = load_signal(os.path.join(base_dir, file_list[750]))
        t = np.arange(0, 1, 1 / Fs)

        _, X1 = get_freq_domain_representation(
            x_1, Fs=Fs, norm_flag=norm_flag, se_flag=se_flag
        )
        freq_sig, X2 = get_freq_domain_representation(
            x_2, Fs=Fs, norm_flag=norm_flag, se_flag=se_flag
        )

        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        ax[0, 0].plot(t, x_1, color="b")
        ax[0, 1].plot(t, x_2, color="b")

        for axs in ax[0, :]:
            axs.grid(True)
            axs.set_xlabel("Time (s)", fontsize=18)

        ax[1, 0].plot(freq_sig, X1, color="b")
        ax[1, 1].plot(freq_sig, X2, color="b")

        for axs in ax[1, :]:
            axs.grid(True)
            axs.set_xlabel("Frequency (Hz)", fontsize=18)

        ax[0, 0].set_title("Healthy signal", fontsize=18)
        ax[0, 1].set_title("Unhealthy signal", fontsize=18)

        ax[0, 0].set_ylabel("Time-series signal", fontsize=18)
        ax[1, 0].set_ylabel("Amplitude spectra", fontsize=18)
        plt.savefig(
            os.path.join(
                save_dir, "figures", f"exp_{experiment_id}_signal_examples.png"
            ),
            dpi=300,
        )
        plt.show()

        # Remove zero values
        OSES_matrix = OSES_matrix[:, :-1]
        print(OSES_matrix.shape)

        # Drop the list dimension of Z (constants)
        X, Y = np.meshgrid(freq, iter_range)

        # View iterations to solution
        fig, ax = plt.subplots()
        ax.plot(iter_range, iteration_list, color="b", lw=0.75)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Iterations to solution")
        ax.grid(True)
        plt.savefig(
            os.path.join(save_dir, "figures", f"exp_{experiment_id}_iterations.png"),
            dpi=300,
        )
        plt.show()

        # View specific solutions
        fig, ax = plt.subplots(4, 1, figsize=(8, 12))

        ax[0].plot(freq, OSES_matrix[300, :], color="b", label="Record 300")
        ax[1].plot(freq, OSES_matrix[540, :], color="b", label="Record 540")
        ax[2].plot(freq, OSES_matrix[700, :], color="b", label="Record 700")
        ax[3].plot(freq, OSES_matrix[800, :], color="b", label="Record 800")

        for axs in ax:
            axs.grid(True)
            axs.legend()
            axs.set_xlabel("Frequency (Hz)")
            axs.set_ylabel("Model\nparameters")
        plt.legend()
        plt.savefig(
            os.path.join(
                save_dir, "figures", f"exp_{experiment_id}_specific_results.png"
            ),
            dpi=300,
        )
        plt.show()

        # View OSES surface
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"), figsize=(12, 12))
        norm = mcolors.Normalize(vmin=OSES_matrix.min(), vmax=OSES_matrix.max())

        ax.plot_surface(
            X, Y, OSES_matrix, cmap=plt.cm.jet, rstride=2, cstride=1, norm=norm
        )

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Sample index")
        ax.set_zlabel("OSES")
        plt.savefig(
            os.path.join(save_dir, "figures", f"exp_{experiment_id}_OSES.png"), dpi=300
        )
        plt.show()
