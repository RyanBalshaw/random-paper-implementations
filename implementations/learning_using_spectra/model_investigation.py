from typing import Any

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from implementations.learning_using_spectra.parametric_matrix_models import (
    AffineEigenvaluePMM,
)


def booth_function(x: float, y: float) -> float:
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2


def himmelblau_function(x: float, y: float) -> float:
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


def ackley_function(x: float, y: float) -> float:
    return (
        -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        + np.e
        + 20
    )


def beale_function(x: float, y: float) -> float:
    return (
        (1.5 - x + x * y) ** 2
        + (2.25 - x + x * y**2) ** 2
        + (2.625 - x + x * y**3) ** 2
    )


def gaussians(x, y):
    u = 5 - 10 * x
    v = 5 - 10 * y

    return np.exp(-(u**2) / 2) + 3 / 4 * np.exp(-(v**2) / 2) * (
        1 + np.exp(-(u**2) / 2)
    )


def trigonometric(x, y):
    return 2 * np.cos(10 * x) * np.sin(10 * y) + np.sin(10 * x * y)


function_lookup: dict[str : dict[str:Any]] = {
    "booth": {
        "function": booth_function,
        "domain": {"x": [-10, 10], "y": [-10, 10]},
    },
    "himmelblau": {
        "function": himmelblau_function,
        "domain": {"x": [-5, 10], "y": [-10, 5]},
    },
    "ackley": {
        "function": ackley_function,
        "domain": {"x": [-5, 10], "y": [-10, 5]},
    },
    "beale": {
        "function": beale_function,
        "domain": {"x": [-4.5, 10], "y": [-10, 4.5]},
    },
    "gaussians": {
        "function": gaussians,
        "domain": {"x": [0, 1], "y": [0, 1]},
    },
    "trigonometric": {
        "function": trigonometric,
        "domain": {"x": [0, 1], "y": [0, 1]},
    },
}


def make_trainset(
    function: callable,
    domain: dict[str, list[float]],
    n_train: int = 200,
    train_noise: float = 0.2,
) -> tuple[float, float, float]:
    # Extract function information
    np.random.seed(42)
    x = np.random.uniform(domain["x"][0], domain["x"][1], n_train)
    y = np.random.uniform(domain["y"][0], domain["y"][1], n_train)

    z = function(x, y) + train_noise * np.random.randn(n_train)

    z = (z - np.mean(z)) / np.std(z)

    return x, y, z


if __name__ == "__main__":
    # TODO: Formalise train and test sets
    # Define function to approximate
    function_name = "trigonometric"
    n_train = 1000
    train_noise = 0.0
    batch_size = 512
    learning_rate = 0.01
    epochs = 200
    matrix_dims = 64
    num_eig = 3
    which = "IA"

    # Get data
    function_params = function_lookup[function_name]
    func_domain = function_params["domain"]
    func = function_params["function"]

    x_train, y_train, z_train = make_trainset(
        function=func, domain=func_domain, n_train=n_train, train_noise=train_noise
    )

    # Create a dataset
    dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                np.hstack(
                    (
                        x_train[:, None].astype(np.float32),
                        y_train[:, None].astype(np.float32),
                    )
                ),
                z_train[:, None].astype(np.float32),
            )
        )
        .shuffle(1000)
        .batch(batch_size)
    )

    # Visualise training problem
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(x_train, y_train, z_train, color="red")
    X, Y = np.meshgrid(
        np.linspace(func_domain["x"][0], func_domain["x"][1], 100),
        np.linspace(func_domain["y"][0], func_domain["y"][1], 100),
    )
    ax.plot_surface(X, Y, func(X, Y))
    # plt.show()

    # Define the Tensorflow model
    input_xy = tf.keras.layers.Input(shape=(2,), name="data_input", dtype=tf.float32)
    aepmm = AffineEigenvaluePMM(
        matrix_size=matrix_dims,
        num_features=2,
        enforce_hermitian=True,
        bias_term=True,
        init_scale=1e-2,
        jitter=1e-6,
        num_eig=num_eig,
        which=which,
        agg_eigs=True,
    )
    eigvals_func = aepmm(input_xy)

    model = tf.keras.Model(inputs=input_xy, outputs=eigvals_func)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer, loss="mse", steps_per_execution=10)
    model.summary()

    # Train the model
    history = model.fit(dataset, epochs=epochs)

    # Visualise training loss
    plt.figure(figsize=(5, 4))
    plt.plot(history.history["loss"], label="train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    # plt.show()

    # Get the fitted surface and the predicted at the training data
    xs = np.linspace(func_domain["x"][0], func_domain["x"][1], 100, dtype=np.float32)
    ys = np.linspace(func_domain["y"][0], func_domain["y"][1], 100, dtype=np.float32)
    Xg, Yg = np.meshgrid(xs, ys)

    Zg = model.predict(
        np.hstack(
            (Xg.reshape(-1, 1).astype(np.float32), Yg.reshape(-1, 1).astype(np.float32))
        ),
        verbose=0,
    )
    Zg = Zg.reshape(Xg.shape)

    z_pred_train = model.predict(
        np.hstack(
            (x_train[:, None].astype(np.float32), y_train[:, None].astype(np.float32))
        ),
        verbose=0,
    ).squeeze()

    # Visualise the fitted surface
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # ax.scatter(
    #     x_train, y_train, z_train, color="red", s=8, alpha=0.7, label="training data"
    # )
    surf = ax.plot_surface(
        Xg, Yg, Zg, cmap="viridis", alpha=0.7, linewidth=0, antialiased=True
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Fitted model vs training points")
    ax.legend()
    # plt.show()

    # Visualise the error distribution
    residuals = z_pred_train - z_train
    abs_errors = np.abs(residuals)

    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(residuals**2))
    bias = np.mean(residuals)

    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=40, color="steelblue", edgecolor="white", alpha=0.9)
    plt.axvline(0, color="k", linestyle="--", linewidth=1, label="zero error")
    plt.xlabel("Residual (prediction - true)")
    plt.ylabel("Count")
    plt.title(f"Training residuals\nMAE={mae:.3f}, RMSE={rmse:.3f}, Bias={bias:.3f}")
    plt.legend()
    plt.grid(alpha=0.3)
    # plt.show()

    # Visualise the prediction fit
    plt.figure(figsize=(5, 5))
    plt.scatter(z_train, z_pred_train, s=12, alpha=0.6)
    lims = [
        min(z_train.min(), z_pred_train.min()),
        max(z_train.max(), z_pred_train.max()),
    ]
    plt.plot(lims, lims, "r--", linewidth=1, label="ideal")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("True z")
    plt.ylabel("Predicted z")
    plt.title("Predicted vs True (training)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
