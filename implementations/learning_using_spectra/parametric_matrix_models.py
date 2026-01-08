from enum import StrEnum
from typing import Any, Optional

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


class EigenSelection(StrEnum):
    smallest_algebraic = "SA"
    largest_algebraic = "LA"
    smallest_magnitude = "SM"
    largest_magnitude = "LM"
    exterior_algebraically = "EA"
    exterior_by_magnitude = "EM"
    interior_algebraically = "IA"
    interior_by_magnitude = "IM"


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


class SymmetricMatrix(tf.keras.constraints.Constraint):
    # Enforce Hermitian matrix
    def call(self, w):
        return 0.5 * (w + tf.transpose(w))


class AffineHermitianMatrix(tf.keras.layers.Layer):
    def __init__(
        self,
        matrix_size: int,
        num_features: int,
        init_scale: float = 1e-2,
        jitter: float | None = None,
        enforce_hermitian: bool = True,
        bias_term: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.matrix_size = matrix_size
        self.num_features = num_features
        self.init_scale = init_scale
        self.jitter = jitter
        self.enforce_hermitian = enforce_hermitian
        self.bias_term = bias_term

        # number of matrices = features + 1 if bias_term else features
        self.q = self.num_features + 1 if self.bias_term else self.num_features

    def build(self, input_shape):
        self.M = self.add_weight(
            shape=(self.q, self.matrix_size, self.matrix_size),
            trainable=True,
            name="projection_matrices",
            constraint=SymmetricMatrix() if self.enforce_hermitian else None,
            initializer=tf.keras.initializers.RandomNormal(stddev=self.init_scale),
        )

    def call(self, x):
        # Create projection matrix
        if self.bias_term:
            ones = tf.ones([tf.shape(x)[0], 1], dtype=x.dtype)
            x = tf.concat([ones, x], axis=-1)
        else:
            x = x

        # Sum over N square matrices weighted by the features
        # Note: Adding bias is equivalent to adding M0 + sum_i x_i M_i, 1 <= i <= 3
        projection_matrix = tf.einsum("Bi,ijk->Bjk", x, self.M)  # (B, n, n)

        # Optional small diagonal jitter to stabilize eigenspectrum
        if self.jitter and self.jitter > 0:
            eye = tf.eye(self.matrix_size, dtype=x.dtype)[None, :, :]
            projection_matrix += self.jitter * eye

        return projection_matrix


def _select_indices(sorted_idx: tf.Tensor, k: int, which: str) -> tf.Tensor:
    """
    Which eigenvectors to use based on eigenvalue.
    Options are:
    - 'SA' for smallest algebraic
    - 'LA' for largest algebraic
    - 'SM' for smallest magnitude
    - 'LM' for largest magnitude
    - 'EA' for exterior algebraically
    - 'EM' for exterior by magnitude
    - 'IA' for interior algebraically
    - 'IM' for interior by magnitude
    """
    # sorted_idx: (batch, n) indices that sort eigenvalues ascending by chosen metric
    # returns (batch, k) indices

    which = which.upper()
    n = tf.shape(sorted_idx)[-1]

    match which[0].lower():
        case "s":  # Smallest
            return sorted_idx[:, :k]
        case "l":  # Largest
            return sorted_idx[:, -k:]
        case "e":  # Exterior: half of each end
            k_half = k // 2
            k_rem = k - k_half
            left = sorted_idx[:, :k_half]
            right = sorted_idx[:, -k_rem:]
            return tf.concat([left, right], axis=-1)
        case "i":  # Interior: around the centre
            n_half = n // 2
            k_half = k // 2
            k_rem = k - k_half
            left = sorted_idx[:, n_half - k_half : n_half]
            right = sorted_idx[:, n_half : n_half + k_rem]
            return tf.concat([left, right], axis=-1)
        case _:  # Else
            raise ValueError("which must start with one of S, L, E, or I.")


class EigenvaluesSelector(tf.keras.layers.Layer):
    def __init__(
        self,
        num_eig: int | None = 1,  # None returns all
        which: str | EigenSelection = "SA",
        return_vectors: bool = False,
        name: str | None = None,
    ):
        super().__init__(name=name)
        if num_eig is not None and (not isinstance(num_eig, int) or num_eig <= 0):
            raise ValueError("num_eig must be a positive int or None")

        if which not in EigenSelection:
            raise ValueError(
                f"which must be one of {list(EigenSelection._value2member_map_.keys())}"
            )

        self.num_eig = num_eig
        self.which = EigenSelection(which)
        self.return_vectors = return_vectors

    def call(self, M: tf.Tensor) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor]:
        # M shape: (batch, n, n)
        if M.shape.rank != 3 or M.shape[-1] != M.shape[-2]:
            raise ValueError("Input must be a batch of square matrices (b, n, n)")

        # Compute eigenvalues and eigenvectors
        E, V = tf.linalg.eigh(M)

        if self.num_eig is None:
            return (E, V) if self.return_vectors else E

        # Choose eigenvalue sort by algebraic (A) or magnitude (M)
        if self.which.value.endswith("A"):
            # algebraic sort
            sorted_idx = tf.argsort(E, axis=-1, stable=True)
        else:
            # magnitude sort
            sorted_idx = tf.argsort(tf.abs(E), axis=-1, stable=True)

        select_idx = _select_indices(sorted_idx, self.num_eig, self.which)

        # Gather selected eigenvalues and vectors per batch
        # First sort columns by sorted_idx, then pick according to idx
        E_sorted = tf.gather(E, sorted_idx, batch_dims=1, axis=-1)  # (b, n)
        E_sel = tf.gather(E_sorted, select_idx, batch_dims=1, axis=-1)  # (b, k)

        if V is None or not self.return_vectors:
            return E_sel

        V_sorted = tf.gather(V, sorted_idx, batch_dims=1, axis=-1)  # (b, n, n)
        V_sel = tf.gather(V_sorted, select_idx, batch_dims=1, axis=-1)  # (b, n, k)

        return E_sel, V_sel


class AffineEigenvaluePMM(tf.keras.layers.Layer):
    def __init__(
        self,
        matrix_size: int,
        num_features: int,
        enforce_hermitian: bool = True,
        bias_term: bool = True,
        init_scale: float = 1e-2,
        jitter: float = 0.0,
        num_eig: int = 1,
        which: str = "SA",
        agg_eigs: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.agg_eigs = agg_eigs

        self.affine = AffineHermitianMatrix(
            matrix_size=matrix_size,
            num_features=num_features,
            enforce_hermitian=enforce_hermitian,
            bias_term=bias_term,
            init_scale=init_scale,
            jitter=jitter,
            name="aff_proj" if name is None else f"aff_proj_{name}",
        )

        self.selector = EigenvaluesSelector(
            num_eig=num_eig,
            which=which,
            return_vectors=False,  # Explicitly false in this learning scenario
            name="eigen_selection" if name is None else f"eigen_selection{name}",
        )

    def call(self, features: tf.Tensor):
        # Project
        M = self.affine(features)

        # Compute eigenvalues
        eigs = self.selector(M)

        # Aggregate
        if self.agg_eigs:
            return tf.keras.ops.sum(eigs, axis=1, keepdims=True)
        else:
            return eigs


if __name__ == "__main__":
    # TODO: Formalise train and test sets
    # Define function to approximate
    function_name = "gaussians"
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
