import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from implementations.learning_using_spectra.parametric_matrix_models import (
    AffineEigenvaluePMM,
)


def create_hankel_windows(time_series, window_size):
    # batch_size=None yields individual samples (not batches)
    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=time_series[:-1],
        targets=time_series[window_size:],
        sequence_length=window_size,
        batch_size=None,
        shuffle=False,
    )

    X, y = [], []
    for inputs, targets in dataset:
        X.append(inputs)
        y.append(targets)

    # Use tf.stack to create the batch dimension
    # X becomes (Batch_Size, Window_Size)
    # y becomes (Batch_Size,)
    return tf.stack(X), tf.stack(y)


def autoregressive_forecast(model, initial_window, steps):
    """
    Forecasting by feeding the model's predictions back into itself.
    """
    # Start with the last window of training data
    current_window = initial_window.copy()  # Shape (window_size,)
    predictions = []

    print(f"Starting autoregressive forecast for {steps} steps...")

    for _ in range(steps):
        # 1. Prepare input: Add batch dimension (1, window_size)
        input_tensor = tf.convert_to_tensor(current_window[None, :], dtype=tf.float32)

        # 2. Predict next value
        # We use model() instead of model.predict() for speed in loops
        pred_tensor = model(input_tensor, training=False)
        pred_val = float(pred_tensor.numpy()[0, 0])

        predictions.append(pred_val)

        # 3. Update window: Shift left, append new prediction
        # Old: [x0, x1, x2, x3]
        # New: [x1, x2, x3, pred]
        current_window = np.roll(current_window, -1)
        current_window[-1] = pred_val

    return np.array(predictions)


# 1. Hyperparameters
window_size = 128  # This is 'num_features' (p)
matrix_size = 16  # Size of the learned Hamiltonian
output_size = 1  # Predicting next scalar value


# Example: Mackey-Glass or Sine wave
Fs = 50
t_train = np.arange(0, 100, 1 / Fs)
t_test = np.arange(100, 150, 1 / Fs)

assert len(t_test) > window_size


def generate_signal(t):
    return np.sin(5 * t) + 0.5 * np.sin(10 * t) + 0.1 * np.sin(20 * t)  # 1D Time Series


x_train = generate_signal(t_train)
y_test_raw = generate_signal(t_test)

overlap = x_train[-window_size:]
test_data_combined = np.concatenate([overlap, y_test_raw])

# Create Windows (Features)
X_train, y_train = create_hankel_windows(x_train, window_size)
X_test, y_test = create_hankel_windows(test_data_combined, window_size)

print(f"Input Shape: {X_train.shape}")  # (Batch, 10)
plt.figure(figsize=(12, 6))

# Plot a slice of Training Data for context
plt.plot(t_train, x_train, label="Training Data", color="gray", alpha=0.5)
plt.plot(
    t_train[:window_size],
    x_train[:window_size],
    label="Window size",
    color="red",
    alpha=0.5,
)
plt.xlabel("Time (t)")
plt.ylabel("Signal Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-1.5, 1.5)
plt.show()

# 2. Instantiate Model
input_x = tf.keras.layers.Input(shape=(window_size,))
aepmm = AffineEigenvaluePMM(
    matrix_size=matrix_size,
    num_features=window_size,
    enforce_hermitian=True,
    bias_term=True,
    init_scale=1e-2,
    jitter=1e-6,
    num_eig=16,
    which="EA",
    agg_eigs=True,
)

eigvals_func = aepmm(input_x)
ts_model = tf.keras.Model(inputs=input_x, outputs=eigvals_func)


# 3. Compile and Train
ts_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse")

ts_model.summary()

# Note: We cast inputs to complex64 inside the layer,
# but passing float32 is fine as the entry point.
ts_model.fit(X_train, y_train, epochs=20, batch_size=128)

# 4. Make Autoregressive predictions
# We start from the VERY LAST window of the training set.
seed_window = x_train[-window_size:]
steps_to_forecast = len(y_test_raw)

# Run the recursive loop
y_pred_autoregressive = autoregressive_forecast(
    ts_model, seed_window, steps_to_forecast
)

# 5. Calculate Metrics
mse = mean_squared_error(y_test.numpy(), y_pred_autoregressive)
mae = mean_absolute_error(y_test.numpy(), y_pred_autoregressive)
rmse = np.sqrt(mse)

print("\nEvaluation Metrics (Extrapolation):")
print(f"MSE:  {mse:.5f}")
print(f"MAE:  {mae:.5f}")
print(f"RMSE: {rmse:.5f}")

# 6. Visualization
plt.figure(figsize=(12, 6))

# Plot a slice of Training Data for context
plt.plot(
    t_train[-window_size:],
    x_train[-window_size:],
    label="Training Data (window size)",
    color="gray",
    alpha=0.5,
)

# Plot True Future
plt.plot(t_test, y_test_raw, "k--", label="True Future (Unknown to Model)", alpha=0.8)

# Plot Predictions
# Adjust t_test indices to match X_test alignment
plt.plot(t_test, y_pred_autoregressive, "r-", label="PMM Prediction", linewidth=1.5)

plt.axvline(x=t_test[0], color="blue", linestyle=":", label="Training Cutoff")
plt.title(
    f"PMM Extrapolation Performance\nWindow Size: {window_size}, "
    f"Matrix Size: {matrix_size}"
)
plt.xlabel("Time (t)")
plt.ylabel("Signal Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-1.5, 1.5)
plt.show()

# 7. Diagnostic: Check Eigenvalues (The "Frequencies")
# This checks if the model learned stable physics.
# Real eigenvalues = stable. Complex eigenvalues = unstable/decaying.

# Get the affine layer
affine_layer = ts_model.get_layer("affine_eigenvalue_pmm")

# Project a few random test windows to get their Hamiltonians
# We cast to complex64 because your layer expects it
sample_inputs = X_test[:5]
sample_matrices = affine_layer.affine(sample_inputs)  # (5, n, n)

# Compute eigenvalues
# TODO: Show evolutions of individual eigenvalues through time
# OR: Show eigenvalues/eigenvectors of learnt matrices in affine_layer
