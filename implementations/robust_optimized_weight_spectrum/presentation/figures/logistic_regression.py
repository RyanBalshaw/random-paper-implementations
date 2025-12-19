"""
NOTE: If you stumble on this code, Gemini wrote it. I just make it represent what I
wanted.
"""

import matplotlib.pyplot as plt
import numpy as np

# --- 1. Generate Sample Data for Binary Classification (1D) ---
# We'll create two clusters of data points along a single feature (x1)
np.random.seed(42)  # for reproducibility

# Class 0 data
num_samples_0 = 50
mean_0 = 2
std_dev_0 = 1
X_0 = np.random.normal(mean_0, std_dev_0, num_samples_0)
y_0 = np.zeros(num_samples_0)  # Label for class 0

# Class 1 data
num_samples_1 = 50
mean_1 = 5
std_dev_1 = 1
X_1 = np.random.normal(mean_1, std_dev_1, num_samples_1)
y_1 = np.ones(num_samples_1)  # Label for class 1

# Combine data
X = np.hstack((X_0, X_1)).reshape(-1, 1)  # Reshape to a column vector for consistency
y = np.hstack((y_0, y_1))

# --- 2. Simulate Model Parameters (w) for 1D ---
# We'll assume a model of the form: z = w0 + w1*x1
# So, w = [w0, w1]
# The decision boundary is where z = 0, meaning w0 + w1*x1 = 0.
# This can be rearranged to solve for x1: x1 = -w0 / w1

# Let's pick some illustrative parameters for w to create a separating point
w0_simulated = -3.5  # Intercept
w1_simulated = 1.0  # Coefficient for x1
simulated_weights = np.array([w0_simulated, w1_simulated])

# Calculate the decision boundary point: x1 = -w0 / w1
decision_boundary_x = None
if simulated_weights[1] != 0:  # Avoid division by zero
    decision_boundary_x = -simulated_weights[0] / simulated_weights[1]


# --- 3. Define the Sigmoid Function ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# --- 4. Determine plot range for x-axis ---
x1_min, x1_max = X.min() - 1, X.max() + 1
x_plot_range = np.linspace(x1_min, x1_max, 500)

# Prepare data for pre-logit calculation (add intercept term)
X_plot_with_intercept = np.c_[np.ones(x_plot_range.shape[0]), x_plot_range]
z_values_for_plot = np.dot(X_plot_with_intercept, simulated_weights)

# Calculate probabilities using the sigmoid function
probabilities_for_plot = sigmoid(z_values_for_plot)

# --- 5. Plot 1: Pre-Logit Function and Data Distribution ---
fig1, ax1 = plt.subplots(figsize=(12, 10))

# Plot data points along the x-axis with a slight y-jitter for visibility
y_jitter_amount = 0.1  # Adjust for desired spread
ax1.scatter(
    X_0,
    np.zeros_like(X_0)
    + np.random.uniform(-y_jitter_amount, y_jitter_amount, num_samples_0),
    color="blue",
    label="Class 0 (True)",
    alpha=0.7,
    edgecolors="w",
    s=80,
    zorder=3,
)
ax1.scatter(
    X_1,
    np.zeros_like(X_1)
    + np.random.uniform(-y_jitter_amount, y_jitter_amount, num_samples_1),
    color="red",
    label="Class 1 (True)",
    alpha=0.7,
    edgecolors="w",
    s=80,
    zorder=3,
)

# Plot the pre-logit line
ax1.plot(
    x_plot_range,
    z_values_for_plot,
    color="purple",
    linestyle="-",
    linewidth=2,
    label="Pre-Logit Function ($z = w_0 + w_1 x_1$)",
    zorder=1,
)
ax1.axhline(
    0, color="gray", linestyle=":", linewidth=0.8, zorder=0
)  # Reference line at z=0

# Draw the Decision Boundary (Vertical Line)
if decision_boundary_x is not None:
    ax1.axvline(
        x=decision_boundary_x,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Decision Boundary ($z = 0$)",
        zorder=2,
    )
    # Add text label for boundary
    ax1.text(
        decision_boundary_x,
        ax1.get_ylim()[1] * 0.1,
        f"x={decision_boundary_x:.2f}",
        horizontalalignment="center",
        color="black",
        fontsize=20,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

# Shade Regions based on "Pre-Logit" Value (implied classification)
if decision_boundary_x is not None:
    ax1.axvspan(
        x1_min,
        decision_boundary_x,
        color="blue",
        alpha=0.1,
        label="Predicted Class 0 (z < 0)",
        zorder=0,
    )
    ax1.axvspan(
        decision_boundary_x,
        x1_max,
        color="red",
        alpha=0.1,
        label="Predicted Class 1 (z > 0)",
        zorder=0,
    )
else:
    ax1.text(
        x1_min + (x1_max - x1_min) / 2,
        ax1.get_ylim()[1] * 0.5,
        "No x1-dependent boundary (w1=0)",
        horizontalalignment="center",
        color="black",
        fontsize=20,
    )

# Customizing Plot 1
ax1.set_title("Logistic Regression: Linear Function", fontsize=20, pad=15)
ax1.set_xlabel("Feature 1 ($x_1$)", fontsize=20, labelpad=10)
ax1.set_ylabel("Pre-Logit Value ($z$)", color="purple", fontsize=20, labelpad=10)

# Set y-axis limits for Plot 1
y_min_z = z_values_for_plot.min() - 0.5
y_max_z = z_values_for_plot.max() + 0.5
ax1.set_ylim(y_min_z, y_max_z)

ax1.set_xlim(x1_min, x1_max)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.tick_params(axis="x", labelsize=16)
ax1.tick_params(axis="y", labelcolor="purple", labelsize=16)
ax1.legend(fontsize=20, loc="upper left")
plt.tight_layout()
plt.savefig("logistic_regression_pre_logit.png", dpi=300)
plt.show()

# --- 6. Plot 2: Sigmoid Function and Probability ---
fig2, ax2 = plt.subplots(figsize=(12, 10))

# Plot the sigmoid function
ax2.plot(
    x_plot_range,
    probabilities_for_plot,
    color="green",
    linestyle="-",
    linewidth=2,
    label="Sigmoid Function ($P = \sigma(z)$)",  # noqa
    zorder=1,
)
ax2.axhline(
    0.5,
    color="darkgreen",
    linestyle=":",
    linewidth=0.8,
    label="Probability Threshold ($P = 0.5$)",
    zorder=0,
)  # Reference line at P=0.5

# Draw the Decision Boundary (Vertical Line)
if decision_boundary_x is not None:
    ax2.axvline(
        x=decision_boundary_x,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Decision Boundary ($P = 0.5$)",
        zorder=2,
    )

    # Add a point on the sigmoid curve at the decision boundary
    prob_at_boundary = sigmoid(0)  # Should be 0.5
    ax2.plot(
        decision_boundary_x,
        prob_at_boundary,
        "o",
        color="darkgreen",
        markersize=8,
        zorder=4,
    )
    ax2.text(
        decision_boundary_x + (x1_max - x1_min) * 0.02,
        prob_at_boundary,
        f"P={prob_at_boundary:.1f}",
        verticalalignment="center",
        color="darkgreen",
        fontsize=20,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

# ADDED: Plot data circles at the bottom of the second figure
# Use y=0 for Class 0 points and y=1 for Class 1 points, with jitter
y_jitter_amount_for_prob_plot = 0.02  # Smaller jitter for probability plot
ax2.scatter(
    X_0,
    np.zeros_like(X_0)
    + np.random.uniform(
        -y_jitter_amount_for_prob_plot, y_jitter_amount_for_prob_plot, num_samples_0
    ),
    color="blue",
    alpha=0.7,
    edgecolors="w",
    s=80,
    zorder=3,
    label="Class 0 (True Data)",
)
ax2.scatter(
    X_1,
    np.ones_like(X_1)
    + np.random.uniform(
        -y_jitter_amount_for_prob_plot, y_jitter_amount_for_prob_plot, num_samples_1
    ),
    color="red",
    alpha=0.7,
    edgecolors="w",
    s=80,
    zorder=3,
    label="Class 1 (True Data)",
)


# Shade Regions based on Probability (implied classification)
if decision_boundary_x is not None:
    ax2.axvspan(
        x1_min,
        decision_boundary_x,
        color="blue",
        alpha=0.1,
        label="Predicted Class 0 (P < 0.5)",
        zorder=0,
    )
    ax2.axvspan(
        decision_boundary_x,
        x1_max,
        color="red",
        alpha=0.1,
        label="Predicted Class 1 (P > 0.5)",
        zorder=0,
    )
else:
    ax2.text(
        x1_min + (x1_max - x1_min) / 2,
        ax2.get_ylim()[1] * 0.5,
        "No x1-dependent boundary (w1=0)",
        horizontalalignment="center",
        color="black",
        fontsize=20,
    )

# Customizing Plot 2
ax2.set_title("Logistic Regression: Probabilities", fontsize=20, pad=15)
ax2.set_xlabel("Feature 1 ($x_1$)", fontsize=20, labelpad=10)
ax2.set_ylabel("Probability ($P$)", color="green", fontsize=20, labelpad=10)

# Set y-axis limits for Plot 2
ax2.set_ylim(-0.05, 1.05)  # Probabilities are between 0 and 1

ax2.set_xlim(x1_min, x1_max)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.tick_params(axis="x", labelsize=10)
ax2.tick_params(axis="y", labelcolor="green", labelsize=10)
ax2.legend(fontsize=20, loc="upper left")
plt.tight_layout()
plt.savefig("logistic_regression_post_logit.png", dpi=300)
plt.show()


# Define the Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Generate a range of z values (from negative to positive infinity, practically)
z_values = np.linspace(-10, 10, 500)

# Calculate the corresponding sigmoid probabilities
probabilities = sigmoid(z_values)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(
    z_values,
    probabilities,
    color="green",
    linewidth=2,
    label="Sigmoid Function $\sigma(z)$",  # noqa
)

# Add key reference lines
plt.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="P = 0.5")
plt.axvline(0, color="gray", linestyle="--", linewidth=0.8, label="z = 0")
plt.plot(
    0, 0.5, "o", color="darkgreen", markersize=8, label="($z=0, P=0.5$)"
)  # Point at (0, 0.5)

# Customize the plot
plt.title("The Sigmoid Function", fontsize=20, pad=15)
plt.xlabel("Pre-Logit Value ($z$)", fontsize=18, labelpad=10)
plt.ylabel("$\sigma(z)$", fontsize=18, labelpad=10)  # noqa
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=18)  # Set y-ticks to show probabilities
plt.ylim(-0.05, 1.05)  # Ensure y-axis covers 0 to 1 with slight buffer
plt.tight_layout()
plt.savefig("sigmoid.png", dpi=300)
plt.show()

# Generate x values for the logarithm function
# The logarithm is only defined for x > 0.
# We'll use a range that starts close to 0 but not exactly 0 to avoid errors.
x_values = np.linspace(
    0.1, 10, 500
)  # Start from 0.1 to avoid log(0) which is undefined

# Calculate the corresponding y values using the natural logarithm (base e)
# np.log() calculates the natural logarithm. Use np.log10() for base 10, or np.log2()
# for base 2.
y_values = np.log(x_values)

# Create the plot
plt.figure(figsize=(10, 7))
plt.plot(
    x_values,
    y_values,
    color="purple",
    linewidth=2,
    label="Natural Logarithm $y = \ln(x)$",  # noqa
)

# Add key reference lines
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)  # x-axis (where y=0)
plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)  # y-axis (where x=0)

# Mark the important point (1, 0) where the logarithm crosses the x-axis
plt.plot(1, 0, "o", color="darkgreen", markersize=8, label="Point (1, 0)")
plt.text(0.4, 0.2, "(1, 0)", fontsize=16, color="darkgreen")

# Customize the plot
plt.title(
    "Plot of the Natural Logarithm Function $y = \ln(x)$", fontsize=16, pad=15  # noqa
)
plt.xlabel("x", fontsize=18, labelpad=10)
plt.ylabel("y", fontsize=18, labelpad=10)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Adjust y-axis limits to better show the behavior near x=0
plt.ylim(-3, 2.5)  # Example limits, adjust as needed

plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.savefig("natural_logarithm.png", dpi=300)
plt.show()
