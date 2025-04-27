import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs  # , make_classification

from implementations.robust_optimized_weight_spectrum.paper_utils import (
    LogisticRegression,
    RegType,
)


def simple_problem():
    # X, y = make_classification(n_features=2, n_informative=2, n_redundant=0)
    X, y = make_blobs(
        n_samples=400, n_features=2, centers=2, cluster_std=1, random_state=3
    )
    # X, y = make_circles(n_samples=1000,noise=0.01, random_state=0)

    plt.figure(figsize=(10, 6))
    colors = ListedColormap(["#FF9999", "#66B2FF"])

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=colors, edgecolors="k", s=50, alpha=0.8)
    plt.title("Binary Classification Dataset (make_blobs)", fontsize=14)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    plt.grid(alpha=0.3)

    # Add legend
    for class_id, class_name in enumerate(["Class 0", "Class 1"]):
        plt.scatter([], [], c=colors(class_id), label=class_name, edgecolors="k", s=50)
    plt.legend()

    plt.tight_layout()
    plt.show()

    model = LogisticRegression(
        learning_rate=0.1,
        max_iter=400,
        regulariser_type=RegType.L1,
        alpha=0.001,
        optimiser="gradient_descent",  #
        tol=1e-4,
    )
    model.fit(X, y)

    plt.figure(figsize=(10, 5))
    plt.plot(model._loss_values, color="#1f77b4", linewidth=2)
    plt.title("Loss Convergence During Training", fontsize=14)
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    X_grid, Y_grid = np.meshgrid(
        np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
        np.linspace(X[:, 1].min(), X[:, 1].max(), 100),
    )
    X_test = np.hstack([X_grid.ravel().reshape(-1, 1), Y_grid.ravel().reshape(-1, 1)])

    D = model.predict_proba(X_test)
    Z = model.predict_log_odds(X_test)

    # Plot 3: Enhanced Contour Plot with Decision Boundary
    plt.figure(figsize=(10, 8))
    # Custom colormap for probabilities
    cmap = plt.cm.RdYlBu_r  # Red-Yellow-Blue reversed (red = class 1, blue = class 0)

    # Create filled contour plot for probabilities
    contour = plt.contourf(
        X_grid,
        Y_grid,
        D.reshape(100, 100),
        levels=np.linspace(0, 1, 21),
        cmap=cmap,
        alpha=0.8,
    )

    # Add decision boundary (where probability = 0.5, or log-odds = 0)
    plt.contour(
        X_grid,
        Y_grid,
        Z.reshape(100, 100),
        levels=[0],
        colors="k",
        linestyles="-",
        linewidths=2,
    )

    # Plot data points with distinct colors and larger size
    colors = ListedColormap(["#0000AA", "#AA0000"])
    plt.scatter(
        X[:, 0], X[:, 1], c=y, cmap=colors, edgecolors="w", s=80, alpha=1.0, zorder=2
    )

    # Add colorbar and labels
    plt.colorbar(contour, label="Probability of Class 1")
    plt.title("Decision Boundary and Probability Contours", fontsize=16)
    plt.xlabel("Feature 1", fontsize=14)
    plt.ylabel("Feature 2", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Plot 4: 3D Surface Plot of Log-Odds
    # Create a single figure with two 3D subplots side by side
    fig = plt.figure(figsize=(20, 9))

    # First subplot for Log-Odds
    ax1 = fig.add_subplot(121, projection="3d")  # 1 row, 2 cols, position 1

    # Create the log-odds surface
    surf1 = ax1.plot_surface(
        X_grid,
        Y_grid,
        Z.reshape(100, 100),
        cmap="viridis",
        alpha=0.8,
        linewidth=0,
        antialiased=True,
    )

    # Add data points - project them onto the z=0 plane for clarity
    for i, (x_i, y_i) in enumerate(zip(X[:, 0], X[:, 1])):
        if y[i] == 0:
            ax1.scatter([x_i], [y_i], [0], color="blue", s=30, edgecolor="k")
        else:
            ax1.scatter([x_i], [y_i], [0], color="red", s=30, edgecolor="k")

    # Add colorbar
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5, label="Log-Odds")

    # Add labels
    ax1.set_xlabel("Feature 1", fontsize=14)
    ax1.set_ylabel("Feature 2", fontsize=14)
    ax1.set_zlabel("Log-Odds Value", fontsize=14)
    ax1.set_title("Log-Odds Surface with Decision Boundary", fontsize=16)
    # ax1.view_init(elev=30, azim=45)  # Adjust the viewing angle

    # Second subplot for Probability (D)
    ax2 = fig.add_subplot(122, projection="3d")  # 1 row, 2 cols, position 2

    # Create the probability surface
    surf2 = ax2.plot_surface(
        X_grid,
        Y_grid,
        D.reshape(100, 100),
        cmap="RdYlBu_r",
        alpha=0.8,
        linewidth=0,
        antialiased=True,
    )

    # Add data points - project them onto z=0 or z=1 based on their class
    for i, (x_i, y_i) in enumerate(zip(X[:, 0], X[:, 1])):
        z_i = 0 if y[i] == 0 else 1  # Project to bottom or top
        if y[i] == 0:
            ax2.scatter([x_i], [y_i], [z_i], color="blue", s=30, edgecolor="k")
        else:
            ax2.scatter([x_i], [y_i], [z_i], color="red", s=30, edgecolor="k")

    # Add colorbar
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5, label="Probability of Class 1")

    # Add labels
    ax2.set_xlabel("Feature 1", fontsize=14)
    ax2.set_ylabel("Feature 2", fontsize=14)
    ax2.set_zlabel("Probability", fontsize=14)
    ax2.set_title("Probability Surface with Decision Boundary", fontsize=16)
    # ax2.view_init(elev=30, azim=45)  # Same viewing angle for comparison
    ax2.set_zlim(0, 1)  # Set z limits to [0,1] for probability

    plt.tight_layout(pad=4.0)  # Add extra padding between subplots
    plt.show()

    # gradient_error, hessian_error = model.check_gradients_and_hessian(X, y)


if __name__ == "__main__":
    simple_problem()
