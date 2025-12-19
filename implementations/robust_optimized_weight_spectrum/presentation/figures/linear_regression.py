import matplotlib.pyplot as plt
import numpy as np

# --- Data Points ---
# Example data points (same as the HTML example)
data = [
    {'x': 10, 'y': 20}, {'x': 15, 'y': 25}, {'x': 20, 'y': 30},
    {'x': 25, 'y': 35}, {'x': 30, 'y': 40}, {'x': 35, 'y': 42},
    {'x': 40, 'y': 48}, {'x': 45, 'y': 55}, {'x': 50, 'y': 52},
    {'x': 55, 'y': 60}, {'x': 60, 'y': 65}, {'x': 65, 'y': 68},
    {'x': 70, 'y': 70}, {'x': 75, 'y': 78}, {'x': 80, 'y': 82}
]

# Extract x and y values into separate lists
x_values = np.array([d['x'] for d in data])
y_values = np.array([d['y'] for d in data])

# --- Regression Calculation (Least Squares Method) ---
# Using numpy's polyfit for convenience, which calculates the coefficients
# for a polynomial of degree 1 (linear regression).
# It returns [slope, intercept]
slope, intercept = np.polyfit(x_values, y_values, 1)

# Generate y-values for the regression line
# We create a range of x values from the min to max of our data to draw the line
regression_x = np.array([x_values.min(), x_values.max()])
regression_y = slope * regression_x + intercept

# --- Adding different complexity lines ---
# Create a smoother range of x values for drawing the functions
x_func_range = np.linspace(x_values.min() - 5, x_values.max() + 5, 500) # Extend range slightly for visualization

# 1. Constant Function: y = C
constant_y = np.full_like(x_func_range, 50) # Example constant value

# 2. Another Linear Function: y = mx + b
linear_m = 0.7
linear_b = 15
linear_y = linear_m * x_func_range + linear_b

# 3. Cubic Function: y = ax^3 + bx^2 + cx + d
# Using example coefficients for a visually distinct cubic curve
cubic_a = 0.0005
cubic_b = -0.05
cubic_c = 3
cubic_d = 10
cubic_y = cubic_a * x_func_range**3 + cubic_b * x_func_range**2 + cubic_c * x_func_range + cubic_d

# --- Plotting ---
plt.figure(figsize=(10, 7)) # Set the figure size for better readability
plt.scatter(x_values, y_values, color='skyblue', label='Data Points', s=100, alpha=0.8, edgecolors='w', linewidths=0.5) # Scatter plot for data points

# Plot the regression line
plt.plot(regression_x, regression_y, color='red', label=f'Linear Regression (y = {slope:.2f}x + {intercept:.2f})', linewidth=2)

# Plot the additional complexity lines
plt.plot(x_func_range, constant_y, color='green', linestyle='--', label='Constant Function (y = 50)', linewidth=2)
plt.plot(x_func_range, linear_y, color='purple', linestyle=':', label=f'Another Linear Function (y = {linear_m:.2f}x + {linear_b:.2f})', linewidth=2)
plt.plot(x_func_range, cubic_y, color='orange', linestyle='-.', label='Cubic Function', linewidth=2)


# --- Customizing the Plot ---
plt.title('Linear Regression Model', fontsize=26, pad=15)
plt.xlabel('X-axis (Independent Variable)', fontsize=22, labelpad=10)
plt.ylabel('Y-axis (Dependent Variable)', fontsize=22, labelpad=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=20, loc='upper left') # Adjust legend location to avoid overlap
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.savefig("linear_regression.png", dpi=300)
plt.show() # Display the plot
