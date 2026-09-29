
# Mt. Fuji Gradient Descent Assignment

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Mt. Fuji elevation data


csv_path = "mtfuji_data.csv"

np.set_printoptions(suppress=True)

fuji = np.loadtxt(
    csv_path,
    delimiter=",",
    skiprows=1
)

# Extract point numbers and elevations
point_numbers = fuji[:, 0]
elevation = fuji[:, 3]

# Problem 1: Data visualization


plt.figure(figsize=(10, 5))

plt.plot(point_numbers, elevation)

plt.xlabel("Point Number")
plt.ylabel("Elevation (m)")
plt.title("Cross-Section of Mt. Fuji")

plt.grid(True)
plt.show()


# Problem 2: Calculate the gradient

def calculate_gradient(current_point):
    """
    Calculate the gradient using the backward difference.

    The gradient is calculated from the current point
    and the previous point.
    """

    # Point 0 has no previous point
    if current_point == 0:
        return 0


    # Current point information
    current_elevation = fuji[current_point, 3]
    current_x = fuji[current_point, 0]


    # Previous point information
    previous_elevation = fuji[current_point - 1, 3]
    previous_x = fuji[current_point - 1, 0]


    # Backward difference
    gradient = (
        (current_elevation - previous_elevation)
        / (current_x - previous_x)
    )


    return gradient


# Problem 3: Calculate the destination point


def calculate_destination(current_point, alpha=0.2):
    """
    Calculate the next point using gradient descent.

    destination = current_point - alpha * gradient
    """

    # Calculate the current gradient
    gradient = calculate_gradient(current_point)


    # Apply the gradient descent formula
    destination = current_point - alpha * gradient


    # Round to an integer because point numbers are integers
    destination = int(round(destination))


    # Keep the point inside the valid dataset range
    destination = max(
        0,
        min(destination, len(fuji) - 1)
    )


    return destination


# Problem 4: Go down Mt. Fuji

def descend_mountain(start_point, alpha=0.2):
    """
    Descend Mt. Fuji from a specified starting point.

    The function stops when:
    - the next point is the same as the current point, or
    - the next point has already been visited.
    """

    # Store every point visited during the descent
    path = [start_point]

    # Keep track of visited points to detect loops
    visited = {start_point}

    # Start from the specified point
    current_point = start_point

    while True:

        # Calculate the next destination
        next_point = calculate_destination(
            current_point,
            alpha
        )

        # Stop if there is no movement
        # or if a previously visited point is reached
        if next_point == current_point or next_point in visited:
            break

        # Record the new point
        visited.add(next_point)
        path.append(next_point)

        # Move to the new point
        current_point = next_point

    return path


# Problem 5: Visualize descent from point 136

path_136 = descend_mountain(136)


# Convert the path to NumPy arrays
path_points = np.array(path_136)
path_elevations = fuji[path_points, 3]


print("Problem 4 result")
print("----------------")
print("Starting point:", 136)
print("Final point:", path_136[-1])
print("Final elevation:", fuji[path_136[-1], 3], "m")
print("Number of movements:", len(path_136) - 1)


# Plot the descent
plt.figure(figsize=(10, 5))

plt.plot(
    point_numbers,
    elevation,
    label="Mt. Fuji elevation"
)

plt.scatter(
    path_points,
    path_elevations,
    label="Descent path"
)

plt.xlabel("Point Number")
plt.ylabel("Elevation (m)")
plt.title("Gradient Descent from Point 136")

plt.grid(True)
plt.legend()
plt.show()

# Problem 6: Change the initial value

starting_points = [100, 136, 142, 200]


print("\nProblem 6 results")
print("-----------------")

for start_point in starting_points:

    path = descend_mountain(start_point)

    final_point = path[-1]

    final_elevation = fuji[final_point, 3]

    print("Starting point:", start_point)
    print("Final point:", final_point)
    print("Final elevation:", final_elevation, "m")
    print("Number of movements:", len(path) - 1)
    print()


# Problem 7: Visualize different initial values


plt.figure(figsize=(10, 5))

plt.plot(
    point_numbers,
    elevation,
    label="Mt. Fuji elevation"
)


for start_point in starting_points:

    path = descend_mountain(start_point)

    path_points = np.array(path)

    path_elevations = fuji[path_points, 3]

    plt.scatter(
        path_points,
        path_elevations,
        label=f"Start {start_point}"
    )


plt.xlabel("Point Number")
plt.ylabel("Elevation (m)")
plt.title("Gradient Descent from Different Initial Points")

plt.grid(True)
plt.legend()
plt.show()


# Problem 8: Change the learning rate


learning_rates = [0.1, 0.2, 0.5, 1.0]

start_point = 136


# Visualize the effect of different learning rates
plt.figure(figsize=(10, 5))

plt.plot(
    point_numbers,
    elevation,
    label="Mt. Fuji elevation"
)


for alpha in learning_rates:

    path = descend_mountain(
        start_point,
        alpha
    )

    path_points = np.array(path)

    path_elevations = fuji[path_points, 3]

    plt.scatter(
        path_points,
        path_elevations,
        label=f"α = {alpha}"
    )


plt.xlabel("Point Number")
plt.ylabel("Elevation (m)")
plt.title("Effect of Learning Rate on Gradient Descent")

plt.grid(True)
plt.legend()
plt.show()


# Problem 8: Numerical comparison of learning rates


print("\nProblem 8 results")
print("-----------------")
print("Learning Rate Comparison")
print()


for alpha in learning_rates:

    path = descend_mountain(
        136,
        alpha
    )

    final_point = path[-1]

    final_elevation = fuji[final_point, 3]

    movements = len(path) - 1


    print(f"Alpha: {alpha}")
    print(f"Final point: {final_point}")
    print(f"Final elevation: {final_elevation:.2f} m")
    print(f"Number of movements: {movements}")
    print()
