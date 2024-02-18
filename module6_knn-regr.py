import numpy as np

def knn_regression(n, k, points, x):

    # Convert list of points to numpy array for efficient computation
    points_array = np.array(points)
    x_array = points_array[:, 0]  # Extract x-values
    y_array = points_array[:, 1]  # Extract y-values
    
    # Calculate Euclidean distances
    distances = np.sqrt((x - x_array) ** 2)

    ## Get the indices of the k nearest neighbors
    nearest_indices = np.argsort(distances)[:k]

    ## Calculate the average y-value of the k nearest neighbors
    y_pred = np.mean(y_array[nearest_indices])
    
    return y_pred


def main():

    # Ask the user for input N (positive integer) and reads it.
    n = int(input("Please input a positive integer N: "))
    print(f"N = {n}")

    # Ask the user for input k (positive integer) and reads it.
    k = int(input("Please input a positive integer k: "))
    print(f"k = {k}")

    # Ask the user to provide N (x, y) points (one by one)
    # Read all of them: first: x value, then: y value for every point one by one. 
    # X and Y are the real numbers.
    points = []
    print(f"\nPlease input {n} (x, y) points one by one.")
    for i in range(n):
        xi = float(input(f"Enter x for point {i + 1}: "))
        yi = float(input(f"Enter y for point {i + 1}: "))
        points.append((xi, yi))
    
    # Ask the user for input X 
    x = float(input("Please input X: "))
    print(f"X = {x}")

    # Perform k-NN Regression
    ## Ensure k is less than or equal to N
    if k > n:
        print("Error: k cannot be greater than N.")
        return
    
    ## If k <= N, perform k-NN Regression
    y_pred = knn_regression(n, k, points, x)
    ## Output the result
    print(f"The predicted Y value for X={x} using {k}-NN Regression is: {y_pred}")



if __name__ == "__main__":
    main()
