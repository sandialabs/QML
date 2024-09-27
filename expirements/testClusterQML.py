import csv
import numpy as np

def read_matrix_from_csv(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            matrix.append([float(val) for val in row])
    return np.array(matrix)

def read_matrix_from_edge_list(file_path):
    matrix = []
    
    # Read the edge list file
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # Iterate through each line
        for line in lines:
            # Split values by spaces and convert them accordingly
            row, col, weight = map(float, line.split())
            
            # Assuming indices start from 1, you can subtract 1 to convert to 0-based indices
            row, col = int(row) - 1, int(col) - 1
            
            # Extend matrix if needed
            while len(matrix) <= max(row, col):
                matrix.append([])
            
            # Ensure the row list is long enough
            while len(matrix[row]) <= col:
                matrix[row].append(0.0)
            
            # Assign the weight to the corresponding position in the matrix
            matrix[row][col] = weight
    # Check and ensure all rows have the same length
    max_row_length = max(len(row) for row in matrix)
    for row in matrix:
        while len(row) < max_row_length:
            row.append(0.0)
    
    # Convert the matrix to a NumPy array
    matrix = np.array(matrix)
    return matrix

def compute_matrix_difference(matrix1, matrix2):
    return matrix1 - matrix2

# Replace 'matrix1.csv' and 'matrix2.csv' with the actual file paths of your CSV files
# file_path_matrix1 = '../serial.out'
# file_path_matrix2 = '../cluster.out'
file_path_matrix1 = '../serialAverage.out'
file_path_matrix2 = '../clusterAverage.out'
file_path_matrix3 = '../nystromAverage.out'

# Read matrices from CSV files
matrix1 = read_matrix_from_edge_list(file_path_matrix1)
matrix2 = read_matrix_from_edge_list(file_path_matrix2)
matrix3 = read_matrix_from_edge_list(file_path_matrix3)

# Compute the difference of values
difference_matrix = compute_matrix_difference(matrix1, matrix2)
difference_matrix2 = compute_matrix_difference(matrix1, matrix3)

# Print the result
print("Matrix 1:")
print(matrix1)
print("\nMatrix 2:")
print(matrix2)
print("\nMatrix 3:")
print(matrix3)
print("\nDifference Matrix:")
print(difference_matrix)
print("\nnystrom Difference Matrix:")
print(difference_matrix2)

sumAnd = 0.0
sumOr = 0.0
countAnd = 0
countOr = 0
for i in range(matrix1.shape[0]):
    for j in range(matrix1.shape[0]):
        if (matrix1[i,j] != 0.0 or matrix2[i,j] != 0.0):
            # print(i, j, matrix1[i,j], matrix2[i,j], difference_matrix[i,j])
            sumOr += abs(difference_matrix[i,j])
            countOr += 1
        if (matrix1[i,j] != 0.0 and matrix2[i,j] != 0.0):
            # print(i, j, matrix1[i,j], matrix2[i,j], difference_matrix[i,j])
            sumAnd += abs(difference_matrix[i,j])
            countAnd += 1
print("diff cluster sum1 or and", sumOr, sumAnd)
print("edge count", countOr, countAnd)
sumAnd = 0.0
sumOr = 0.0
countAnd = 0
countOr = 0
for i in range(matrix1.shape[0]):
    for j in range(matrix1.shape[0]):
        if (matrix1[i,j] != 0.0 or matrix3[i,j] != 0.0):
            # print(i, j, matrix1[i,j], matrix3[i,j], difference_matrix2[i,j])
            sumOr += abs(difference_matrix2[i,j])
            countOr += 1
        if (matrix1[i,j] != 0.0 and matrix3[i,j] != 0.0):
            # print(i, j, matrix1[i,j], matrix3[i,j], difference_matrix2[i,j])
            sumAnd += abs(difference_matrix2[i,j])
            countAnd += 1
print("diff nystrom sum2 or and ", sumOr, sumAnd)
print("edge count", countOr, countAnd)