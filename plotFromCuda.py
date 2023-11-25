import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import pandas as pd
import string
from matplotlib.legend_handler import HandlerTuple
# from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(threshold=sys.maxsize)

dim = 43 # 4
n = 413 # 449

# Define the path to your binary file
file_path = sys.argv[1]  # Update with your file path
if (len(sys.argv) > 2):
    color_path = sys.argv[2]
    try:
        cf = pd.read_csv(color_path, sep=',', header=None)
        color = cf.values / 19
        # color = np.fromfile(color_path, dtype=np.float32)
        color = color[0:n]
        print("color size: ", color.shape)
        # print(matrix)
    except FileNotFoundError:
        print(f"File '{color_path}' not found.")
        exit(1)
    except ValueError:
        print(f"Error reading the file '{color_path}'. Ensure the shape matches the data size.")
        exit(1)
    filename = 'bioEmbedColored.pickle'
    with open(filename, 'wb') as file:
        pickle.dump(color,file)
print("color ", color)

if (len(sys.argv) > 3):
    label_path = sys.argv[3]
    try:
        df = pd.read_csv(label_path, sep=',', header=None)
        print(df.values)
        labels = np.ndarray.tolist(df.values)
        # label = np.genfromtxt(label_path, delimiter=',')
        print("label ", labels)
        labels = labels[0:n]
        # print("label size: ", labels.shape)
        # print(matrix)
    except FileNotFoundError:
        print(f"File '{label_path}' not found.")
        exit(1)
    except ValueError:
        print(f"Error reading the file '{label_path}'. Ensure the shape matches the data size.")
        exit(1)
    filename = 'bioEmbedLabel.pickle'
    with open(filename, 'wb') as file:
        pickle.dump(labels,file)
    # print("labels ", labels)
    flat_list = []
    for sublist in labels:
        for item in sublist:
            flat_list.append(item)
    labels = flat_list
    # print("labels ", labels)

# Define the shape of the matrix (rows, columns)
# Update these values with the actual dimensions of your matrix
if (n % 8 != 0):
    nAdjust = n + 8 - (n % 8)
else:
    nAdjust = n
matrix_shape = (nAdjust, nAdjust-1)  # Example shape

# Read the binary file into a NumPy array
try:
    matrix = np.fromfile(file_path, dtype=np.double).reshape(matrix_shape)
    matrix = matrix[0:n,:]
    print("matrix size: ", matrix.shape)
    print(matrix)
    filename = 'bioEmbed.pickle'
    matrix = matrix - np.mean(matrix, axis=0)
    print("mean ", np.mean(matrix, axis=0))
    print("matrix centered ", matrix)
    maxVal = np.max(np.max(matrix))
    matrix = matrix / maxVal
    with open(filename, 'wb') as file:
        pickle.dump(matrix[:,0:dim],file)
except FileNotFoundError:
    print(f"File '{file_path}' not found.")
    exit(1)
except ValueError:
    print(f"Error reading the file '{file_path}'. Ensure the shape matches the data size.")
    exit(1)

# # Check if the matrix has at least three dimensions
# if len(matrix.shape) < 3:
#     print("The matrix does not have at least three dimensions.")
#     exit(1)

# Extract the first three dimensions of the matrix
dim1 = matrix[:, 0]
dim2 = matrix[:, 1]
dim3 = matrix[:, 2]
print("dim1 ", dim1)
print("dim3 ", dim3)

# label_color_mapping = {}
# for i, label in enumerate(labels):
#     print("enum label ", label)
#     label = label[0]
#     if label not in label_color_mapping:
#         label_color_mapping[label] = color[i]

# Create a 3D scatter plot for the first three dimensions
fig = plt.figure()
# ax = plt.axes(projection ="3d")
ax = fig.add_subplot(projection='3d') 

# unique_labels = set(labels)  # Create a set of unique labels

# # Create a dictionary to map unique labels to their corresponding colors
# label_color_mapping = {}
# for i, label in enumerate(unique_labels):
#     label_color_mapping[label] = color[labels.index(label)]

# # Create the scatter plot with unique labels and their respective colors
# for label in unique_labels:
#     plt.scatter(dim1[label.index(label)], dim2[label.index(label)], dim3[label.index(label)], label=label, c=label_color_mapping[label])

# # Create the scatter plot with unique labels and their respective colors
# for label, color in label_color_mapping.items():
#     plt.scatter(dim1[label.index(label)], dim2[label.index(label)], dim3[label.index(label)], label=label, c=color)



legend_entries = {}

# Create the scatter plot with unique labels and their respective colors
for i, label in enumerate(labels):
    # print("color i ", color[i])
    c = (float(color[i]*5) % 1.0, float(abs(color[i]*5-0.50) % 1.0), float(abs(color[i]*5-0.75)) % 1.0, 0.3)
    if label not in legend_entries:
        legend_entries[label] = ax.scatter(dim1[i], dim2[i], dim3[i], c=c, label=label)
    else:
        ax.scatter(dim1[i], dim2[i], dim3[i], c=c)


# Create a custom legend based on the unique labels and colors
handles = [legend_entries[label] for label in legend_entries]
plt.legend(handles=handles)




# if (len(sys.argv) > 2):
#     print("plot color label")
#     ax.scatter(dim1, dim2, dim3, c=color, label=label, marker='o')
#     ax.legend()
# else:
#     ax.scatter(dim1, dim2, dim3, c='r', marker='o')

# Set labels for the axes
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')

# Show the plot
plt.show()