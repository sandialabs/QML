import numpy as np
import pickle
import csv
import pandas as pd
import sklearn as sk
import sklearn.datasets
import matplotlib.pyplot as plt
import gzip
import scipy as sp

def read_mnist_images(filename):
    with gzip.open(filename, 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    return x_train
    # with open(filename, 'rb') as file:
    #     data = np.frombuffer(file.read(), dtype=np.uint8, offset=16)
    # # return data.reshape(-1, 28 * 28)
    # return data

def read_mnist_labels(filename):
    with gzip.open(filename, 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    return y_train
    # with open(filename, 'rb') as file:
    #     data = np.frombuffer(file.read(), dtype=np.uint8, offset=8)
    # return data


# # pickle example
# # matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
# # matrix = np.array([[1+2j, 2-4j], [5+6j, 7-8j]])
# # matrix = np.array([[1+2j, np.nan], [5+6j, 7-8j]])
# filename = 'matrix.pickle'
# with open(filename, 'wb') as file:
#     pickle.dump(matrix,file)
# print("Matrix saved to", filename)


# # complex csv example
# matrix = np.array([[1+2j, 2-4j], [5+6j, 7-8j]])
# file_path = "complexMat.csv"
# with open(file_path, "w" , newline="") as csvfile:
#     writer = csv.writer(csvfile,delimiter=",")
#     for row in matrix:
#         writer.writerow([element.real for element in row] + [element.imag for element in row]) 

# # panda pickle
# matrix = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
# filename = 'panda.pickle'
# matrix.to_pickle(filename)

# sklearn swiss roll
X, color = sk.datasets.make_swiss_roll(2000, random_state=10)
matrix = np.array(X)
minX = np.min(matrix[:,0])
minY = np.min(matrix[:,1])
minZ = np.min(matrix[:,2])
matrix[:,0] -= minX
matrix[:,1] -= minY
matrix[:,2] -= minZ
maxX = np.max(matrix[:,0])
maxY = np.max(matrix[:,1])
maxZ = np.max(matrix[:,2])
matrix[:,0] /= maxX
matrix[:,1] /= maxY
matrix[:,2] /= maxZ
# matrix[:,1] /= 10

# print("start",matrix)
# print(color)
# #sorting
# sortMat = np.zeros([matrix.shape[0],matrix.shape[1]+1])
# sortMat[:,0] = color
# sortMat[:,1:sortMat.shape[1]] = matrix
# print("sort", sortMat)
# np.sort(sortMat, axis=0)
# sortMat = matrix[color.argsort(),:]
# print("sorted",sortMat)
matrix = matrix[color.argsort(),:]
color = np.sort(color)

from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=10)
neigh.fit(matrix)
print(neigh.kneighbors(matrix))
print(matrix)
print(color)


filename = 'swiss2000.pickle'
with open(filename, 'wb') as file:
    pickle.dump(matrix,file)
filename = 'swiss2000Colored.pickle'
with open(filename, 'wb') as file:
    pickle.dump(color,file)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111,projection='3d')
ax.scatter(matrix[:,0], matrix[:,1], matrix[:,2], c=color, cmap=plt.cm.Spectral)
plt.show()


# # sklearn mnist
# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split
# digits = load_digits()
# X_train, X_test, y_train, y_test = train_test_split(
#     digits.data, digits.target, stratify=digits.target, random_state=42
# )
# data = X_train
# color = y_train

# filename = 'mnist.pickle'
# with open(filename, 'wb') as file:
#     pickle.dump(data,file)
# filename = 'mnistColored.pickle'
# with open(filename, 'wb') as file:
#     pickle.dump(color,file)
# # stats
# count = data.shape[0]
# print(count)
# norms = np.zeros([count,count])
# for i in range(0,count):
#     for j in range(0,count):
#         norms[i,j] = np.linalg.norm(data[i,:]-data[j,:])
# print(norms)
# print(np.mean(norms))
# print(np.std(norms))




# # fashion mnist
# # import tensorflow as tf
# # from keras.datasets import fashion_mnist

# # Load the Fashion MNIST dataset
# # (x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
# # (x_train, y_train) = tf.datasets.image_classification.FashionMNIST
# # (x_train, y_train), (testX, testy) = fashion_mnist.load_data()
# # tf.keras.datasets.fashion_mnist.load_data()

# dataFile = "/Users/pnooste/Documents/extra/data/fashion-mnist/data/fashion/" + "train-images-idx3-ubyte.gz"
# labelFile = "/Users/pnooste/Documents/extra/data/fashion-mnist/data/fashion/" + "train-labels-idx1-ubyte.gz"

# x_train = read_mnist_images(dataFile)
# y_train = read_mnist_labels(labelFile)

# print(x_train.shape)
# print(x_train)
# print(y_train.shape)
# print(y_train)

# # Select 1000 random samples
# random_indices = np.random.choice(len(x_train), size=60000, replace=False)
# x_samples = x_train[random_indices]
# y_samples = y_train[random_indices]

# # Reshape the images to 1D arrays
# x_samples = x_samples.reshape(-1, 28 * 28)

# # # Load the Fashion MNIST dataset
# # (x_train, y_train), _ = np.load('fashion_mnist.npy', allow_pickle=True)

# # # Select 1000 random samples
# # random_indices = np.random.choice(len(x_train), size=1000, replace=False)
# # x_samples = x_train[random_indices]
# # y_samples = y_train[random_indices]

# # # Reshape the images to 1D arrays
# # x_samples = x_samples.reshape(-1, 28 * 28)

# # # Write samples to file
# # with open('fashion_mnist_samples.txt', 'w') as file:
# #     for i in range(len(x_samples)):
# #         image = x_samples[i]
# #         label = y_samples[i]
# #         file.write(f'Label: {label}\n')
# #         file.write(','.join(str(pix) for pix in image) + '\n')
# #         file.write('\n')
# print(x_samples.shape)
# print(x_samples)
# print(y_samples.shape)
# print(y_samples)
# x_samples = x_samples / 255

# filename = 'fullNist.pickle'
# with open(filename, 'wb') as file:
#     pickle.dump(x_samples,file)
# filename = 'fullNistColored.pickle'
# with open(filename, 'wb') as file:
#     pickle.dump(y_samples,file)




# # biology data
# datafile = "/Users/pnooste/Documents/extra/data/PHATE/data/TreeData.mat"

# dict = sp.io.loadmat(datafile)
# # items = dict.items()
# # data = np.array(items)
# data = dict["M"]
# colors = dict["C"]
# print(".mat debug")
# print(data.shape)
# print(colors.shape)
# print(dict["C"])
# print(dict["M"])
# # print(data)

# filename = 'treeData.pickle'
# with open(filename, 'wb') as file:
#     pickle.dump(data,file)
# filename = 'treeDataColored.pickle'
# with open(filename, 'wb') as file:
#     pickle.dump(colors,file)