import numpy as np
import pickle
import csv
import pandas as pd
import sklearn as sk
import sklearn.datasets
import matplotlib.pyplot as plt

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
X, color = sk.datasets.make_swiss_roll(400, random_state=10)
matrix = np.array(X)
matrix[:,1] = matrix[:,1] / 5
filename = 'swiss.pickle'
with open(filename, 'wb') as file:
    pickle.dump(matrix,file)
filename = 'swissColored.pickle'
with open(filename, 'wb') as file:
    pickle.dump(color,file)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111,projection='3d')
ax.scatter(matrix[:,0], matrix[:,1], matrix[:,2], c=color, cmap=plt.cm.Spectral)
plt.show()