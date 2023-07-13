import numpy as np
import matplotlib.pyplot as plt
import sys

np.set_printoptions(threshold=sys.maxsize)

datafile = "../QML_test_input.dat.csv"
colorfile = "treeDataColored.pickle"
aMatfile = "../QML_test_input.dat.out"
data = np.genfromtxt(datafile, delimiter=',')
A = np.genfromtxt(aMatfile, delimiter=',')
colors = np.load(colorfile, allow_pickle=True)
print(colors)

# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111)
# ax.scatter(data[:,0], data[:,1], c=colors, cmap=plt.cm.Spectral)
# plt.show()

# edge list
count = np.count_nonzero(A) / 2
edgeList = np.zeros([int(count),2])
place = 0
maxDiff = 0
for i in range(0,A.shape[0]):
    for j in range(i+1,A.shape[1]):
        if (A[i,j] != 0.0):
            edgeList[place,:] = [i, j]
            place += 1
            if (abs(i - j) > maxDiff):
                maxDiff = j - i
print(edgeList)
print(A.shape)
print(maxDiff)

fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111)
ax = fig.add_subplot(111,projection='3d')
# ax.scatter(data[:,0], data[:,1], c=colors, cmap=plt.cm.Spectral)
ax.scatter(data[:,0], data[:,1], data[:,2], c=colors, cmap=plt.cm.Spectral)
plt.show()