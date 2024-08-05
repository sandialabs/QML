import numpy as np
import matplotlib.pyplot as plt
import sys

np.set_printoptions(threshold=sys.maxsize)

datafile = "C:\\Users\\Peter\\Documents\\school\\graphResearch\\gitRe\\checkgit\\embedP1.bin" #"QML_test_input.dat.csv"
colorfile = "data/swiss2000Colored.pickle"
aMatfile = "QML_test_input.dat.out"
# data = np.genfromtxt(datafile, delimiter=',')
data = np.fromfile(datafile, dtype=np.double) 
A = np.genfromtxt(aMatfile, delimiter=',')
# A = np.fromfile(aMatfile, dtype=np.double) 
print("data shape ", data.shape)
colors = np.load(colorfile, allow_pickle=True)
# print(colors)
data = np.reshape(data, (colors.shape[0], colors.shape[0]-1))
print("data shape ", data.shape)

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
# print(edgeList)
# print(A.shape)
print(maxDiff)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
# ax = fig.add_subplot(111,projection='3d')
ax.scatter(data[:,0], data[:,1], c=colors, cmap=plt.cm.Spectral)
# ax.scatter(data[:,0], data[:,1], data[:,2], c=colors, cmap=plt.cm.Spectral)
plt.show()