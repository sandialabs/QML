import numpy as np
import matplotlib.pyplot as plt

datafile = "QML/QML_test_input.dat.csv"
colorfile = "swissColored.pickle"
data = np.genfromtxt(datafile, delimiter=',')
colors = np.load(colorfile, allow_pickle=True)
print(colors)

# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111)
# ax.scatter(data[:,0], data[:,1], c=colors, cmap=plt.cm.Spectral)
# plt.show()

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111,projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2], c=colors, cmap=plt.cm.Spectral)
plt.show()