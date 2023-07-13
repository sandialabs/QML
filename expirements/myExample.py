"""
UMAP on the MNIST Digits dataset
--------------------------------

A simple example demonstrating how to use UMAP on a larger
dataset such as MNIST. We first pull the MNIST dataset and
then use UMAP to reduce it to only 2-dimensions for
easy visualisation.

Note that UMAP manages to both group the individual digit
classes, but also to retain the overall global structure
among the different digit classes -- keeping 1 far from
0, and grouping triplets of 3,5,8 and 4,7,9 which can
blend into one another in some cases.
"""
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

datafile = "swiss.pickle"
try:
    data = np.load(datafile, allow_pickle=True)
except:
    data = pd.read_pickle(datafile)
    data = data.to_numpy()
    print("panda")

colorfile = "swissColored.pickle"
color = np.load(colorfile, allow_pickle=True)


# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split
# digits = load_digits()
# X_train, X_test, y_train, y_test = train_test_split(
#     digits.data, digits.target, stratify=digits.target, random_state=42
# )
# data = X_train
# color = y_train
# print(data)
# print(color)
# print(data.shape)
# print(color.shape)


# UMAP
sns.set(context="paper", style="white")

# mnist = fetch_openml("mnist_784", version=1)

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(data)

fig, ax = plt.subplots(figsize=(8, 6))
# color = mnist.target.astype(int)
plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=10)
plt.setp(ax, xticks=[], yticks=[])
plt.title("UMAP mnist", fontsize=18)

plt.show()



# sklearn laplacian eigenmaps
from sklearn.manifold import SpectralEmbedding
# apply spectral embedding with output dimension = 2
model = SpectralEmbedding(n_components=2, n_neighbors=45)
proj = model.fit_transform(data)
print(proj)

# plot the spectral embedding
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'

fig = plt.figure(figsize=(8,6))

plt.scatter(proj[:, 0], proj[:, 1], c=color, cmap="Spectral", s=10)
plt.title('Laplacian Eigenmap of mnist (sklearn)', size=16)
plt.legend(loc='upper left', fontsize=18)
plt.show()




# sklearn t-sne
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(data)
# apply spectral embedding with output dimension = 2
print(X_embedded)
print(X_embedded.shape)

# plot the spectral embedding
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'

fig = plt.figure(figsize=(8,6))

plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=color, cmap="Spectral", s=10)
plt.title('T-SNE of mnist (sklearn)', size=16)
plt.legend(loc='upper left', fontsize=18)
plt.show()



# diffusion map
# sklearn t-sne
from pydiffmap import diffusion_map as dm
from pydiffmap import kernel as kr
# dmap = dm.DiffusionMap(n_evecs=2, k=10, epsilon=0.5, alpha=0.5)
kernel = kr.Kernel(epsilon=0.5, k=10)
kernel.fit(data)
# kernel.compute(Y=data)
dmap = dm.DiffusionMap(kernel_object=kernel, n_evecs=3, alpha=0.5)
embeded = dmap.fit_transform(data)
# embeded = dmap.embedding_
print(embeded)
print(embeded.shape)

# plot the spectral embedding
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'

fig = plt.figure(figsize=(8,6))

# plt.scatter(embeded[:, 0], embeded[:, 1], c=color, cmap="Spectral", s=10)
plt.scatter(embeded[:, 0], embeded[:, 1], embeded[:,2], c=color, cmap="Spectral", s=10)
plt.title('Diffusion Map of mnist (pydiffmap)', size=16)
plt.legend(loc='upper left', fontsize=18)
plt.show()