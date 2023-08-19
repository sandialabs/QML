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
import random
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import eigh



def find_diffusion_matrix(X=None, alpha=0.15):
    """Function to find the diffusion matrix P
        
        >Parameters:
        alpha - to be used for gaussian kernel function
        X - feature matrix as numpy array
        
        >Returns:
        P_prime, P, Di, K, D_left
    """
    alpha = alpha
        
    dists = euclidean_distances(X, X)
    K = np.exp(-dists**2 / alpha)
    
    r = np.sum(K, axis=0)
    Di = np.diag(1/r)
    P = np.matmul(Di, K)
    
    D_right = np.diag((r)**0.5)
    D_left = np.diag((r)**-0.5)
    P_prime = np.matmul(D_right, np.matmul(P,D_left))

    return P_prime, P, Di, K, D_left

def find_diffusion_map(P_prime, D_left, n_eign=2):
    """Function to find the diffusion coordinates in the diffusion space
        
        >Parameters:
        P_prime - Symmetrized version of Diffusion Matrix P
        D_left - D^{-1/2} matrix
        n_eigen - Number of eigen vectors to return. This is effectively 
                    the dimensions to keep in diffusion space.
        
        >Returns:
        Diffusion_map as np.array object
    """   
    n_eign = n_eign
    
    eigenValues, eigenVectors = eigh(P_prime)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    
    diffusion_coordinates = np.matmul(D_left, eigenVectors)
    
    return diffusion_coordinates[:,:n_eign]

def apply_diffusions(X=None, alpha_start=0.001, alpha_end= 0.009, title='Diffused points'):
    # d_maps = []
    alpha = 100 # np.linspace(alpha_start, alpha_end, 10)
    P_prime, P, Di, K, D_left = find_diffusion_matrix(X, alpha=alpha)
    d_maps = find_diffusion_map(P_prime, D_left, n_eign=2)
    return d_maps, alpha



datafile = "../data/swiss2000.pickle"
try:
    data = np.load(datafile, allow_pickle=True)
except:
    data = pd.read_pickle(datafile)
    data = data.to_numpy()
    print("panda")

colorfile = "../data/swiss2000Colored.pickle"
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
plt.title("UMAP", fontsize=18)

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
plt.title('Laplacian Eigenmap', size=16)
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
plt.title('T-SNE', size=16)
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
dmap = dm.DiffusionMap(kernel_object=kernel, n_evecs=2, alpha=0.5)
embeded = dmap.fit_transform(data)
# embeded = dmap.embedding_
print(embeded)
print(embeded.shape)

# plot the spectral embedding
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'

fig = plt.figure(figsize=(8,6))

print(embeded.shape)
print(np.array([10]*len(color)).shape)
sizes = np.array([10]*len(color))
print(embeded[:, 0].shape)
plt.scatter(embeded[:, 0], embeded[:, 1], c=color, cmap="Spectral", s=10)
# plt.scatter(embeded[:, 0], embeded[:, 1], embeded[:,2], s=sizes, c=color, cmap="Spectral")
plt.title('Diffusion Map', size=16)
plt.legend(loc='upper left', fontsize=18)
plt.show()

# embeded, alphas = apply_diffusions(data)
# embeded = np.array(embeded)
# print(embeded)
# print(embeded.shape)
# print(embeded[:, 0].shape)
# plt.scatter(embeded[:, 0], embeded[:, 1], c=color, cmap="Spectral", s=10)
# plt.title('Diffusion Map', size=16)
# plt.legend(loc='upper left', fontsize=18)
# plt.show()