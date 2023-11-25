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
from sklearn.metrics import silhouette_score
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

def plotDataColorLabels(X, ax, color, labels, labelColorMapping, cmap):
    for label, colour in labelColorMapping.items():
        tempColors = color[(color==colour).reshape(-1)]
        tempX = X[(color==colour).reshape(-1),:]
        tempLabels = labels[(labels==label)]
        # print("tempX", tempX[:,0:3])
        # print("tempX size", tempX.size)
        # print("tempColors size", tempColors.size)
        # print("tempLabels size", tempLabels.size)
        # print("color, ", color.size, color, color[0])
        colorSingle = cmap(np.where(unique_classes == colour))
        print("colorSingle ", colorSingle, colour)
        colorPrint = np.full((tempX[:,0].shape[0],4), colorSingle)
        print("X shape", tempX.shape, colorPrint.shape, label)
        if (tempX.shape[1] > 2):
            print("plotting")
            ax.scatter(tempX[:,0], tempX[:,1], tempX[:,2], c=colorPrint, label=label)
        else:
            ax.scatter(tempX[:,0], tempX[:,1], c=colorPrint, label=label)
        # ax.scatter(tempX[:,0], tempX[:,1], tempX[:,2], label=label)
    plt.legend()
    

datafile = "bioEmbed.pickle"
try:
    data = np.load(datafile, allow_pickle=True)
except:
    data = pd.read_pickle(datafile)
    data = data.to_numpy()
    print("panda")

colorfile = "bioGraphColored.csv"
# color = np.load(colorfile, allow_pickle=True)
color = np.loadtxt(open(colorfile, "rb"), delimiter=",", dtype=str)


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

score = silhouette_score(data[:,:], color)
print("Graph Embedding full rank silhouette score: ", score)
score = silhouette_score(data[:,1:3], color)
print("Graph Embedding silhouette score: ", score)

# plot graph embedding
unique_classes = np.unique(np.array(color))
# Generate a colormap with a different color for each class
num_classes = len(unique_classes)
# print("num_classes", num_classes)
cmap = plt.get_cmap('Spectral', num_classes) # viridis
# print("cmap ", cmap)
labels = np.loadtxt(open('bioGraphLabels.csv', "rb"), delimiter=",", dtype=str)
print("colors", color, color.shape)
print("labels", labels, labels.shape)
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
labelColorMapping = {}
for i, label in enumerate(labels):
    if label not in labelColorMapping:
        print("label color, ", label, color[i], i)
        labelColorMapping[label] = color[i]
plotDataColorLabels(data, ax, color, labels, labelColorMapping, cmap)
plt.show()

# scores = np.zeros(unique_classes.shape)
# for i, clas in enumerate(unique_classes):
#     tempColors = np.ones(color.shape) * 2
#     tempColors[color == clas] = 1
#     scores[i] = silhouette_score(data, tempColors)
# print("single class silhouette scores", scores)


# # UMAP
# sns.set(context="paper", style="white")
# # mnist = fetch_openml("mnist_784", version=1)

# # UMAP(a=None, angular_rp_forest=False, b=None,
# #      force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
# #      local_connectivity=1.0, low_memory=False, metric='euclidean',
# #      metric_kwds=None, min_dist=0.1, n_components=2, n_epochs=None,
# #      n_neighbors=15, negative_sample_rate=5, output_metric='euclidean',
# #      output_metric_kwds=None, random_state=42, repulsion_strength=1.0,
# #      set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
# #      target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
# #      transform_queue_size=4.0, transform_seed=42, unique=False, verbose=False)
# reducer = umap.UMAP(random_state=42, n_neighbors=4, n_components=3)
# embedding = reducer.fit_transform(data)

# # fig, ax = plt.subplots(figsize=(8, 6))
# # color = mnist.target.astype(int)
# # plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="hsv", s=10)
# plotDataColorLabels(embedding, ax, color, labels, labelColorMapping, cmap)
# # plt.setp(ax, xticks=[], yticks=[])
# plt.title("UMAP", fontsize=18)

# plt.show()
# score = silhouette_score(embedding, color)
# print("UMAP silhouette score: ", score)
# scores = np.zeros(unique_classes.shape)
# for i, clas in enumerate(unique_classes):
#     tempColors = np.ones(color.shape) * 2
#     tempColors[color == clas] = 1
#     scores[i] = silhouette_score(embedding, tempColors)
# print("single class silhouette scores", scores)


# sklearn laplacian eigenmaps
from sklearn.manifold import SpectralEmbedding
# apply spectral embedding with output dimension = 2
model = SpectralEmbedding(n_components=3, n_neighbors=10)
proj = model.fit_transform(data)
print(proj)

# plot the spectral embedding
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'

# fig = plt.figure(figsize=(8,6))

# plt.scatter(proj[:, 0], proj[:, 1], c=color, cmap="Spectral", s=10)
plotDataColorLabels(proj, ax, color, labels, labelColorMapping, cmap)
plt.title('Laplacian Eigenmap', size=16)
# plt.legend(loc='upper left', fontsize=18)
plt.show()

score = silhouette_score(proj, color)
print("Laplacian Eigenmap silhouette score: ", score)
scores = np.zeros(unique_classes.shape)
for i, clas in enumerate(unique_classes):
    tempColors = np.ones(color.shape) * 2
    tempColors[color == clas] = 1
    scores[i] = silhouette_score(proj, tempColors)
print("single class silhouette scores", scores)




# # sklearn t-sne
# from sklearn.manifold import TSNE
# X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(data)
# # apply spectral embedding with output dimension = 2
# print(X_embedded)
# print(X_embedded.shape)

# # plot the spectral embedding
# plt.style.use('default')
# plt.rcParams['figure.facecolor'] = 'white'

# fig = plt.figure(figsize=(8,6))

# plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=color, cmap="Spectral", s=10)
# plt.title('T-SNE', size=16)
# plt.legend(loc='upper left', fontsize=18)
# plt.show()





# diffusion map
from pydiffmap import diffusion_map as dm
from pydiffmap import kernel as kr

# logeps = np.zeros((100,1))
# norm1W = np.zeros((100,1))
# logeps[0] = -2
# for i in range(1,logeps.size):
#     logeps[i] = logeps[i-1] + 0.1
# k = 6
# ji = 0
# for j in logeps:
#     W = np.exp((-k) / np.exp(j) )
#     norm1W[ji] = np.linalg.norm(W,1)
#     ji += 1
# fig = plt.figure(figsize=(8,6))
# plt.plot(logeps, norm1W)
# plt.show()

# dmap = dm.DiffusionMap(n_evecs=2, k=10, epsilon=0.5, alpha=0.5)
kernel = kr.Kernel(epsilon=0.01, k=6)
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

# fig = plt.figure(figsize=(8,6))
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

print(embeded.shape)
print(np.array([10]*len(color)).shape)
sizes = np.array([10]*len(color))
print(embeded[:, 0].shape)
plotDataColorLabels(embeded, ax, color, labels, labelColorMapping, cmap)
# plt.scatter(embeded[:, 0], embeded[:, 1], c=color, cmap="hsv", s=10)
# plt.scatter(embeded[:, 0], embeded[:, 1], embeded[:,2], s=sizes, c=color, cmap="Spectral")
plt.title('Diffusion Map', size=16)
# plt.legend(loc='upper left', fontsize=18)
plt.show()

score = silhouette_score(embeded, color)
print("Diffusion Map silhouette score: ", score)
scores = np.zeros(unique_classes.shape)
for i, clas in enumerate(unique_classes):
    tempColors = np.ones(color.shape) * 2
    tempColors[color == clas] = 1
    scores[i] = silhouette_score(embeded, tempColors)
print("single class silhouette scores", scores)

# embeded, alphas = apply_diffusions(data)
# embeded = np.array(embeded)
# print(embeded)
# print(embeded.shape)
# print(embeded[:, 0].shape)
# plt.scatter(embeded[:, 0], embeded[:, 1], c=color, cmap="Spectral", s=10)
# plt.title('Diffusion Map', size=16)
# plt.legend(loc='upper left', fontsize=18)
# plt.show()