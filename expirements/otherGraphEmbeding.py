import networkx as nx
from node2vec import Node2Vec
import numpy as np
from sklearn.manifold import smacof
from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def plotDataColorLabels(X, ax, color, labels, labelColorMapping, cmap):
    for label, colour in labelColorMapping.items():
        tempColors = color[(color==colour).reshape(-1)]
        tempX = X[(color==colour).reshape(-1),:]
        tempLabels = labels[(labels==label)]
        # print("tempX size", tempX.size)
        # print("tempColors size", tempColors.size)
        # print("tempLabels size", tempLabels.size)
        # print("color, ", color.size, color, color[0])
        colorSingle = cmap(np.where(unique_classes == colour[0]))
        # print("colorSingle ", colorSingle, colour)
        colorPrint = np.full((tempX[:,0].shape[0],4), colorSingle)
        # print("X shape", tempX.shape)
        if (tempX.shape[1] > 2):
            ax.scatter(tempX[:,0], tempX[:,1], tempX[:,2], c=colorPrint, label=label)
        else:
            ax.scatter(tempX[:,0], tempX[:,1], c=colorPrint, label=label)
        # ax.scatter(tempX[:,0], tempX[:,1], tempX[:,2], label=label)
    plt.legend()

colorfile = "bioEmbedColored.pickle"
color = np.array(np.load(colorfile, allow_pickle=True))
# plot graph embedding
unique_classes = np.unique(np.array(color))
# Generate a colormap with a different color for each class
num_classes = len(unique_classes)
# print("num_classes", num_classes)
cmap = plt.get_cmap('Spectral', num_classes) # viridis
# print("cmap ", cmap)
labels = np.loadtxt(open('bioGraphLabels.csv', "rb"), delimiter=",", dtype=str)
# print("colors", color, color.shape)
# print("labels", labels, labels.shape)
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
labelColorMapping = {}
for i, label in enumerate(labels):
    if label not in labelColorMapping:
        print("label color, ", label, color[i], i)
        labelColorMapping[label] = color[i]




filename = 'bioGraph.pickle'
data = np.load(filename, allow_pickle=True)
data[data == np.inf] = 0.0
data = np.maximum( data, data.transpose() )
# print("data ", data, data.shape)


# # sk learn smacof
# # Generate synthetic data
# # np.random.seed(42)
# # n_samples = 10
# # original_data = np.random.rand(n_samples, 3)

# # # Compute pairwise Euclidean distances
# # distances = euclidean_distances(original_data)

# distances = np.array(data)

# # Apply SMACOF algorithm for MDS
# mds = smacof(n_components=3, dissimilarities=distances, random_state=42)
# print("smacof output ", mds)
# embedded_data = mds[0]
# # embedded_data = mds.fit_transform(distances)

# # plt.subplot(1, 2, 2)
# plotDataColorLabels(embedded_data, ax, color, labels, labelColorMapping, cmap)

# plt.tight_layout()
# plt.show()
# score = silhouette_score(embedded_data[:,:], color)
# print("scacof silhouette score: ", score)
# scores = np.zeros(unique_classes.shape)
# for i, clas in enumerate(unique_classes):
#     tempColors = np.ones(color.shape) * 2
#     tempColors[color == clas] = 1
#     scores[i] = silhouette_score(embedded_data, tempColors)
# print("silhouette scores", scores)








# node 2 vec
# Create a graph
# graph = data
# graph = nx.fast_gnp_random_graph(n=100, p=0.5)
graph = nx.from_numpy_matrix(data)
print("graph ", graph)

# Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
node2vec = Node2Vec(graph, dimensions=3, walk_length=30, num_walks=200, workers=4)  # Use temp_folder for big graphs

# Embed nodes
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
print("model",  model)

# Visualize the embeddings
# In this example, we'll visualize the embeddings for the first 10 nodes
node_ids = list(graph.nodes())[:]

# Get the embeddings for the selected nodes
embeddings = [model.wv[str(node_id)] for node_id in node_ids]
n = len(node_ids)
print("n ", n, node_ids)
embeddingX = np.zeros((n,3))
for i in range(n):
    embeddingX[i,:] = embeddings[i]
print("node2vec", embeddingX)
print(embeddings[0])
print(embeddings[0][0])
plotDataColorLabels(embeddingX, ax, color, labels, labelColorMapping, cmap)

plt.show()
score = silhouette_score(embeddingX[:,:], color)
print("node2vec silhouette score: ", score)
scores = np.zeros(unique_classes.shape)
for i, clas in enumerate(unique_classes):
    tempColors = np.ones(color.shape) * 2
    tempColors[color == clas] = 1
    scores[i] = silhouette_score(embeddingX, tempColors)
print("silhouette scores", scores)