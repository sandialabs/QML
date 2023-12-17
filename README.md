[![Python package](https://github.com/sandialabs/sbovqaopt/actions/workflows/python-package.yml/badge.svg)](https://github.com/sandialabs/sbovqaopt/actions/workflows/python-package.yml)

# QML: Implementation of quantum manifold learning (QML)

The `QML` package provides an implementation of manifold learning via quantum dynamics as introduced in [[SLoD](#slod)] and [[QCC](#qcc)]. This is an implementation without any parallelization.

## Installation

The `QML` package only consists of one file qml_serial.py, which contains all functions required.

## Usage

The QML code is executed on data stored in a CSV, h5, sql, xlsx, or json file. The data must just store a matrix as more complex formats are not processed.
The code is called with a input file that specifies all parameters, e.g.,

```
python3 qml_serial.py QML_test_input.data
```

For an example input file and required parameters see `QML_test_input.dat`. Some useful parameters are described below. For a full discribtion refer to psuedocode in Appendix VI of [[SLoD](#slod)] for interpretation of code and parameters.

* `logepsilon` : This sets the scale length (`logepsilon` $= 2 \log \sqrt{\epsilon}$) at which the $\sqrt{\epsilon}$-nearest-neighbour graph, and hence the graph Laplacian, is constructed from the dataset.
* `alpha` : This scales with `logepsilon` to determine the error term $h$. This term bounds the error on the geodesics found through propagation.
* `dt` : Determines time step size for propagating geodesics. Paths are found at an unit speed, so time step is equivilant to distance.
* `nProp` : Number of time steps taken. Taking more steps along geodesics increases the number of paths found between points as more distance is covered.
* `nColl` : Number of initial momenta for geodesics. Each starting momentum is a direction towards its "nColl" nearest neighbors. Increasing this parameter will start the propagation in more directions.
* `H_test` : Boolean to autmatically tune error terms. The optimal values may not be the minimum found from this test, but it often close to the minimum.

The outputs of QML are:
    - Saves geodesic distance matrix to file "f.out", where "f" is the input file name
    - Optionally, also plots an embedding of the graph if SHOW_EMBEDDING = 2 or 3 (this number sets the embedding dimension) in the input file

## Development

For development purposes, the package can be obtained by cloning the repository locally:

```
git clone https://github.com/sandialabs/QML.git
```

## Manfold learning comparison
For the [Swissroll dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_swiss_roll.html) of 2000 points, QML is better able to *unroll* the dataset than other manifold learning techniques, when embedding into three dimensions. QML works by approximating wave dynamics on the dataset. Using this, a wavepacket localized at a sample point, with large oscillations in a prescribed direction, can be propagated to reveal the geodesic flow on the manifold. The hitting time between points gives a sparse graph of the data that can be embedded, giving a low-dimensional representation; here, we use the method of [1](#oostema) for the embedding.

We compare this with other popular graph embedding techniques: UMAP, Laplacian Eigenmaps, Diffusion Maps and $t$-SNE.

- [UMAP](https://umap-learn.readthedocs.io/en/latest/) ([3](#umap)) aims to preserve the topological structure of the manifold and approximates it with a *fuzzy topological structure*, which it then embeds into a lower-dimensional Euclidean space. The assumptions are that the data is uniformly distribued on the manifold and that the manifold is locally connected and its Riemannian metric is a scalar matrix in ambient coordinates in some neighbourhood of each point.

- [t-SNE](https://lvdmaaten.github.io/tsne/) ([4](#tsne)) aims to reduce dimensionality by placing Gaussian affinities between points in the high-dimensional ambient space and finding lower-dimensional coordinates that minimize the Kullback-Liebler divergence between the high-dimensional Gaussian distributions and lower-dimensional Student $t$-distributions.

- Laplacian Eigenmaps ([5](#lap-eigmaps)) produces lower dimensional coordinates by using the values of some of the eigenfunctions of the graph Laplacian of a graph on the dataset constructed by connecting points within a certain scale-length based on the density of point samples.

- Diffusion Maps ([6](#diffmaps)) works in the same way as Laplacian Eigenmaps, but scales each eigenfunction that provides coordinates, by a power of the corresponding eigenvalue, corresponding to time of diffusion of the diffusive process underlying the method.

  On the Swiss roll with 2000 uniformly sampled points, QML, UMAP, Laplacian Eigenmaps, and Diffusion Maps all embed the dataset into a roughly flat two-dimensional submanifold of three-dimensional coordinate space. The embeddings produced by Laplacian Eigenmaps and Diffusion Maps *lose* an additional dimension of the ground truth, due to the initial two chosen eigenfunctions having dependence only on one of the directions of the manifold. This is a noted drawback of these methods (see *e.g.*, [7](#eigen-redundancy)).  The embedding by $t$-SNE retains the dimensionality, but the mapping from the original dataset onto the lower dimensional representation is not faithful to the intrinsic local structure of the dataset (note the color-coding of points from one end of the roll to the other and the corresponding $t$-SNE embedding in the figure below). QML and UMAP retain information along the flat dimension of the data. However, only QML maintains a uniform distribution of the data and manages to straighten the Swiss roll, while all the others result in a curved and less uniform representation (see the gaps in the embedding by UMAP in the figure below).


<p align="center">
    <img src="images/MLcompareFull.PNG" alt="drawing" style="width:600px;"/>
</p>

## Geodesic method comparison

Quantum Manifold Learning (QML) computes geodesics using quantum dynamics derived from a diffusion process. The accuracy of the geodesics scales with the amount of data in the input. QML scales to higher-dimensional data more efficiently than common geodesic methods. We compare the resolution of geodesics by QML to the *Heat Method* (our terminology) of [2](#crane-heat) and shortest paths on local neighbourhood graphs. The Heat Method attempts to recover geodesic distances to a given point by approximately solving the heat equation on the dataset with a singular point source as the initial condition, computing its normalized gradient vector field and then approximately solving the Poisson equation with respect to this gradient field. This method needs structured approximations to the manifold, or proceeds by an initial Voronoi cell decomposition from the dataset. Dijkstra's algorithm can also be used on a nearest neighbours graph of Euclidean distances to approximate geodesic distance.

We apply these methods to two sets of discrete point coordinates on the unit sphere in $\mathbb{R}^3$. When applied to a structured approximation to the sphere, which is a natural setting for the Heat Method as described in [2](#crane-heat), QML recovers geodesics distances with roughly the same accuracy as the Heat Method. When points are sampled uniformly at random, the Heat Method is applied to a Voronoi cell decomposition given by a sub-sampling of points, while QML is applied directly on the full, *unstructured* sample set. Again, in this setting, QML and Heat Method recover geodesic distances with similar accuracy. Both methods outperform distances obtained from using Dijkstra's method on a nearest neighbor graph of degree 6. Additionally, QML can output a geodesic path, as shown on the top row of the figure below. This is constructed by propagating a directed and localized wavepacket for multiple time steps and plotting the position along the way.

<p align="center">
    <img src="images/geoCompareFull.PNG" alt="drawing" style="width:600px;"/>
</p>

## Citations

If you use or refer to this project in any publication, please cite the corresponding papers:

> [SLoD] <a id="slod"></a> Akshat Kumar, Mohan Sarovar. _Shining light on data: Geometric data analysis through quantum dynamics_ [arXiv:2212.00682](https://arxiv.org/abs/2212.00682) (2022).

> [QCC] <a id="qcc"></a> Akshat Kumar. *On a quantum-classical correspondence: from graphs to manifolds* [arXiv:2112.10748](https://arxiv.org/abs/2112.10748) (2022).

## References

> 1. <a id="oostema"></a> Peter Oostema, Franz Franchetti. _Leveraging High Dimensional Spatial Graph Embedding as a Heuristic for Graph Algorithms_, [IEEE IPDPSW](https://spiral.ece.cmu.edu/pub-spiral/pubfile/PDCO2021_338.pdf) (2021).

> 2. <a id="crane-heat"></a> Keenan Crane, Clarisse Weischedel, Max Wardetzky. _The Heat Method for Distance Computation_, [CACM](https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/paperCACM.pdf) (2017).

> 3. <a id="umap"></a> Leland McInnes, John Healy, James Melville. *Umap: Uniform manifold approximation and projection for dimension reduction*, [arXiv:1802.03426](https://arxiv.org/abs/1802.03426) (2018).

> 4. <a id="tsne"></a> L.J.P. van der Maaten, G.E. Hinton. *Visualizing High-Dimensional Data Using t-SNE*, [JMLR](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) (2008).

> 5. <a id="lap-eigmaps"></a> Mikhail Belkin, Partha Niyogi. *Laplacian eigenmaps for dimensionality reduction and data representation*, Neural Computation (2003).

> 6. <a id="diffmaps"></a> Ronald Coifman, Stéphane Lafon. *Diffusion maps*, ACHA (2006).

> 7. <a id="eigen-redundancy"></a> Y. Goldberg, A. Zakai, D. Kushnir, Y. Ritov, *Manifold learning: The price of normalization*, JMLR (2008).

