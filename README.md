[![Python package](https://github.com/sandialabs/sbovqaopt/actions/workflows/python-package.yml/badge.svg)](https://github.com/sandialabs/sbovqaopt/actions/workflows/python-package.yml)

# QML: Implementation of quantum manifold learning (QML)

The `QML` package provides an implementation of manifold learning via quantum dynamics as introduced in [arXiv:2112.11161](https://arxiv.org/abs/2112.11161). This is an implementation without any parallelization.

## Installation

The `QML` package only consists of one file qml_serial.py, which contains all functions required.

## Usage

The QML code is executed on data stored in a CSV, h5, sql, xlsx, or json file. The data must just store a matrix as more complex formats are not processed.
The code is called with a input file that specifies all parameters, e.g.,

```
python3 qml_serial.py QML_test_input.data
```

For an example input file and required parameters see `QML_test_input.dat`. Some useful parameters are described below. For a full discribtion refer to psuedocode in Appendix VI of [arXiv:2112.11161](https://arxiv.org/abs/2112.11161) for interpretation of code and parameters.

* `logepsilon` : This is an error term for scaling parameters used in constructing the graph Laplacian.
* `alpha` : This scales with `logepsilon` to determine the error term "h". This term bounds the error on the geodesics found through propagation.
* `dt` : Determines time step size for propagating geodesics. Paths are found at an unit speed, so time step is equivilant to distance.
* `nProp` : Number of time steps taken. Taking more steps along geodesics increase the number of paths found between points as more distance is covered.
* `nColl` : Number of initial momenta for geodesics. Each starting momentum start is a direction towards its "nColl" nearest neighbors. Increasing this parameter will start the propagation in more directions.
* `H_test` : Boolean to autmatically tune error terms. The optimal values may not be the minimum found from this test, but it often close to the minimum.

The outputs of QML are:
    - Saves geodesic distance matrix to file "f.out", where "f" is the input file name
    - Optionally, also plots an embedding of the graph if SHOW_EMBEDDING = 2 or 3 (this number sets the embedding dimension) in the input file

## Development

For development purposes, the package can be obtained by cloning the repository locally:

```
git clone https://github.com/sandialabs/QML.git
```

## Geodesic method comparison

Quantum Manifold Learning (QML) computes geodesics using a diffusion process derived from quantum dynamics. The accuracy of the geodesics scales with the amount of data in the input. QML scales to higher-dimensional data more efficiently than common geodesic methods. When applied to samplings of the sphere, geodesics are recovered with similar accuracy to the heat method. Both methods outperform distances obtained from using Dijkstra's method on a nearest neighbor graph of degree 6.

<p align="center">
    <img src="images/geoCompareFull.PNG" alt="drawing" style="width:500px;"/>
</p>

## Citation

If you use or refer to this project in any publication, please cite the corresponding paper:

> Akshat Kumar, Mohan Sarovar. _Manifold learning via quantum dynamics._ [arXiv:2112.11161](https://arxiv.org/abs/2112.11161) (2022).
