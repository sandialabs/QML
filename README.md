[![Python package](https://github.com/sandialabs/sbovqaopt/actions/workflows/python-package.yml/badge.svg)](https://github.com/sandialabs/sbovqaopt/actions/workflows/python-package.yml)

# QML: Implementation of quantum manifold learning (QML)

The `QML` package provides an implementation of manifold learning via quantum dynamics as introduced in [arXiv:2112.11161](https://arxiv.org/abs/2112.11161). This is an implementation without any parallelization.


## Installation

The `QML` package only consists of one file qml_serial.py, which contains all functions required.

## Usage

The QML code is executed on data encoded stored in a CSV file. Each row of the file is parsed as a datapoint.
The code is called with a input file that specifies all parameters, e.g.,

```
qml_serial.py QML_test_input.data
```

For an example input file and required parameters see `QML_test_input.dat`. Also, refer to psuedocode in Appendix VI of [arXiv:2112.11161](https://arxiv.org/abs/2112.11161) for interpretation of code and parameters.

The outputs of QML are:
    - Saves geodesic distance matrix to file "f.out", where "f" is the input file name
    - Optionally, also plots an embedding of the graph if SHOW_EMBEDDING = 2 or 3 (this number sets the embedding dimension) in the input file

## Development

For development purposes, the package can be obtained by cloning the repository locally:

```
git clone https://github.com/sandialabs/QML.git
```

## Citation

If you use or refer to this project in any publication, please cite the corresponding paper:

> Akshat Kumar, Mohan Sarovar. _Manifold learning via quantum dynamics._ [arXiv:2112.11161](https://arxiv.org/abs/2112.11161) (2022).
