import os
import h5py
import numpy as np
import pandas as pd

def initialize(inp):
    """
        Initialize parameters

        Inputs:
            inp: dictionary with parameters read from file. Not all parameters may be specified, and need to convert boolean parameters from text string to bool

        Outputs:
            qml_params: parameters dictionary
    """

    qml_params = dict()
    if 'logepsilon' in inp:
        qml_params['logepsilon'] = inp['logepsilon']
    else:
        qml_params['logepsilon'] = -1

    if 'alpha' in inp:
        qml_params['alpha'] = inp['alpha']
    else:
        qml_params['alpha'] = 1.5

    if 'dt' in inp:
        qml_params['dt'] = inp['dt']
    else:
        qml_params['dt'] = 0.1

    if 'nProp' in inp:
        qml_params['nProp'] = inp['nProp']
    else:
        qml_params['nProp'] = 10

    if 'nColl' in inp:
        qml_params['nColl'] = inp['nColl']
    else:
        qml_params['nColl'] = 1

    if 'PCA_PREP' in inp:
        if inp['PCA_PREP'].lower() in ['true', '1', 't']:
            qml_params['PCA_PREP'] = True
        else:
            qml_params['PCA_PREP'] = False
    else:
        qml_params['PCA_PREP'] = False

    if 'PCA_MEAS' in inp:
        if inp['PCA_MEAS'].lower() in ['true', '1', 't']:
            qml_params['PCA_MEAS'] = True
        else:
            qml_params['PCA_MEAS'] = False
    else:
        qml_params['PCA_MEAS'] = False

    if 'PCA_dims' in inp:
        qml_params['PCA_dims'] = inp['PCA_dims']
    else:
        qml_params['PCA_dims'] = 0

    if 'delta_PCA' in inp:
        qml_params['delta_PCA'] = inp['delta_PCA']
    else:
        qml_params['delta_PCA'] = 1.5

    if 'gamma' in inp:
        qml_params['gamma'] = inp['gamma']
    else:
        qml_params['gamma'] = 0.1

    if 'prob_thresh' in inp:
        qml_params['prob_thresh'] = inp['prob_thresh']
    else:
        qml_params['prob_thresh'] = 0

    if 'USE_MAX' in inp:
        if inp['USE_MAX'].lower() in ['true', '1', 't']:
            qml_params['USE_MAX'] = True
        else:
            qml_params['USE_MAX'] = False
    else:
        qml_params['USE_MAX'] = False

    if 'verbose' in inp:
        if inp['verbose'].lower() in ['true', '1', 't']:
            qml_params['verbose'] = True
        else:
            qml_params['verbose'] = False
    else:
        qml_params['verbose'] = False

    if 'SHOW_EMBEDDING' in inp:
        if inp['SHOW_EMBEDDING'].lower() in ['2d', '2']:
            qml_params['SHOW_EMBEDDING'] = 2
        elif inp['SHOW_EMBEDDING'].lower() in ['3d', '3']:
            qml_params['SHOW_EMBEDDING'] = 3
        else:
            qml_params['SHOW_EMBEDDING'] = 0
    else:
        qml_params['SHOW_EMBEDDING'] = 0

    if 'datafile' in inp:
        qml_params['datafile'] = inp['datafile']
    else:
        qml_params['datafile'] = 'data.csv'

    if 'colorfile' in inp:
        qml_params['colorfile'] = inp['colorfile']
    else:
        qml_params['colorfile'] = False

    if 'labelfile' in inp:
        qml_params['labelfile'] = inp['labelfile']
    else:
        qml_params['labelfile'] = False

    if 'H_test' in inp:
        if inp['H_test'].lower() in ['true', '1', 't']:
            qml_params['H_test'] = True
        else:
            qml_params['H_test'] = False
    else:
        qml_params['H_test'] = False

    if 'H_test_avg' in inp:
        qml_params['H_test_avg'] = inp['H_test_avg']
    else:
        qml_params['H_test_avg'] = 20

    return qml_params

def read_in_matrix(datafile, verbose):
    ext = os.path.splitext(datafile)[1]
    # print(ext)
    if ext == ".csv":
        data = np.genfromtxt(datafile, delimiter=',')
    elif ext == ".pkl" or ext == ".pickle" or ext == ".npy":
        try:
            data = np.load(datafile, allow_pickle=True)
        except:
            data = pd.read_pickle(datafile)
            data = data.to_numpy()
            print(data)
            print("panda")
        # data = pd.read_pickle(datafile)
    elif ext == ".hdf" or ext == ".h5":
        # Broken for example file, may be too complicated
        try:
            data = pd.read_hdf(datafile).to_numpy()
        except:
            hf = h5py.File(datafile, 'r')
            data = []
            for i in hf.values():
                data.append(i)
            print(data)
            data = np.array(data)
    elif ext == ".sql":
        data = pd.read_sql(datafile).to_numpy()
    elif ext == ".xlsx":
        data = pd.read_xlsx(datafile).to_numpy()
    elif ext == ".json":
        data = pd.read_json(datafile).to_numpy()
    elif ext == ".html":
        data = pd.read_html(datafile).to_numpy()
    # elif ext == ".mat":
    #     dict = sp.io.loadmat(datafile)
    #     items = dict.items()
    #     data = np.array(items)
    #     print(".mat debug")
    #     print(data.shape)
    #     print(items)
    #     # print(data)
    # elif ext == ".mtx":
    #     data = sp.io.mmread(datafile)
    #     print("mtx")
    #     print(data)
    else:
        print("Cannot parse data file: " + datafile + """. Supported file types
        include .csv, .pickle, .pkl, .hdf, .h5, .sql, .xlsx, .json, and .html.""")
        raise Exception("Unsupported data file")
    # print(data.shape)
    # print(len(data.shape))
    # print(type(data.dtype))
    # print(data)
    if (len(data.shape) > 2):
        print("Only data from tensors of dimension 2 are supported.")
        raise Exception("Unsuppored data")
    
    nan_bools = np.isnan(data)
    if (True in nan_bools):
        if (verbose):
            print("NaN found in input. Removing data points with issue.")
        data = data[~np.isnan(data).any(axis=1), :]

    complex_bools = np.iscomplex(data)
    if (True in complex_bools):
        if (verbose):
            print("Method only take real values. Converting to real matrix.")
        data = np.real(data)
    
    # print(data)
    return data
 