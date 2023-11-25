# File: qml_serial.py
#
# ===========================================================
# Implementation of quantum manifold learning (QML), with  no parallelization.
# Reference:
#   "Manifold Learning via Quantum Dynamics". Akshat Kumar & Mohan Sarovar
#   arXiv:2112.11161  https://arxiv.org/abs/2112.11161
#
# ===========================================================
# Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
# 

import time
import os
import sys
import json
import numpy as np
import scipy.spatial as spatial
from scipy.sparse.linalg import inv as spinv
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import igraph as ig
import pandas as pd
import h5py
import scipy as sp

# np.set_printoptions(threshold=sys.maxsize)

# ------------------------------------
# Functions
# ------------------------------------
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
    print(ext)
    if ext == ".csv":
        data = np.genfromtxt(datafile, delimiter=',')
    elif ext == ".pkl" or ext == ".pickle" or ext == ".npy":
        try:
            data = np.load(datafile, allow_pickle=True)
        except:
            data = pd.read_pickle(datafile)
            data = data.to_numpy()
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
    
def PCA_for_ts(data, pt, no_dims):
    """
    Perform local PCA around a point to estimate tangent space

    Inputs:
        data: dataset (an NxM matrix)
        pt: the index of the point around which to perform the local PCA
        no_dims: the number of dimensions to truncare the local PCA (the local tangent space dimension)
    Outputs:
        mappedX: data points in PCA coordinates
        mapping: PCA mapping
    """

    K = np.shape(data)[0]

    # center data
    X = np.squeeze(data - data[pt,])

    # calculate covariance matrix
    M = (1/K) * (np.transpose(X) @ X)

    lam, v = sp.linalg.eig(M)
    idx = np.argsort(lam) # sorted in ascending order
    idx = idx[::-1] # reverse order to get descending eigenvalues
    lam = np.real(lam[idx])
    v = v[:,idx]

    if no_dims<1:
        g = [i for i, e in enumerate(np.cumsum(lam/np.sum(lam))) if e>no_dims]
        no_dims = g[0]

    lam_trunc = lam[:no_dims]
    v_trunc = v[:,:no_dims]

    mappedX = X @ v_trunc
    mapping = {'map': v_trunc, 'lambdas': lam_trunc, 'fullmap': v, 'full_lambdas': lam}

    return mappedX, mapping


def qmaniGetU_nnGL( k, dt, epsilon, verbose=0, trunc=0 ):
    """
    Get unitary propagator from data

    Inputs:
        k: Euclidean distance matrix for dataset
        dt: time step
        epsilon: epsilon parameter
        verbose: verbosity flag
        trunc: how many eigenvalues of graph Laplacian to truncate at (0=no truncation)
    Outputs:
        Udt: unitary propagator (symmetrized)
        D_normalizer: normalization matrix (to recover non-unitary propagator)
    """
    if verbose>0:
        print("Construct graph Laplacian")

    epsForDisconnect = 0.0000000000001
    L = np.exp( np.divide(k, -epsilon) ) + epsForDisconnect
    # normalization
    D = np.matrix(L).sum(1)
    one_over_D = sp.sparse.diags(1/np.squeeze(np.asarray(D)), format="csc")
    La = one_over_D @ L @ one_over_D

    # second normalization to recover Markov operator
    Da = np.matrix(La).sum(1)
    D_normalizer = sp.sparse.diags(np.asarray(np.transpose(np.sqrt(1/Da)))[0], format="csc")
    M = D_normalizer @ La @ D_normalizer

    if verbose>0:
        print("Eigendecomposition")

    w, v = sp.linalg.eig(M)
    idx = np.argsort(w) # sorted in ascending order
    idx = idx[::-1] # reverse order to get descending eigenvalues
    w = np.real(w[idx])
    v = v[:,idx]
    v_inv = v.conj().T

    if trunc>0:
        print("Doing spectral truncation to", trunc )
        wt = w[:trunc]
        vt = v[:,:trunc]
        v_invt = v_inv[:trunc,:]

        M_new = sp.sparse.diags( np.exp( 1j*dt*np.real(np.sqrt((4*(1-wt))/epsilon)) ) )

        Udt = vt @ M_new @ v_invt
    else:
        M_new = sp.sparse.diags( np.exp( 1j*dt*np.real(np.sqrt((4*np.abs(1-w))/epsilon)) ), format="csc" )

        Udt = v @ M_new @ v_inv

    return Udt, D_normalizer


def pick_closest_to_mean(x, prob, thresh):
    """
    Return the index in x that is the point that is closest to the mean determined by prob

    Inputs:
        x: dataset (NxM matrix)
        prob: probability distribution(s) over dataset (Nx1 vector or NxK vector if there are K distributions to compute means with respect to)
        thresh: probability threshold. If >0, all values below thresh*max(prob) are ignored and prob is renormalized
    Outputs:
        ind: index(indices) for the data point closest to mean(s)
        dist: distance(s) (Euclidean) between mean(s) and closest data point(s)
    """
    nc = np.shape(prob)[1]

    # renormalize probability distribution if thresh>0
    if thresh>0:
        min_prob = thresh*np.max(prob,0)
        prob = np.multiply(prob, prob>min_prob)
        for jj in range(np.shape(prob)[1]):
            n = np.sum(prob[:,jj])
            prob[:,jj] = prob[:,jj]/n

    ind = np.zeros(nc, dtype=np.uint)
    dist = np.zeros(nc)
    indSec = np.arange(nc, dtype=np.uint)

    # find closest point for each ncol
    temp = np.transpose(x) @ prob
    dists = spatial.distance.cdist(x, np.transpose(temp))
    ind = np.argmin(dists, axis=0)
    dist = dists[ind,indSec]

    return ind, dist

def pick_closest_to_mean_pca(pt, k, x, delta_PCA, PCA_map, prob, thresh):
    """
    Return the index in x that is the point that is closest to the mean determined by prob,
    but data is given in PCA coords

    Inputs:
        pt: the point around which to do local PCA
        k: Euclidean distance matrix for dataset
        x: dataset (NxM matrix)
        delta_PCA: cutoff distance for determining points to include in local PCA
        PCA_map: the PCA projection map
        prob: probability distribution(s) over dataset (Nx1 vector)
        thresh: probability threshold. If >0, all values below thresh*max(prob) are ignored and prob is renormalized
    Outputs:
        ind: index for the data point closest to mean
        dist: distance (Euclidean) between mean and closest data point
    """

    # renormalize probability distribution if thresh>0
    if thresh>0:
        min_prob = thresh*np.max(prob,0)
        prob = np.multiply(prob, prob>min_prob)

    # get PCA around pt
    neighbors_idx = np.nonzero( k[pt,] < delta_PCA )[0]
    orig_pt_idx = np.nonzero(neighbors_idx == pt)[0][0]
    neighbors_idx = neighbors_idx.astype(int)
    orig_pt_idx = orig_pt_idx.astype(int)

    mappedX = x[neighbors_idx,] @ PCA_map
    coords = mappedX - mappedX[orig_pt_idx,:]

    # compute how much probability mass is outside the PCA neighborhood, and warn if it's more than 0.1 of total mass
    tot_prob = np.sum(prob)
    nonPCA = np.setdiff1d(range(len(prob)), neighbors_idx)
    frac_prob_outisde_PCA = np.sum(prob[nonPCA.astype(int)]) / tot_prob

    if frac_prob_outisde_PCA > 0.1:
        print("WARNING: appreciable probability mass outside PCA space: " + str(frac_prob_outisde_PCA))

    # renormalize probability mass in PCA neighborhood
    renorm_prob = prob[neighbors_idx]
    renorm_prob = renorm_prob / np.sum(renorm_prob)

    # compute mean in PCA coords
    mean_pos = np.transpose(coords) @ renorm_prob

    dists = spatial.distance.cdist(coords, np.array([mean_pos]))
    mindist_at = np.argmin(dists)

    ind = neighbors_idx[mindist_at]
    dist = dists[mindist_at]

    return ind, dist

def propagate(pt, qml_params, h, Npts, Us, x, k):
    """
    Propagate coherent state from a point and return destination point(s) where it propagates to (no PCA)

    Inputs:
        pt: the starting point for propagation
        qml_params: QML parameters
        h: h parameter
        Npts: number of data points in data set
        Us: quantum propagator
        x: dataset (NxM matrix)
        k: Euclidean distance matrix for dataset

    Outputs:
        idx_store: (nProp x nColl) matrix that contains destination points for each propagation step (nProp) and each propagation direction (nColl)
    """


    # extract parameters
    verbose = qml_params['verbose']
    nColl = qml_params['nColl']
    nProp = qml_params['nProp']
    prob_thresh = qml_params['prob_thresh']
    USE_MAX = qml_params['USE_MAX']

    # container to store the destination points after propagation
    # we do nColl propagations (each with a different momentum vector), for nProp time steps
    idx_store = np.zeros([nProp, nColl], dtype=int)
    Idx_store = np.zeros([nProp, nColl], dtype=int)

    # if verbose, output progress
    if verbose:
        if np.mod(pt, 100)==0:
            print("Propagating " + str(pt) + "/" + str(Npts))

    # container for initial states (each initial state is a column in this matrix)
    psi0_coll = np.zeros([Npts,nColl],dtype=complex)

    # sort points according to Euclidean distance from starting point (pt)
    sorted_idx = np.squeeze(np.argsort(k[pt,]))
    # print(pt, sorted_idx)

    # take the nColl closest points
    closest_pts = sorted_idx[1:nColl+1]

    # for each of the nColl initial states, set the momentum to be a (normalized) vector from starting point (pt) to
    # one of the closest points to it
    p0 = x[closest_pts,:] - x[pt,:]
    p0 = np.transpose(p0)/np.linalg.norm(p0, axis=1)

    # coherent state elements
    psi0_coll = np.transpose(np.multiply(np.exp( -k[:,pt]/(2*h) ), np.transpose(np.exp((-1j/h) * ((x - x[pt,:]) @ p0)))))
    
    # normalize coherent state
    psi0_coll = psi0_coll / np.linalg.norm(psi0_coll,axis=0)

    # propagate each of the initial states
    psi_coll = psi0_coll
    for pn in range(nProp):

        # propagate by one timestep (dt)
        psi_coll = Us @ psi_coll

        # normalize each state after propagation
        psi_coll = psi_coll / np.linalg.norm(psi_coll, axis=0)

        # extract probabilites from propagated states
        values = np.abs(psi_coll)**2

        # for each of the nColl propagations, extract max (if USE_MAX is set) or mean position
        if USE_MAX:
            idx_store[pn,:] = np.argmax(values,axis=0)
        else:
            ind, dist = pick_closest_to_mean(x, values, prob_thresh)
            idx_store[pn,:] = ind

    return idx_store

def propagate_PCA(pt, qml_params, h, Npts, Us, PCA_map, x, k):
    """
    Propagate coherent state from a point and return destination point(s) where it propagates to (with PCA)

    Inputs:
        pt: the starting point for propagation
        qml_params: QML parameters
        h: h parameter
        Npts: number of data points in data set
        Us: quantum propagator
        PCA_map: precomputed local PCA projection matrices
        x: dataset (NxM matrix)
        k: Euclidean distance matrix for dataset

    Outputs:
        idx_store: (nProp x nColl) matrix that contains destination points for each propagation step (nProp) and each propagation direction (nColl)
    """

    # extract parameters
    verbose = qml_params['verbose']
    nColl = qml_params['nColl']
    nProp = qml_params['nProp']
    prob_thresh = qml_params['prob_thresh']
    PCA_PREP = qml_params['PCA_PREP']
    PCA_MEAS = qml_params['PCA_MEAS']
    delta_PCA = qml_params['delta_PCA']
    USE_MAX = qml_params['USE_MAX']

    # container to store the destination points after propagation
    # we do nColl propagations (each with a different momentum vector), for nProp time steps
    idx_store = np.zeros([nProp, nColl], dtype=int)

    # if verbose, output progress
    if verbose:
        if np.mod(pt, 100)==0:
            print("Propagating " + str(pt) + "/" + str(Npts))

    # container for initial states (each initial state is a column in this matrix)
    psi0_coll = np.zeros([Npts,nColl],dtype=complex)

    if PCA_PREP:
        # get PCA around pt (for a neighborhood of point that are delta_PCA Euclidean distance from pt)
        neighbors_idx = np.nonzero( k[pt,] < delta_PCA )[0]
        orig_pt_idx = np.nonzero(neighbors_idx == pt)[0][0]
        neighbors_idx = neighbors_idx.astype(int)
        orig_pt_idx = orig_pt_idx.astype(int)
        mappedX = x[neighbors_idx,] @ PCA_map[pt]

        # compute distance matrix in PCA space
        kpca = spatial.distance.squareform(spatial.distance.pdist(mappedX, 'sqeuclidean'))

        # center PCA coordinates to pt
        coords = mappedX - mappedX[orig_pt_idx,:]

        # get closest points in PCA space
        sorted_PCA_idx = np.squeeze(np.argsort(kpca[orig_pt_idx,:]))

        # take the nColl closest points
        closest_pts = sorted_PCA_idx[1:(nColl+1)]

        # formulate nColl initial states, each with a momentum vector towards the closest points
        for ki in range(nColl):
            p0 = coords[closest_pts[ki],:] - coords[orig_pt_idx,:]
            p0 = p0/np.linalg.norm(p0)

            # coherent state formulated in PCA coordinates
            psi0_coll[neighbors_idx, ki] = np.multiply(
                                np.exp(-kpca[:,orig_pt_idx]/(2*h)),
                                np.exp((-1j/h) * (coords @ np.transpose(p0))) )

            # normalize coherent state
            psi0_coll[:, ki] = psi0_coll[:, ki] / np.linalg.norm(psi0_coll[:, ki])

    else:
        # if PCA is not to be used for initial state, formulate initial state in extrinsic coordinates

        # sort points according to Euclidean distance (in extrinsic coordinates) from starting point (pt)
        sorted_idx = np.squeeze(np.argsort(k[pt,]))
        closest_pts = sorted_idx[1:nColl+1]

        for ki in range(nColl):
            # for each of the nColl initial states, set the momentum to be a (normalized) vector from starting point (pt) to
            # one of the closest points to it
            p0 = x[closest_pts[ki],:] - x[pt,:]
            p0 = p0/np.linalg.norm(p0)

            # coherent state
            psi0_coll[:,ki] = np.multiply( np.exp( -k[:,pt]/(2*h) ), np.exp((-1j/h) * ((x - x[pt,:]) @ np.transpose(p0))) )

            # normalize
            psi0_coll[:,ki] = psi0_coll[:,ki] / np.linalg.norm(psi0_coll[:,ki])

    # propagate each of the initial states
    psi_coll = psi0_coll
    for pn in range(nProp):

        # propagate by one timestep (dt)
        psi_coll = Us @ psi_coll

        # normalize each state after propagation
        for ki in range(nColl):
            psi_coll[:,ki] = psi_coll[:,ki] / np.linalg.norm(psi_coll[:,ki])

        # extract probabilites from propagated states
        values = np.abs(psi_coll)**2

        # for each of the nColl propagations, extract max (if USE_MAX is set) or mean position
        if USE_MAX:
            for ki in range(nColl):
                idx_store[pn,ki] = np.argmax(values[:,ki])
        else:
            if PCA_MEAS:
                # use PCA coordinates to calculate mean
                for ki in range(nColl):
                    # do local PCA around the max of the distribution
                    ptl = np.argmax(values[:,ki])
                    ind, dist = pick_closest_to_mean_pca(ptl, k, x, delta_PCA, PCA_map[ptl], values[:,ki], prob_thresh)

                    idx_store[pn,ki] = ind
            else:
                # if no PCA for measurements, calculate mean in extrinsic coordinates
                ind, dist = pick_closest_to_mean(x, values, prob_thresh)
                idx_store[pn,:] = ind

    return idx_store


# ------------------------------------
# Main function handle
# ------------------------------------
def run(qml_params):
    """
    Compute quantum propagator from data, and execute QML propagations to determine geodesic distance matrix

    Inputs:
        qml_params: QML parameters
    Outputs:
        D: the geodesic distance matrix
    """

    # current time
    s_time = time.time()

    # extract parameters
    logepsilon = qml_params['logepsilon']
    alpha = qml_params['alpha']
    dt = qml_params['dt']
    nProp = qml_params['nProp']
    nColl = qml_params['nColl']
    PCA_PREP = qml_params['PCA_PREP']
    PCA_MEAS = qml_params['PCA_MEAS']
    PCA_dims = qml_params['PCA_dims']
    delta_PCA = qml_params['delta_PCA']
    gamma = qml_params['gamma']
    USE_MAX = qml_params['USE_MAX']
    prob_thresh = qml_params['prob_thresh']
    verbose = qml_params['verbose']
    SHOW_EMBEDDING = qml_params['SHOW_EMBEDDING']

    # form epsilon and h
    epsilon = np.exp(logepsilon)
    h = epsilon**(1/(2+alpha))

    # load data
    try:
        # x = np.genfromtxt(qml_params['datafile'], delimiter=',')
        x = read_in_matrix(qml_params['datafile'], verbose)
    except:
        print("Cannot open data file: " + qml_params['datafile'] + "... Exiting.")
        raise Exception("Cannot open data file")
    else:
        # Npts is the number of data points
        Npts = np.shape(x)[0]

        for i in range(Npts):
            for j in range(Npts):
                if (np.isfinite(x[i,j])):
                    x[i,j] = 1 - x[i,j]
                    print(x[i,j])
        # compute Euclidean squared distance matrix
        # k = spatial.distance.squareform(spatial.distance.pdist(x, 'sqeuclidean'))
        k = x

#PCA
        PCA = PCA_PREP | PCA_MEAS
        if PCA:
            # if PCA is required for state preparation or measurement, prepare local PCA maps for all points ahead of time
            if delta_PCA == 0:
                # if delta_PCA is not specified, set it to 2*h
                delta_PCA = 2*h
            PCA_map = dict()

            # loop over all data points
            for pt in range(Npts):
                # get local PCA mapping from smaller neighborhood (see discussion in Sec. III.B of the Appendix of arXiv:2112.11161)
                neighbors_idx = np.nonzero( k[pt,] < (delta_PCA * gamma) )[0]
                orig_pt_idx = np.nonzero(neighbors_idx == pt)[0][0]

                neigh_sz = len(neighbors_idx)
                scale = 2
                # if neighborhood size is too small to get an accurate PCA mapping, expand it
                while neigh_sz < 50:
                    if verbose:
                        print("pt " + str(pt) + ": Not enough points in PCA neighborhood, expanding...")
                    neighbors_idx = np.nonzero( k[pt,] < scale*(delta_PCA * gamma) )[0]
                    orig_pt_idx = np.nonzero(neighbors_idx == pt)[0][0]

                    neigh_sz = len(neighbors_idx)
                    scale = scale+1

                neighbors_idx = neighbors_idx.astype(int)
                orig_pt_idx = orig_pt_idx.astype(int)

                # once the neighborhood is obtained, compute local PCA map and mapping of points in neighborhood
                mappedX, mapping = PCA_for_ts(x[neighbors_idx,], orig_pt_idx, PCA_dims)

                # if some PCA dims have very small eigenvalues (due to a fixed PCA_dims), truncate these
                mapping['map'] = mapping['map'][:, mapping['lambdas']>1e-4]
                deficit = PCA_dims - np.shape(mapping['map'])[1]
                if deficit>0:
                    if verbose:
                        print("pt " + str(pt) + ": Deficit in PCA by " + str(deficit))
                    mapping['map'] = np.append(mapping['map'], np.zeros([np.shape(mapping['map'])[0], deficit]))

                # store local PCA projection matrix for this point
                PCA_map[pt] = mapping['map']

                # if verbose, output progress
                if verbose:
                    if pt % 50==0:
                        print("PCA done for " + str(pt) + "/" + str(Npts))

# QPROP
        # compute quantum propagator
        Udt, D_normalizer = qmaniGetU_nnGL( x, dt, epsilon, verbose, trunc=0 )
        D_normalizer_inv = spinv(D_normalizer)
        Us = D_normalizer @ Udt @ (D_normalizer_inv)

# approx spatial representation of graph
        # compute Euclidean squared distance matrix
        g = ig.Graph.Weighted_Adjacency(np.where(np.isfinite(x), x, 0))
        fig = plt.figure(figsize=(6,6))
        lyout3d = g.layout_fruchterman_reingold_3d()
        ax = fig.add_subplot(111, projection='3d')
        ed = np.array(lyout3d.coords)
        # load color map
        if qml_params['colorfile']!=False:
            try:
                # colors = np.genfromtxt(qml_params['colorfile'], delimiter=',')
                print(qml_params['colorfile'])
                colors = read_in_matrix(qml_params['colorfile'], qml_params['verbose'])
            except:
                print("Cannot open data file: " + qml_params['colorfile'] + "... Exiting.")
                raise Exception("Cannot open color file")
            else:
                ax.scatter(ed[:,0], ed[:,1], ed[:,2], c=colors, cmap=plt.cm.Spectral)
        else:
            ax.scatter(ed[:,0], ed[:,1], ed[:,2])
        ax.axis('off')
        ax.set_title('3D embedding')
        print("edge count:", np.count_nonzero(np.isfinite(x)))
        plt.show()
        print("3d layout: ", ed)
        k = spatial.distance.squareform(spatial.distance.pdist(ed, 'sqeuclidean'))
        x = ed

# Propagate
        # container to store destination points after propagation
        peak_idxs = dict()

        # propagate from each point in dataset, and store destination points
        if PCA:
            for pt in range(Npts):
                peak_idxs[pt] = propagate_PCA(pt, qml_params, h, Npts, Us, PCA_map, x, k)
        else:
            for pt in range(Npts):
                peak_idxs[pt] = propagate(pt, qml_params, h, Npts, Us, x, k)


# Fill in geodesic distance matrix
        # container for geodesic distances
        D = np.zeros([Npts, Npts])

        # for each of the Npts points, and for each of the nProp propagation times, and for each of the nColl propagations,
        # store the distance to the destination as the propagated time (and symmetrize D)
        for pt in range(Npts):
            for pn in range(nProp):
                for ki in range(nColl):
                    if (D[pt,peak_idxs[pt][pn,ki]]==0) | (D[pt,peak_idxs[pt][pn,ki]]>(pn+1)*dt):
                        D[pt,peak_idxs[pt][pn, ki]] = (pn+1)*dt
                        D[peak_idxs[pt][pn,ki], pt] = (pn+1)*dt

            # set the diagonal elements to zero by force
            D[pt,pt]=0


        # output time taken
        e_time = time.time()
        print(f"QML Done. Time taken = {e_time-s_time}")

        return D

def get_hamiltonian(k, epsilon):
    """
    Compute data-driven Hamiltonian

    Inputs:
        k: Euclidean distance matrix for dataset
        epsilon: epsilon parameter

    Outputs:
        H: data-driven Hamiltonian
    """
    epsForDisconnect = 0.0000000000001
    L = np.exp( np.divide(k, -epsilon) ) + epsForDisconnect

    # normalization
    D = np.matrix(L).sum(1)
    one_over_D = sp.sparse.diags(1/np.squeeze(np.asarray(D)), format="csc")
    La = one_over_D @ L @ one_over_D

    # second normalization to recover Markov operator
    Da = np.matrix(La).sum(1)
    D_normalizer = sp.sparse.diags(np.asarray(np.transpose(1/Da))[0], format="csc")
    M = D_normalizer @ La

    H = (4/epsilon) * (np.identity(np.shape(M)[0]) - M)

    return H

def perform_hamiltonian_test(qml_params):
    """
    Test data-driven Hamiltonian with various values of epsilon and h.
    Funciton plots error and asks user to choose log(epsilon) and log(h) to proceed with.

    Inputs:
        qml_params: QML parameters
    Outputs:
        retval: a dictionary containing the user inputted log(epsilon) and log(h) values
    """

    # range of parameters to test over
    logeps_v = np.arange(-30,-20,0.5)
    logh_v =  np.arange(-20,-10,0.5)
    Neps = np.shape(logeps_v)[0]

    # number of states to evaluate expectation over
    avg = qml_params['H_test_avg']

    # load data
    try:
        # x = np.genfromtxt(qml_params['datafile'], delimiter=',')
        x = read_in_matrix(qml_params['datafile'], qml_params['verbose'])
    except:
        print("Cannot open data file: " + qml_params['datafile'] + "... Exiting.")
        raise Exception("Cannot open data file")
    else:
        # Npts is the number of data points
        Npts = np.shape(x)[0]

        # compute Euclidean squared distance matrix
        g = ig.Graph.Weighted_Adjacency(np.where(np.isfinite(x), x, 0))
        fig = plt.figure(figsize=(6,6))
        lyout3d = g.layout_fruchterman_reingold_3d()
        ax = fig.add_subplot(111, projection='3d')
        ed = np.array(lyout3d.coords)
        # load color map
        if qml_params['colorfile']!=False:
            try:
                # colors = np.genfromtxt(qml_params['colorfile'], delimiter=',')
                print(qml_params['colorfile'])
                colors = read_in_matrix(qml_params['colorfile'], qml_params['verbose'])
            except:
                print("Cannot open data file: " + qml_params['colorfile'] + "... Exiting.")
                raise Exception("Cannot open color file")
            else:
                ax.scatter(ed[:,0], ed[:,1], ed[:,2], c=colors, cmap=plt.cm.Spectral)
        else:
            ax.scatter(ed[:,0], ed[:,1], ed[:,2])
        ax.axis('off')
        ax.set_title('3D embedding')
        print("edge count:", np.count_nonzero(np.isfinite(x)))
        plt.show()
        print("3d layout: ", ed)
        k = spatial.distance.squareform(spatial.distance.pdist(ed, 'sqeuclidean'))

        # container for storing devitations/errors
        devs = np.zeros([len(logeps_v), len(logh_v)])

        # loop over parameters and evaluate error in expectations value of data-driven Hamiltonian under coherent state

        # loop over epsilon
        for le_i, le in enumerate(logeps_v):

            if qml_params['verbose']:
                if np.mod(le_i, 10)==0:
                    print("log epsilon = {} ({}/{})".format(le, le_i, Neps))

            epsilon = np.exp(le)

            # calculate Hamiltonian
            H = get_hamiltonian(x, epsilon) # graph x

            # loop over h
            for lh_i, lh in enumerate(logh_v):
                h = np.exp(lh)
                temp = np.zeros([avg,])

                # calculate deviations for coherent states centered at avg initial points
                for ii in range(avg):
                    # choose random initial point
                    pt = np.random.randint(0,Npts)

                    # sort other points according to their distance from pt
                    sorted_idx = np.squeeze(np.argsort(k[pt,]))

                    # pick momentum as (normalized) vector to closest point
                    p0 = ed[sorted_idx[1],:] - ed[pt,:]
                    p0 = p0/np.linalg.norm(p0)

                    # formulate coherent state (in extrinsic coordinates)
                    psi0 = np.multiply( np.exp( -k[:,pt]/(2*h) ), np.exp((-1j/h) * ((ed - ed[pt,:]) @ np.transpose(p0))) )
                    psi0 = psi0 / np.linalg.norm(psi0)

                    # calculate error in expectation value (should be 1 since Hamiltonian approximates p^2 and |p|=1)
                    temp[ii] = np.abs((h**2) * np.inner(np.conj(psi0).T, np.matmul(H,psi0)) - 1)

                # store deviation
                devs[le_i, lh_i] = np.average(temp)

        for i in range(devs.shape[0]):
            for j in range(devs.shape[1]):
                if (devs[i,j] > 1):
                    devs[i,j] = 1

        # plot
        loge, logh = np.meshgrid(logeps_v, logh_v, indexing='ij')
        fig, ax = plt.subplots()
        print("loge", loge)
        print("logh", logh)
        print("devs", devs)
        im = ax.pcolormesh(loge, logh, devs)
        # im = ax.pcolormesh(np.transpose(logh), np.transpose(loge), devs)
        fig.colorbar(im)

        ax.set_xlabel('log(eps)')
        ax.set_ylabel('log(h)')
        ax.set_title('Deviation -- choose log(epsilon) and log(h) values')

        plt.savefig('h_test.png')
        plt.show()

        retval = {}
        entry = input('Enter log(epsilon) value: ')
        retval['logepsilon'] = float(entry)

        entry = input('Enter log(h) value: ')
        retval['logh'] = float(entry)

        return retval

# ------------------------------------
# main
# ------------------------------------
if __name__ == '__main__':
    """
    When called from command line, the parameter is the name of text file that contains input parameters

    Output:
        - Saves geodesic distance matrix to file "f.out", where "f" is the input filename
        - Optionally, also plots an embedding of the graph if SHOW_EMBEDDING = 2 or 3 (this number sets the embedding dimension) in the input file
    """

    assert (len(sys.argv)==2), "QML takes one argument, an input filename."
    print('------------------------------------------------------')
    print('QML Loading parameters from file ' + sys.argv[1] + '...')
    print('------------------------------------------------------' + '\n')

    # load parameters and datafile name
    try:
        f = open(sys.argv[1])
    except:
        print("Cannot open input file: " + sys.argv[1] + "... Exiting.")
    else:
        data = f.read()
        inp = json.loads(data)

        # initialize qml_params
        qml_params = initialize(inp)

        print(qml_params)
        print('\n')

        # if H_test is set, perform it
        if qml_params['H_test']:
            print("Performing Hamiltonian test ...")
            vals = perform_hamiltonian_test(qml_params)

            # set epsilon and alpha according to values selected from H_test
            qml_params['logepsilon'] = vals['logepsilon']
            qml_params['alpha'] = vals['logepsilon']/vals['logh'] - 2
            print( 'Using log(eps)={}, alpha={}'.format(qml_params['logepsilon'], qml_params['alpha']))

        # run QML
        D = run(qml_params)
        print(np.count_nonzero(D))

        # save geodesic distance matrix to file
        fname = "{}.out".format(sys.argv[1])
        np.savetxt(fname, D, fmt='%.10f', delimiter=',')

        if qml_params['SHOW_EMBEDDING']==2:
            print("Computing 2D embedding using geodesic distance matrix ...")
            g = ig.Graph.Weighted_Adjacency(D)
            fig = plt.figure(figsize=(6,6))

            lyout2d = g.layout_fruchterman_reingold()
            ax = fig.add_subplot(111)
            ed = np.array(lyout2d.coords)
            ig.plot(g, layout=lyout2d, target=ax, edge_width=0)
            ax.axis('off')
            ax.set_title('2D embedding')

            plt.show()

        if qml_params['SHOW_EMBEDDING']==3:
            print("Computing 3D embedding using geodesic distance matrix ...")
            g = ig.Graph.Weighted_Adjacency(D)
            fig = plt.figure(figsize=(6,6))

            lyout3d = g.layout_fruchterman_reingold_3d()
            ax = fig.add_subplot(111, projection='3d')
            ed = np.array(lyout3d.coords)
            # load color map
            if qml_params['colorfile']!=False:
                try:
                    # colors = np.genfromtxt(qml_params['colorfile'], delimiter=',')
                    colors = read_in_matrix(qml_params['colorfile'], qml_params['verbose'])
                except:
                    print("Cannot open data file: " + qml_params['colorfile'] + "... Exiting.")
                    raise Exception("Cannot open color file")
                else:
                    ax.scatter(ed[:,0], ed[:,1], ed[:,2], c=colors, cmap=plt.cm.Spectral)
            else:
                ax.scatter(ed[:,0], ed[:,1], ed[:,2])
            ax.axis('off')
            ax.set_title('3D embedding')

            plt.show()

        np.savetxt("{}.csv".format(sys.argv[1]), ed, delimiter=",")

        # components = g.connected_components(mode='weak')
        # fig, ax = plt.subplots()
        # ig.plot(
        #     components,
        #     target=ax,
        #     palette=ig.RainbowPalette(),
        #     vertex_size=0.07,
        #     vertex_color=list(map(int, ig.rescale(components.membership, (0, 200), clamp=True))),
        #     edge_width=0.7
        # )
        # plt.show()

        # components = g.connected_components(mode='weak')
        # fig, ax = plt.subplots()
        # ig.plot(
        #     components,
        #     target=ax,
        #     palette=ig.RainbowPalette(),
        #     vertex_size=0.07,
        #     vertex_color=list(map(int, ig.rescale(components.membership, (0, 200), clamp=True))),
        #     edge_width=0.7
        # )
        # plt.show()