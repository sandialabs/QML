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
import datetime
import shutil
import pickle
import os
import sys
import json
import numpy as np
import scipy.spatial as spatial
from scipy.sparse.linalg import inv as spinv
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib import cm
import matplotlib.cm as cm
import igraph as ig
import scipy as sp
from sklearn.metrics import silhouette_score
from math import sqrt

from qml_serial_analyze import visualize_propagations
from qml_serial_preprocess import initialize, read_in_matrix
from hover import enable_hover

# np.set_printoptions(threshold=sys.maxsize)

# ------------------------------------
# Functions
# ------------------------------------
   
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

    L = np.exp( np.divide(k, -epsilon) )
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


def pick_closest_to_mean(x, prob, thresh, verbose=False):
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
    nc = np.shape(prob)[1] # num columns

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
    temp = np.transpose(x) @ prob # Mxk matrix
    dists = spatial.distance.cdist(x, np.transpose(temp)) # Nxk matrix

    ind = np.argmin(dists, axis=0) # 1xk matrix
    dist = dists[ind,indSec] # 1xk matrix

    if verbose:
        print('picking closest to mean')
        print(temp.shape)
        print(np.sort(dists, axis=0)[:3])
        print(locals())

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
        verbose=False

        # for each of the nColl propagations, extract max (if USE_MAX is set) or mean position
        if USE_MAX:
            idx_store[pn,:] = np.argmax(values,axis=0)
        else:
            ind, dist = pick_closest_to_mean(x, values, prob_thresh, verbose)
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
        if np.mod(pt, 100)==0 or (np.mod(pt, 25) == 0 and pt > 300):
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
        x, norms = unit_sphere_normalize(x)

        # compute Euclidean squared distance matrix
        k = spatial.distance.squareform(spatial.distance.pdist(x, 'sqeuclidean'))

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
        Udt, D_normalizer = qmaniGetU_nnGL( k, dt, epsilon, verbose, trunc=0 )
        D_normalizer_inv = spinv(D_normalizer)
        Us = D_normalizer @ Udt @ (D_normalizer_inv)
        print(Us.shape)
        print(Us)

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

        return D, peak_idxs, x, k, norms

def get_hamiltonian(k, epsilon):
    """
    Compute data-driven Hamiltonian

    Inputs:
        k: Euclidean distance matrix for dataset
        epsilon: epsilon parameter

    Outputs:
        H: data-driven Hamiltonian
    """

    L = np.exp( np.divide(k, -epsilon) )

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



### NORMALIZATION FUNCTIONS
def box_unnormalize(x, least, greatest, upper_lim=1):
    x /= upper_lim
    x *= greatest
    x += least
    return x


# upper_lim = .1 makes the maximum .1
def box_normalize(x, upper_lim=1):
    # subtract min and div by max (box normalization)
    least = np.min(x)
    x -= least
    # min is now 0
    
    greatest = np.max(x)
    x /= greatest
    # max is now 1

    x *= upper_lim 
    return x, (least, greatest)

def unit_sphere_unnormalize(x, norms, max_norm=True):
    Npts = np.shape(x)[0]

    if max_norm:
        x *= max(norms)
        return x

    for i in range(Npts):
        x[i] *= norms[i]
    return x
    

def unit_sphere_normalize(x, max_norm=True):
    Npts = np.shape(x)[0]
    x = x.astype('float64')

    # div by norm (unit sphere normalization)
    norms = [np.linalg.norm(row) for row in x]
    if max_norm:
        x /= max(norms)
        return x, norms
    
    for i in range(Npts):
        x[i] /= norms[i]
    return x, norms

##### END NORMALIZATION

# matrix printing function
def row_to_str(row):
    s = ''
    for item in row:
        s += f'{item},'
    return s
    
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
    END = -6 # used to be -10
    START = 0  # used to be 6
    logeps_v = np.arange(END,START,0.5)
    logh_v =  np.arange(END,START,0.5)
    Neps = np.shape(logeps_v)[0]

    # number of states to evaluate expectation over
    avg = qml_params['H_test_avg']

    # load data
    try:
        # x = np.genfromtxt(qml_params['datafile'], delimiter=',')
        x = read_in_matrix(qml_params['datafile'], qml_params['verbose']).astype(np.float64)
    except:
        print("Cannot open data file: " + qml_params['datafile'] + "... Exiting.")
        raise Exception("Cannot open data file")
    else:
        # Npts is the number of data points
        Npts = np.shape(x)[0]

        ## NORMALIZATION
        x, _ = unit_sphere_normalize(x)

        for i in range(5):
            print(x[i])

        ### DEBUG PRINTS to check dist between first two images
        # first = x[0]
        # second = x[1]
        # dist = 0
        # with open('first_two_pts.txt', 'w') as f:
        #     for i in range(len(first)):
        #         f.write(f'{first[i]},,,{second[i]}\n')
        #         dist += pow(first[i] - second[i], 2)
        # print(dist)
        # #### 


        # compute Euclidean squared distance matrix
        k = spatial.distance.squareform(spatial.distance.pdist(x, 'sqeuclidean'))

        # container for storing devitations/errors
        devs = np.zeros([len(logeps_v), len(logh_v)])

        # loop over parameters and evaluate error in expectations value of data-driven Hamiltonian under coherent state

        # loop over epsilon
        for le_i, le in enumerate(logeps_v):
            print('out: ', le_i)

            if qml_params['verbose']:
                if np.mod(le_i, 10)==0:
                    print("log epsilon = {} ({}/{})".format(le, le_i, Neps))

            epsilon = np.exp(le)

            # calculate Hamiltonian
            H = get_hamiltonian(k, epsilon)
            #print(H)

            # loop over h
            for lh_i, lh in enumerate(logh_v):
                print('in: ', lh_i)
                h = np.exp(lh)
                temp = np.zeros([avg,])

                # calculate deviations for coherent states centered at avg initial points
                for ii in range(avg):
                    # choose random initial point
                    pt = np.random.randint(0,Npts)

                    # sort other points according to their distance from pt
                    sorted_idx = np.squeeze(np.argsort(k[pt,]))
                    assert sorted_idx[0] == pt, f'{pt} != {sorted_idx[0]}'
                    # print(f'first closest index to {pt}: {sorted_idx[0]}')
                    # print(f'second closest index to {pt}: {sorted_idx[1]}')

                    # pick momentum as (normalized) vector to closest point
                    p0 = x[sorted_idx[1],:] - x[pt,:]

                    ### DEBUG PRINTS to check nearby points
                    # _pts = [x[pt,:], x[sorted_idx[1]], x[sorted_idx[2]], x[sorted_idx[3]]]
                    # print(_pts[0], np.sum(_pts[0] - _pts[0]))
                    # print(_pts[1], np.sum(_pts[1] - _pts[0]))
                    # print(_pts[2], np.sum(_pts[2] - _pts[1]))
                    # print(_pts[3], np.sum(_pts[3] - _pts[2]))
                    # print('-------------')

                    p0 = p0/np.linalg.norm(p0)

                    # formulate coherent state (in extrinsic coordinates)
                    psi0 = np.multiply( np.exp( -k[:,pt]/(2*h) ), np.exp((-1j/h) * ((x - x[pt,:]) @ np.transpose(p0))) )
                    psi0 = psi0 / np.linalg.norm(psi0)

                    # calculate error in expectation value (should be 1 since Hamiltonian approximates p^2 and |p|=1)
                    temp[ii] = np.abs((h**2) * np.inner(np.conj(psi0).T, np.matmul(H,psi0)) - 1)

                # store deviation
                devs[le_i, lh_i] = np.average(temp)

        # for i in range(devs.shape[0]):
        #     for j in range(devs.shape[1]):
        #         if (devs[i,j] > 1):
        #             devs[i,j] = 1

        # plot
        loge, logh = np.meshgrid(logeps_v, logh_v, indexing='ij')
        fig, ax = plt.subplots()
        print("loge", loge)
        print("logh", logh)
        print("devs", devs)
        im = ax.pcolormesh(loge, logh, np.log(devs))
        # im = ax.pcolormesh(np.transpose(logh), np.transpose(loge), devs)
        fig.colorbar(im)

        ax.set_xlabel('log(eps)')
        ax.set_ylabel('log(h)')
        ax.set_title('Deviation -- choose log(epsilon) and log(h) values')

        plt.savefig('h_test.png')

        global LOG_PATH
        plt.savefig(os.path.join(LOG_PATH,'h_test.png'))
        plt.show()

        retval = {}
        entry = input('Enter log(epsilon) value: ')
        retval['logepsilon'] = float(entry)

        entry = input('Enter log(h) value: ')
        retval['logh'] = float(entry)

        return retval
    

# Saves embedding graph and geodesic distance matrix
def results_saver(D):
    global LOG_PATH
    global DIM

    embedding_filename = os.path.join(LOG_PATH, f'embedding_{DIM}d.out')
    np.savetxt(embedding_filename, D, fmt='%.10f', delimiter=',')

    graph_filename = os.path.join(LOG_PATH, f'embedding_{DIM}d_graph.png')
    plt.savefig(graph_filename)


def fill_graph(ax, ed, do_score=False):
    num_dim = len(ed[0])
    
    if qml_params['colorfile']!=False:
        try:
            # colors = np.genfromtxt(qml_params['colorfile'], delimiter=',')
            colors = read_in_matrix(qml_params['colorfile'], qml_params['verbose'])
            # colors = np.loadtxt(open(qml_params['colorfile'], "rb"), delimiter=",", dtype=str)
            colors = np.array(colors)
        except:
            print("Cannot open color file: " + qml_params['colorfile'] + "... Exiting.")
            raise Exception("Cannot open color file")
        else:
            if num_dim == 2:
                sc = ax.scatter(ed[:,0], ed[:,1], c=colors, cmap=plt.cm.Spectral)
            elif num_dim == 3:
                sc = ax.scatter(ed[:,0], ed[:,1], ed[:,2], c=colors, cmap=plt.cm.Spectral)
        
            if do_score:
                score = silhouette_score(ed, colors)
                print("silhouette score: ", score)
                scores = np.zeros(unique_classes.shape)
                i = 0
                # colors = colors.reshape(-1)
                # print("colors", colors, colors.shape)
                # print("unique classes", unique_classes)
                for i, clas in enumerate(unique_classes):
                    # print("clas", clas)
                    # tempColor = labelColorMapping[clas]
                    tempColors = np.ones(colors.shape) * 2
                    tempColors[colors == clas] = 1
                    # tempColors = tempColors.reshape((tempColors.size, 1))
                    # print("tempcolors", tempColors, tempColors.shape)
                    scores[i] = silhouette_score(ed, tempColors)
                # print("class sizes", classSizes.flatten())
                indices = np.argsort(classSizes.flatten())
                # print("indices", indices)
                print("class sizes", classSizes[indices].flatten())
                scores = scores[indices]
                np.set_printoptions(precision=4)
                print("single class silhouette scores", scores)


    else:
        if qml_params['labelfile']!=False:
            try:
                # colors = np.genfromtxt(qml_params['colorfile'], delimiter=',')
                labels = np.loadtxt(open(qml_params['labelfile'], "rb"), delimiter=",", dtype=str)
            except:
                print("Cannot open label file: " + qml_params['labelfile'] + "... Exiting.")
                raise Exception("Cannot open label file")
            else:
                # legend_handles = []
                # legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color))
                
                # print("label size ", labels.size)
                # print(labels)
                # print("color size ", colors.size)
                # print(colors)
                # print("embed size", ed.size)
                # print("n ", ed[:,0].size)

                
                unique_classes = np.unique(np.array(colors))
                # print("unique_classes", unique_classes)
                # Generate a colormap with a different color for each class
                num_classes = len(unique_classes)
                classSizes = np.zeros((num_classes,1))
                # base_cmap = plt.get_cmap('tab20')
                # # Create a new colormap with 99 colors by replicating the base_cmap
                # num_colors = num_classes
                # new_colors = np.concatenate([base_cmap(i * np.ones(5)) for i in range(5)])
                # # Trim the colormap to have the desired number of colors
                # cmap = ListedColormap(new_colors[:num_colors], name='custom_cmap', N=num_colors)
                # print("num_classes", num_classes)
                cmap = plt.get_cmap('hsv', num_classes) # viridis Spectral
                # print("cmap ", cmap)
                print("colors", colors)

                # Get multiple qualitative colormaps
                cmaps = ['tab20b', 'tab20c', 'Set1', 'Set3', 'Dark2', 'Accent']

                # Combine colormaps to create a new colormap
                num_colors = num_classes
                new_colors = []
                print("cmap size", len(cmaps))
                for cmap in cmaps:
                    base_cmap = plt.get_cmap(cmap)
                    new_colors.extend(base_cmap(np.arange(base_cmap.N)))
                    # new_colors.extend(base_cmap(np.linspace(0, 1, num_colors // len(cmaps))))

                # Trim the colormap to have the desired number of colors
                cmap = ListedColormap(new_colors[:num_colors], name='custom_cmap', N=num_colors)


                labelColorMapping = {}
                for i, label in enumerate(labels):
                    if label not in labelColorMapping:
                        # print("label color, ", label, colors[i], i)
                        labelColorMapping[label] = colors[i]
                i = 0
                for label, color in labelColorMapping.items():
                    # print("label color", label, color)
                    # print("colors==color", colors==color)
                    # print("colorsingle", np.where(unique_classes == color))
                    tempColors = colors[(colors==color).reshape(-1)]
                    tempX = ed[(colors==color).reshape(-1),:]
                    tempLabels = labels[(labels==label)]
                    # print("tempX size", tempX.size)
                    # print("tempColors size", tempColors.size)
                    # print("tempLabels size", tempLabels.size)
                    # print("color, ", color.size, color, color[0])
                    colorSingle = cmap(np.where(unique_classes == color))
                    # print("colorSingle ", colorSingle, color)
                    colorPrint = np.full((tempX[:,0].shape[0],4), colorSingle)
                    if (tempX.size > 0):
                        sc = ax.scatter(tempX[:,0], tempX[:,1], tempX[:,2], c=colorPrint, label=label)
                    # ax.scatter(tempX[:,0], tempX[:,1], tempX[:,2], label=label)
                    classSizes[i] = tempX.size
                    i += 1
                plt.legend(ncol=num_classes/25, fontsize="4")

                # legend_entries = {}
                # # Create the scatter plot with unique labels and their respective colors
                # for i, label in enumerate(labels):
                #     if label not in legend_entries:
                #         legend_entries[label] = ax.scatter(ed[i,0], ed[i,1], ed[i,2], c=colors[i], label=label, cmap=plt.cm.Spectral)
                #     else:
                #         ax.scatter(ed[i,0], ed[i,1], ed[i,2], c=colors[i], cmap=plt.cm.Spectral)
                # # Create a custom legend based on the unique labels and colors
                # handles = [legend_entries[label] for label in legend_entries]
                # plt.legend(handles=handles)
                
                # ax.scatter(ed[:,0], ed[:,1], ed[:,2], c=colors, cmap=plt.cm.Spectral, label=labels)
                # plt.legend(loc='upper left')
        else:
            if num_dim == 2:
                sc = ax.scatter(ed[:,0], ed[:,1])
            elif num_dim == 3:
                sc = ax.scatter(ed[:,0], ed[:,1], ed[:,2])
    return sc


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

        ### pre-processing for logging / saving results
        DIM = qml_params['SHOW_EMBEDDING']
        RESULTS_DIR = 'results'
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        log_filename = f'{sys.argv[1].split(".")[0]}-{timestamp}'
        LOG_PATH = os.path.join(RESULTS_DIR, log_filename)
        os.makedirs(LOG_PATH)
        shutil.copyfile(sys.argv[1], os.path.join(LOG_PATH, sys.argv[1])) # copies param file for logging
        ### 

        # if H_test is set, perform it
        if qml_params['H_test']:
            print("Performing Hamiltonian test ...")
            vals = perform_hamiltonian_test(qml_params)

            # set epsilon and alpha according to values selected from H_test
            qml_params['logepsilon'] = vals['logepsilon']
            qml_params['alpha'] = vals['logepsilon']/vals['logh'] - 2
            print( 'Using log(eps)={}, log(h)={}, alpha={}'.format(qml_params['logepsilon'], vals['logh'], qml_params['alpha']))

        # run QML
        D, peak_idxs, x, k, norms = run(qml_params)
        print(k.shape)


        ### SAVING QML RESULTS
        with open('euclidian_matrix.txt', 'w') as f:
            for i in k:
                for j in i:
                    f.write(f'{j}, ')
                f.write('\n')
        print(np.count_nonzero(D))

        with open(os.path.join(LOG_PATH, 'peak_idxs.pickle'), 'wb') as f: 
            pickle.dump(peak_idxs, f)

        original_x = unit_sphere_unnormalize(x, norms)
        with open(os.path.join(LOG_PATH, 'x.pickle'), 'wb') as f: 
            pickle.dump(original_x, f)

        # save geodesic distance matrix to file
        fname = "{}.out".format(sys.argv[1])
        np.savetxt(fname, D, fmt='%.10f', delimiter=',')
        ###### DONE SAVING: CAN BE ANALYZED SEPARATELY USING qml_serial_analyze.py


        if qml_params['SHOW_EMBEDDING']==2:
            print("Computing 2D embedding using geodesic distance matrix ...")
            g = ig.Graph.Weighted_Adjacency(D)
            fig = plt.figure(figsize=(6,6))

            lyout2d = g.layout_fruchterman_reingold()
            ax = fig.add_subplot(111)
            ed = np.array(lyout2d.coords)
            #ig.plot(g, layout=lyout2d, target=ax, edge_width=0)

            sc = fill_graph(ax, ed)
            ax.axis('off')
            ax.set_title('2D embedding')
            results_saver(D)

            with open('data/teapot/angle_orders.txt', 'r') as f:
                hover_data = f.readlines()
            enable_hover(hover_data, fig, ax, sc)

            plt.show()

        if qml_params['SHOW_EMBEDDING']==3:
            #print("Computing 3D embedding using geodesic distance matrix ...")
            #g = ig.Graph.Weighted_Adjacency(D)

            print("Computing 3D embedding using euclidian distance matrix ...")
            g = ig.Graph.Weighted_Adjacency(k)
            fig = plt.figure(figsize=(6,6))

            lyout3d = g.layout_fruchterman_reingold_3d()
            ax = fig.add_subplot(111, projection='3d')
            ed = np.array(lyout3d.coords)
            fill_graph(ax, ed)
            ax.axis('off')
            ax.set_title('3D embedding')

            results_saver(D)
            plt.show()

        np.savetxt("{}.csv".format(sys.argv[1]), ed, delimiter=",")
        ###### Visualizing Propagations: 
        resize = .125
        SHAPE = (int(resize*720),int(resize*819)) # teapot image shape
        with open(os.path.join(LOG_PATH, 'shape.txt'), 'w') as f:
            f.write(f'{SHAPE[0]}\n{SHAPE[1]}')
        visualize_propagations(original_x, peak_idxs, qml_params['nProp'], qml_params['nColl'], START_IDX=3, SHAPE=SHAPE)
        print(f'Saved data to {LOG_PATH}')
       
        
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