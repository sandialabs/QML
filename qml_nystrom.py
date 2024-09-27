# File: qml_cluster.py
#
# ===========================================================
# Implementation of quantum manifold learning (QML), with  no parallelization.
# Reference:
#   "Manifold Learning via Quantum Dynamics". Akshat Kumar & Mohan Sarovar
#   arXiv:2112.11161  https://arxiv.org/abs/2112.11161
#
# propagation is performed locally determinted by nearest point
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
import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import igraph as ig
import pandas as pd
import h5py
import scipy as sp
from scipy.sparse import csr_matrix
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state, check_array
import pickle
import math
import potpourri3d as pp3d
import networkx as nx
# import umap.umap_ as umap

# np.set_printoptions(threshold=sys.maxsize)


def runDistanceCompare(qml_params, NLimit, epsilonAdjust, hAdjust, logsmallEpsilon):
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
    nProp = 10 #qml_params['nProp']
    nColl = 1 #qml_params['nColl']
    qml_params['nProp'] = nProp
    qml_params['nColl'] = nColl
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
    epsilon = np.exp(epsilonAdjust)
    h = np.exp(hAdjust)
    smallEpsilon = np.exp(logsmallEpsilon)

    # epsilon = np.exp(logepsilon)
    # h = epsilon**(1/(2+alpha))
    # epsilon = np.exp(logepsilon + epsilonAdjust)

    # load data
    try:
        # x = np.genfromtxt(qml_params['datafile'], delimiter=',')
        x = read_in_matrix(qml_params['datafile'], verbose)
        num_rows = x.shape[0]
        np.random.seed(10)
        row_indices = range(0,num_rows)

        # randomize data positions
        # Generate an array of indices representing the rows
        row_indices = np.arange(num_rows)
        # Shuffle the row indices randomly
        np.random.shuffle(row_indices)
        # Use the shuffled indices to reorder the rows of the matrix
        x = x[row_indices]
    except:
        print("Cannot open data file: " + qml_params['datafile'] + "... Exiting.")
        raise Exception("Cannot open data file")
    else:
        # Npts is the number of data points
        Npts = np.shape(x)[0]

        # # compute Euclidean squared distance matrix, modified to be from heat
        # P = x
        # solver = pp3d.PointCloudHeatSolver(P)
        # triangulator = pp3d.PointCloudLocalTriangulation(P)
        # triangulation = triangulator.get_local_triangulation()
        # print("triangulation", triangulation)
        # pointsInTraing = np.unique(triangulation)
        # print("triang point count", pointsInTraing.shape)
        # quit()
        # # Compute the geodesic distance to point 4
        # k = np.zeros((Npts,Npts))
        # for i in range(Npts):
        #     k[i,:] = np.asarray(solver.compute_distance(i))
        # compute Euclidean squared distance matrix
        kFull = spatial.distance.squareform(spatial.distance.pdist(x, 'sqeuclidean'))

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
        if (NLimit == 100):
            k = kFull
            Udt, D_normalizer, v, M, Delta, w = qmaniGetU_nnGL_single( k, dt, epsilon, verbose, trunc=0 )
            # Udt, D_normalizer = getU_TriMesh(x, k, dt, epsilon, verbose, trunc=0 )
            D_normalizer_inv = spinv(D_normalizer)
            Us = D_normalizer @ Udt @ (D_normalizer_inv)
        else:
            k = kFull[:,0:NLimit]
            Us, Phi, v_n, D_normalizer_no, M_a = qmaniGetU_nnGL_nystrom_nonDelta(k, dt, x, Npts, epsilon, smallEpsilon, NLimit, verbose=0, trunc=0 )

# Propagate
        # container to store destination points after propagation
        peak_idxs = 0.0
        dist_idxs = 0.0

        # propagate from each point in dataset, and store destination points
        if PCA:
            pt = 0
            peak_idxs, dist_idxs = propagate_PCA(pt, qml_params, h, Npts, Us, PCA_map, x, k)
        else:
            pt = 10
            pt = np.where(row_indices==10)[0][0]
            print("starting point", x[pt,:])
            peak_idxs, dist_idxs = propagateSingle(pt, qml_params, h, Npts, Us, x, kFull)


# Fill in geodesic distance matrix
        # container for geodesic distances
        D = np.zeros([Npts, Npts])

        # for each of the Npts points, and for each of the nProp propagation times, and for each of the nColl propagations,
        # store the distance to the destination as the propagated time (and symmetrize D)
        pt = 0
        difDist = 100000.0
        difDistID = 0
        for pn in range(nProp):
            for ki in range(nColl):
                # if (D[pt,peak_idxs[pn,ki]]==0) | (D[pt,peak_idxs[pn,ki]]>(pn+1)*dt): # first fit
                if (D[pt,peak_idxs[pn,ki]]==0) | (difDist > dist_idxs[pn,ki]) :  # closest fit
                    D[pt,peak_idxs[pn, ki]] = (pn+1)*dt
                    D[peak_idxs[pn,ki], pt] = (pn+1)*dt
                    difDist = dist_idxs[pn,ki]
                    if (peak_idxs[pn,ki] != difDistID):
                        difDistID = peak_idxs[pn,ki]
                        difDist = 100000.0

        # set the diagonal elements to zero by force
        D[pt,pt]=0


        # output time taken
        e_time = time.time()
        print(f"QML Done. Time taken = {e_time-s_time}")


        print("ptx", peak_idxs)
        print("D", D)

        count = 0
        prevPoint = 0
        for i in range(nProp):
            if (peak_idxs[i,0] != prevPoint):
                prevPoint = peak_idxs[i,0]
                count += 1
        xScale = np.zeros((count,1))
        ids = np.zeros((count,1), dtype=int)
        count = 0
        prevPoint = 0
        for i in range(nProp):
            if (peak_idxs[i,0] != prevPoint):
                prevPoint = peak_idxs[i,0]
                xScale[count,0] = (i+1)*dt
                ids[count,0] = (peak_idxs[i,0])
                count += 1

        print("ids", ids)

        distances = np.zeros((count,4))
        # lineData = np.zeros((count,3)) # sphere
        lineData = np.zeros((count,2)) # circle
        # r0, theta0, phi0 = cartesian_to_spherical(x[0,0], x[0,1], x[0,2])
        r1, theta1 = cartesian_to_polar(x[0,0], x[0,1])
        for i in range(count):
            distances[i,0] = D[0, ids[i,0]]
            lineData[i,:] = x[ids[i,0], :]
            # spherical distances
            # r, theta, phi = cartesian_to_spherical(lineData[i,0], lineData[i,1], lineData[i,2])
            r2, theta2 = cartesian_to_polar(lineData[i,0], lineData[i,1])
            # distances[i,1] = geodesic_distance_spherical(r0, theta0, phi0, r, theta, phi)
            distances[i,1] = geodesic_distance_on_circle(r1, theta1, theta2)
            # print(i, r0, theta0, phi0, r, theta, phi)
            print(x[0,:], lineData[i,:])

    

        # = Stateful solves
        P = np.zeros((Npts,3))
        P[:, :2] = x
        solver = pp3d.PointCloudHeatSolver(P)

        # Compute the geodesic distance to point 4
        dists = np.asarray(solver.compute_distance(0))
        # dists1d = dists
        # dists1d = np.linalg.norm(dists,axis=1)
        print("heat", dists, dists.shape)
        for i in range(count):
            distances[i,2] = dists[ids[i,0]]
            


        # # Djikstra distances
        # k = 7
        # graphList = []
        # closest_points = []
        # for i, point in enumerate(x):
        #     closest_points.append(find_closest_points(x, point, k))
        # # Create an empty graph
        # for j in range(5,k):
        #     print("g ", j)
        #     G = nx.Graph()
        #     # Loop over the initial matrix
        #     for i, point in enumerate(x):
        #         count0 = 0
        #         for index, distance in closest_points[i]:
        #             if (count0 < j):
        #                 # print(index, distance, i)
        #                 G.add_edge(i, index, weight=distance)
        #                 count0 += 1
        #             else:
        #                 break
        #     graphList.append(G)
        # # print(G)
        # # print(ids)
        # # print(G.edges())
        # # print(max(nx.connected_components(G),key=len))
        # dijDists = np.zeros((count,k-5))
        # dijError = np.zeros((k-5,1))
        # for j in range(0,k-5):
        #     print("p ", j)
        #     for i in range(count):
        #         try:
        #             shortest_path_indices = nx.shortest_path(graphList[j], source=0, target=ids[i,0])
        #             dijDists[i,j] = sum(graphList[j][u][v]['weight'] for u, v in zip(shortest_path_indices, shortest_path_indices[1:]))
        #         except:
        #             dijDists[i,j] = 9999999999999
        #     dijError[j] = np.linalg.norm(dijDists[:,j] - distances[:,1])
        # dijBestK = np.argmin(dijError)
        # print("dif error ", dijError)
        # print("best k ", dijBestK)
        # distances[:,3] = dijDists[:,dijBestK]


        print("ids", ids)
        print("dist", distances)
        print("xScale", xScale)
        

        # printing to figure blah blah not used for matrix h test type test
        # # # save geodesic distance matrix to file
        # # fname = "{}.out".format(sys.argv[1])
        # # np.savetxt(fname, D, fmt='%.10f', delimiter=',')

        # difDists = np.zeros(distances.shape)
        # for i in range(distances.shape[0]):
        #     for j in range(distances.shape[1]):
        #         difDists[i,j] = abs(distances[i,j] - distances[i,1])

        # fig, ax = plt.subplots(2)
        # ax[0].plot(distances[:,1], distances[:,0], label="QML " + str(NLimit))
        # ax[0].plot(distances[:,1], distances[:,2], label="Heat")
        # # ax[0].plot(distances[:,1], distances[:,3], label="Djisktra")
        # ax[0].set_title('Geodesic Distance Methods')
        # ax[0].legend()
        # ax[1].plot(distances[:,1], difDists[:,0], label="QML " + str(NLimit))
        # ax[1].plot(distances[:,1], difDists[:,2], label="Heat")
        # # ax[1].plot(distances[:,1], difDists[:,3], label="Djisktra")
        # ax[1].legend()
        # ax[0].set_xlabel("Ground truth distance")
        # ax[0].set_ylabel("Geodesic Distance")
        # ax[1].set_xlabel("Ground truth distance")
        # ax[1].set_ylabel("Difference in distance")
        # ax[1].set_yscale("log")

        # # plt.show()
        # plt.savefig("sphereGeo_" + str(NLimit) + ".png")


        return ids, distances, xScale

def cartesian_to_spherical(x, y, z):
    r = math.sqrt(x**2 + y**2 + z**2)
    theta = math.atan2(y, x)
    phi = math.acos(z/r) + math.pi / 2
    return r, theta, phi

def find_closest_points(matrix, input_point, k=1):
    distances = []
    for i, point in enumerate(matrix):
        distance = np.sqrt(np.sum((point - input_point) ** 2))
        distances.append((i, distance))
    
    # Sort the distances in ascending order
    distances.sort(key=lambda x: x[1])
    
    # Extract the k closest points along with their distances
    closest_points = distances[:k]
    
    return closest_points

# def geodesic_distance_spherical(r1, theta1, phi1, r2, theta2, phi2):
#     # Calculate the geodesic distance on the sphere using the Haversine formula
#     delta_theta = theta2 - theta1

#     # Adjust the azimuthal angle (theta) to handle cyclic nature
#     if delta_theta > math.pi:
#         delta_theta -= 2 * math.pi
#     elif delta_theta < -math.pi:
#         delta_theta += 2 * math.pi

#     a = math.sin(phi1) * math.sin(phi2) + math.cos(phi1) * math.cos(phi2) * math.cos(delta_theta)
#     distance = r1 * math.acos(a)
#     return distance

def geodesic_distance_spherical(r1, theta1, phi1, r2, theta2, phi2):
    deltaTheta1 = abs(theta2 - theta1)
    deltaTheta2 = abs(theta1 - theta2)
    if (deltaTheta1 < deltaTheta2):
        deltaTheta = deltaTheta1
    else:
        deltaTheta = deltaTheta2

    # Calculate the geodesic distance on the sphere using the great-circle distance formula
    central_angle = math.acos(math.sin(phi1) * math.sin(phi2) + math.cos(phi1) * math.cos(phi2) * math.cos(deltaTheta))
    distance = r1 * central_angle
    return distance

def cartesian_to_polar(x, y):
    """
    Convert Cartesian coordinates (x, y) to polar coordinates (r, theta).
    Theta is given in radians.
    """
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y, x)
    return r, theta

def geodesic_distance_on_circle(radius, theta1, theta2):
    """
    Calculate the geodesic distance on a circle of given radius between
    two points defined by their angular coordinates theta1 and theta2 (in radians).
    """
    # Calculate the absolute difference between the angles
    diff = abs(theta2 - theta1)
    
    # Ensure the difference is in the range [0, 2*pi]
    diff = diff % (2 * math.pi)
    
    # The geodesic distance is the smaller of the direct distance and the wrap-around distance
    distance = min(diff, 2 * math.pi - diff)
    
    # Multiply by the radius to get the actual arc length
    geodesic_distance = radius * distance
    
    return geodesic_distance




def get_hamiltonian_nystrom(k, epsilon, n, nLimit, smallEpsilon):
    """
    Compute data-driven Hamiltonian

    Inputs:
        k: Euclidean distance matrix for dataset
        epsilon: epsilon parameter

    Outputs:
        H: data-driven Hamiltonian
    """
    k = k[:,0:nLimit]
    print("square dists", k.shape, k)
    T_e = np.exp( np.divide(k, -epsilon) )
    T = np.exp( np.divide(k[0:nLimit,0:nLimit], -epsilon) )
    print("T ", T.shape, T)
    print("T_e ", T_e.shape, T_e)
    # normalization
    D = np.matrix(T).sum(1)
    D_e = np.matrix(T_e).sum(1)
    one_over_D = sp.sparse.diags(1/np.squeeze(np.asarray(D)), format="csc")
    one_over_D_e = sp.sparse.diags(1/np.squeeze(np.asarray(D_e)), format="csc")
    M_s = np.sqrt(one_over_D) @ T @ np.sqrt(one_over_D)
    M_s_e = np.sqrt(one_over_D_e) @ T_e @ np.sqrt(one_over_D)
    M_s = one_over_D @ T @ one_over_D
    M_s_e = one_over_D_e @ T_e @ one_over_D
    # M_s_e = np.sqrt(one_over_D_e) @ T_e @ np.sqrt(one_over_D_e)
    print("M_s ", M_s)
    print("M_s_e", M_s_e)


    # second normalization to recover Markov operator
    N = np.matrix(M_s).sum(1)
    print("N ", N)
    D_normalizer = sp.sparse.diags(np.asarray(np.transpose(np.sqrt(1/N)))[0], format="csc")
    one_over_N = sp.sparse.diags(1/np.squeeze(np.asarray(N)), format="csc")
    one_over_N_sqrt = sp.sparse.diags(np.squeeze(np.asarray(np.sqrt(1/N))), format="csc")
    M_a = one_over_N_sqrt @ one_over_D @ T @ one_over_D @ one_over_N_sqrt
    Delta_a = (4/epsilon) * (np.identity(M_a.shape[0]) - M_a)
    print("Delta_a ", Delta_a)

    # if verbose>0:
    #     print("Eigendecomposition")

    # see if there should be a smaller epsilon for extended values
    epsilon = smallEpsilon

    w, v = sp.linalg.eig(M_a, left=False, right=True)

    for i in range(v.shape[1]):
        if (v[0,i] < 0):
            v[:,i] *= -1
            w[i] *= -1
    vSave = v
    idx = np.argsort(w) # sorted in ascending order
    idx = idx[::-1] # reverse order to get descending eigenvalues
    w = np.real(w[idx])
    v = v[:,idx]
    v_inv = v.conj().T

    # normalized nystrom extention
    print("w, v shapes: ", w.shape, v.shape)
    print("v ", v)
    print("w ", w)
    print("w adjust", 1.0 - epsilon * 0.25 * w)
    # nystrom propagator
    phi_tilde = np.zeros((n, nLimit), dtype='complex128')
    U = np.zeros((n,n), dtype='complex128')
    for i in range(0,nLimit): # range(nLimit,n)
        print("\n i", i)
        phi = v[:,i]
        lamb = w[i]
        print("lambda", lamb)
        # if (np.abs(1 - epsilon * 0.25 * lamb) < 1e-6):
        if (np.abs(lamb) < 1e-9):
        # if (lamb < 1e-18):
            phi = 0.0
            lamb = 1.0
            # print("lambda", lamb)

        # print("(1 - epsilon * 0.25 * lamb)", (1 - epsilon * 0.25 * lamb))
        for j in range(0,n):
            # left side adjustments
            sqrt_1overT_epsi = 1 / np.sum(T_e[j,:], 0)
            # print("sqrt_1overT_epsi ", sqrt_1overT_epsi)
            sqrt_1overM_s = np.sqrt(1 / np.sum(M_s_e[j,:], 0))
            # print("sqrt_1overM_s ", sqrt_1overM_s)
            left_adjust = sqrt_1overT_epsi * sqrt_1overM_s
            # print("left_adjust ", left_adjust)
            # right side adjustments
            right_adjust = np.sum(T,1) * np.sqrt(np.sum(M_s,1))
            # right_adjust = np.sum(T_ex,0) * np.sqrt(np.sum(M_s_ex,0))
            # print("right_adjust ", right_adjust)
            right_sum = np.sum(np.transpose(T_e[j,0:nLimit]) * (phi / right_adjust), 0)
            # print("right_sum ", right_sum)
            # phi_tilde[j,i] = (1.0 / (1.0 - epsilon * 0.25 * lamb)) * left_adjust * right_sum
            phi_tilde[j,i] = (1.0 / (lamb)) * left_adjust * right_sum
    # for i in range(0,n):
    #     phi_tilde[i,:] = phi_tilde[i,:] / np.linalg.norm(phi_tilde[i,:])
    #     # phi_tilde[:,i] = phi_tilde[:,i] / np.linalg.norm(phi_tilde[:,i])

    # # print("U comp shapes", phi_tilde.shape, (np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon))).shape, np.transpose(np.conj(phi_tilde)).shape)
    # print("middle comp", np.diag(np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon))))
    # print("mid vec", np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon)))
    # U = phi_tilde @ np.diag(np.exp(1j * dt * np.real(np.sqrt((4*np.abs(1-w))/epsilon)))) @ np.transpose(np.conj(phi_tilde))

    # reconstruct from extended eigendecomposition 
    H = np.zeros((n,n), dtype="complex128")
    w = w.astype("complex128")
    for i in range(0,nLimit):
        H += w[i] * np.outer(phi_tilde[:,i],phi_tilde[:,i])
    H = (4/epsilon) * (np.identity(np.shape(H)[0]) - H)

    # if (np.isnan(np.min(H))):
    #     print("NAN found")
    #     print("w", w)
    #     print("v", v)
    #     print("T_e", T_e)
    #     print("T_e sums", np.sum(T_e[:,:], 0))
    #     print("M_s sums", np.sum(M_s_e[:,:], 0))
    #     print("sqrt_1overT_epsi", sqrt_1overT_epsi)
    #     print("sqrt_1overM_s", sqrt_1overM_s)
    #     print("one_over_D_e", one_over_D_e)
    #     print("H", H)
    #     quit()

    # L = np.exp( np.divide(k, -epsilon) )
    # # normalization
    # D = np.matrix(L).sum(1)
    # one_over_D = sp.sparse.diags(1/np.squeeze(np.asarray(D)), format="csc")
    # La = one_over_D @ L @ one_over_D
    # # second normalization to recover Markov operator
    # Da = np.matrix(La).sum(1)
    # D_normalizer = sp.sparse.diags(np.asarray(np.transpose(1/Da))[0], format="csc")
    # M = D_normalizer @ La
    # H = (4/epsilon) * (np.identity(np.shape(M)[0]) - M)

    return H

def perform_hamiltonian_test_nystrom(qml_params, nLimit, logsmallEpsilon):
    """
    Test data-driven Hamiltonian with various values of epsilon and h.
    Funciton plots error and asks user to choose log(epsilon) and log(h) to proceed with.

    Inputs:
        qml_params: QML parameters
    Outputs:
        retval: a dictionary containing the user inputted log(epsilon) and log(h) values
    """

    smallEpsilon = np.exp(logsmallEpsilon)

    # range of parameters to test over
    logeps_v = np.arange(-8,2,0.5)
    logh_v =  np.arange(-8,2,0.5)
    Neps = np.shape(logeps_v)[0]

    # number of states to evaluate expectation over
    avg = qml_params['H_test_avg']

    # load data
    try:
        # x = np.genfromtxt(qml_params['datafile'], delimiter=',')
        x = read_in_matrix(qml_params['datafile'], qml_params['verbose'])
        num_rows = x.shape[0]
        row_indices = np.arange(num_rows)
        # Shuffle the row indices randomly
        # np.random.seed(42)
        np.random.shuffle(row_indices)
        # Use the shuffled indices to reorder the rows of the matrix
        x = x[row_indices]
    except:
        print("Cannot open data file: " + qml_params['datafile'] + "... Exiting.")
        raise Exception("Cannot open data file")
    else:
        # Npts is the number of data points
        Npts = np.shape(x)[0]
        n = Npts

        # compute Euclidean squared distance matrix
        k = spatial.distance.squareform(spatial.distance.pdist(x, 'sqeuclidean'))

        # container for storing devitations/errors
        devs = np.zeros([len(logeps_v), len(logh_v)])
        devs_s = np.zeros([len(logeps_v), len(logh_v)])

        # loop over parameters and evaluate error in expectations value of data-driven Hamiltonian under coherent state

        # loop over epsilon
        for le_i, le in enumerate(logeps_v):

            # if qml_params['verbose']:
            if np.mod(le_i, 10)==0:
                print("log epsilon = {} ({}/{})".format(le, le_i, Neps))

            epsilon = np.exp(le)

            # calculate Hamiltonian
            H = get_hamiltonian_nystrom(k, epsilon, n, nLimit, smallEpsilon)
            k_s = k[0:nLimit,0:nLimit]
            x_s = x[0:nLimit,:]
            H_s = get_hamiltonian(k_s,epsilon)

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
                    p0 = x[sorted_idx[1],:] - x[pt,:]
                    p0 = p0/np.linalg.norm(p0)

                    # formulate coherent state (in extrinsic coordinates)
                    psi0 = np.multiply( np.exp( -k[:,pt]/(2*h) ), np.exp((-1j/h) * ((x - x[pt,:]) @ np.transpose(p0))) )
                    psi0 = psi0 / np.linalg.norm(psi0)

                    # calculate error in expectation value (should be 1 since Hamiltonian approximates p^2 and |p|=1)
                    temp[ii] = np.abs((h**2) * np.inner(np.conj(psi0).T, np.matmul(H,psi0)) - 1)

                # store deviation
                devs[le_i, lh_i] = np.average(temp)

            # loop over h for subset of points
            for lh_i, lh in enumerate(logh_v):
                h = np.exp(lh)
                temp = np.zeros([avg,])

                # calculate deviations for coherent states centered at avg initial points
                for ii in range(avg):
                    # choose random initial point
                    pt = np.random.randint(0,nLimit)

                    # sort other points according to their distance from pt
                    sorted_idx = np.squeeze(np.argsort(k_s[pt,]))

                    # pick momentum as (normalized) vector to closest point
                    p0 = x_s[sorted_idx[1],:] - x_s[pt,:]
                    p0 = p0/np.linalg.norm(p0)

                    # formulate coherent state (in extrinsic coordinates)
                    psi0 = np.multiply( np.exp( -k_s[:,pt]/(2*h) ), np.exp((-1j/h) * ((x_s - x_s[pt,:]) @ np.transpose(p0))) )
                    psi0 = psi0 / np.linalg.norm(psi0)

                    # calculate error in expectation value (should be 1 since Hamiltonian approximates p^2 and |p|=1)
                    temp[ii] = np.abs((h**2) * np.inner(np.conj(psi0).T, np.matmul(H_s,psi0)) - 1)

                # store deviation
                devs_s[le_i, lh_i] = np.average(temp)

        # for i in range(devs.shape[0]):
        #     for j in range(devs.shape[1]):
        #         if (devs[i,j] > 1):
        #             devs[i,j] = 1
        # for i in range(devs.shape[0]):
        #     for j in range(devs.shape[1]):
        #         if (devs[i,j] > 1):
        #             devs[i,j] = 1

        # # plot
        # loge, logh = np.meshgrid(logeps_v, logh_v, indexing='ij')
        # fig, ax = plt.subplots()
        # print("loge", loge)
        # print("logh", logh)
        # print("devs", devs)
        # im = ax.pcolormesh(loge, logh, devs)
        # # im = ax.pcolormesh(np.transpose(logh), np.transpose(loge), devs)
        # fig.colorbar(im)

        # ax.set_xlabel('log(eps)')
        # ax.set_ylabel('log(h)')
        # ax.set_title('Deviation -- choose log(epsilon) and log(h) values')

        # plt.savefig('h_test.png')
        # plt.show()

        # retval = {}
        # entry = input('Enter log(epsilon) value: ')
        # retval['logepsilon'] = float(entry)

        # entry = input('Enter log(h) value: ')
        # retval['logh'] = float(entry)

        # return retval

        return devs, devs_s





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

    if 'clusterSize' in inp:
        qml_params['clusterSize'] = inp['clusterSize']
    else:
        qml_params['clusterSize'] = 32

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
            # print(data)
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


def qmaniGetU_nnGL(x, dt, epsilon, num_neighbors, verbose=0, trunc=0 ):
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
    # print("U entry")
    # Sample matrix of points (replace this with your actual matrix)
    points_matrix = x
    # Number of neighbors to find
    # num_neighbors = 64
    # Create a NearestNeighbors object
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='brute').fit(points_matrix)
    # print("Nearest neighbors first")
    # Find the nearest neighbors for each point
    time1 = time.time()
    distances, indices = nbrs.kneighbors(points_matrix)
    time2 = time.time()
    # print(f"QML Done. Time taken = {time2-time1}")
    # print("Nearest neighbors second")
    # Create a matrix listing each point's nearest neighbors
    nearest_neighbors_matrix = np.zeros((len(points_matrix), num_neighbors), dtype=int)
    for i in range(len(points_matrix)):
        nearest_neighbors_matrix[i] = indices[i]
    # Print the result
    print(nearest_neighbors_matrix)
    print(indices, indices.shape)
    # Create local adjacency matrices for each set of nearest neighbors
    adjacency_matrices = []
    epsilons = []
    hs = []
    commonE = 0.0
    commonH = 0.0
    for i in range(points_matrix.shape[0]):
        if (i % 1000 == 0):
            print("h select:", i)
        # Create an empty adjacency matrix
        adj_matrix = np.zeros((num_neighbors, num_neighbors), dtype=float)
        xLocal = x[indices[i]]
        k = spatial.distance.squareform(spatial.distance.pdist(xLocal, 'sqeuclidean'))

        # find best h and epi for local neighborhood
        if (i < 100000):
            localE, localH = param_hamiltonian_test(qml_params, xLocal, k)
        else:
            if (commonE == 0.0):
                arr = np.array(epsilons)
                # most_common_value = np.argmax(np.bincount(arr))
                # Find the indices of the most common value
                bin_size = 0.25
                bins = np.arange(arr.min(), arr.max() + bin_size, bin_size)
                hist, bin_edges = np.histogram(arr, bins=bins)
                # Find the bin with the maximum count
                bin_with_max_count = np.argmax(hist)
                print("bin max count", bin_with_max_count)
                # Find the most common value (midpoint of the bin with the maximum count)
                most_common_value = (bin_edges[bin_with_max_count]) # + bin_edges[bin_with_max_count + 1]) / 2
                print("most common value", most_common_value)
                # Find the indices of the most common value
                # Define a small epsilon value
                epsilon = 2e-1
                # Find the indices of the most common value with epsilon difference
                indices_most_common_value = np.where(np.abs(arr - most_common_value) < epsilon)[0]
                print("index common value", indices_most_common_value)
                localE = epsilons[indices_most_common_value[0]]
                localH = hs[indices_most_common_value[0]]
                commonE = localE
                commonH = localH
            else:
                localE = commonE
                localH = commonH

        epsilons.append(localE)
        hs.append(localH)

        # # Set edges based on the nearest neighbors for the current point
        # jj = 0
        # for j in nearest_neighbors_matrix[i,:]:
        #     kk = jj
        #     for k2 in nearest_neighbors_matrix[i,jj:num_neighbors]:
        #         adj_matrix[jj, kk] = k[jj, kk] #k[j,k2]
        #         adj_matrix[kk, jj] = k[kk, jj] #k[j,k2]
        #         # print("k[], i, j, k, jj, kk, a", k[j,k2], i, j, k2, jj, kk, adj_matrix[jj, kk])
        #         kk += 1
        #     jj += 1
        # # print("A", adj_matrix)
        adjacency_matrices.append(k)
    # print("k ", k)
    # print("A", adjacency_matrices[0])
    # print(adjacency_matrices)


    Udts = []
    D_normalizers = []
    i = 0
    for mat in adjacency_matrices:
        epsilon = np.exp(epsilons[i])
        if (i % 1000 == 0):
            print("Ucomp:", i)
            print("epsilon, ", epsilon)
        i += 1

        k = mat
        # print("k ", k)
        
        # if verbose>0:
            # print("Construct graph Laplacian")

        L = np.exp( np.divide(k, -epsilon) )
        # normalization
        D = np.matrix(L).sum(1)
        one_over_D = sp.sparse.diags(1/np.squeeze(np.asarray(D)), format="csc")
        La = one_over_D @ L @ one_over_D

        # second normalization to recover Markov operator
        Da = np.matrix(La).sum(1)
        D_normalizer = sp.sparse.diags(np.asarray(np.transpose(np.sqrt(1/Da)))[0], format="csc")
        M = D_normalizer @ La @ D_normalizer

        # if verbose>0:
            # print("Eigendecomposition")

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
            
        Udts.append(Udt)
        D_normalizers.append(D_normalizer)
        # print("Udt", Udt)
        # print("sqrt in", (4*np.abs(1-w))/epsilon)
        # print("top", (4*np.abs(1-w)))
        # print("bottem", epsilon)
        # print("sqrt out", np.sqrt((4*np.abs(1-w))/epsilon))
        # quit()


    return Udts, D_normalizers, nearest_neighbors_matrix, epsilons, hs


def qmaniGetU_Nystrom(x, dt, epsilon, num_neighbors, verbose=0, trunc=0 ):
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
    print("U entry")
    # Sample matrix of points (replace this with your actual matrix)
    points_matrix = x
    # Number of neighbors to find
    # num_neighbors = 64
    # Create a NearestNeighbors object
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='brute').fit(points_matrix)
    print("Nearest neighbors first")
    # Find the nearest neighbors for each point
    time1 = time.time()
    distances, indices = nbrs.kneighbors(points_matrix)
    time2 = time.time()
    print(f"QML Done. Time taken = {time2-time1}")
    print("Nearest neighbors second")
    # Create a matrix listing each point's nearest neighbors
    nearest_neighbors_matrix = np.zeros((len(points_matrix), num_neighbors), dtype=int)
    for i in range(len(points_matrix)):
        nearest_neighbors_matrix[i] = indices[i]
    # Print the result
    print(nearest_neighbors_matrix)
    print(indices, indices.shape)
    # Create local adjacency matrices for each set of nearest neighbors
    adjacency_matrices = []
    epsilons = []
    hs = []
    commonE = 0.0
    commonH = 0.0
    for i in range(points_matrix.shape[0]):
        if (i % 10 == 0):
            print(i)
        # Create an empty adjacency matrix
        adj_matrix = np.zeros((num_neighbors, num_neighbors), dtype=float)
        xLocal = x[indices[i]]
        k = spatial.distance.squareform(spatial.distance.pdist(xLocal, 'sqeuclidean'))

        # find best h and epi for local neighborhood
        if (i < 1000):
            localE, localH = param_hamiltonian_test(qml_params, xLocal, k)
        else:
            if (commonE == 0.0):
                arr = np.array(epsilons)
                # most_common_value = np.argmax(np.bincount(arr))
                # Find the indices of the most common value
                bin_size = 0.25
                bins = np.arange(arr.min(), arr.max() + bin_size, bin_size)
                hist, bin_edges = np.histogram(arr, bins=bins)
                # Find the bin with the maximum count
                bin_with_max_count = np.argmax(hist)
                print("bin max count", bin_with_max_count)
                # Find the most common value (midpoint of the bin with the maximum count)
                most_common_value = (bin_edges[bin_with_max_count]) # + bin_edges[bin_with_max_count + 1]) / 2
                print("most common value", most_common_value)
                # Find the indices of the most common value
                # Define a small epsilon value
                epsilon = 2e-1
                # Find the indices of the most common value with epsilon difference
                indices_most_common_value = np.where(np.abs(arr - most_common_value) < epsilon)[0]
                print("index common value", indices_most_common_value)
                localE = epsilons[indices_most_common_value[0]]
                localH = hs[indices_most_common_value[0]]
                commonE = localE
                commonH = localH
            else:
                localE = commonE
                localH = commonH

        epsilons.append(localE)
        hs.append(localH)

        # # Set edges based on the nearest neighbors for the current point
        # jj = 0
        # for j in nearest_neighbors_matrix[i,:]:
        #     kk = jj
        #     for k2 in nearest_neighbors_matrix[i,jj:num_neighbors]:
        #         adj_matrix[jj, kk] = k[jj, kk] #k[j,k2]
        #         adj_matrix[kk, jj] = k[kk, jj] #k[j,k2]
        #         # print("k[], i, j, k, jj, kk, a", k[j,k2], i, j, k2, jj, kk, adj_matrix[jj, kk])
        #         kk += 1
        #     jj += 1
        # # print("A", adj_matrix)
        adjacency_matrices.append(k)
    print("k ", k)
    print("A", adjacency_matrices[0])
    # print(adjacency_matrices)


    Udts = []
    D_normalizers = []
    Phis = []
    lambdas = []
    phis = []
    i = 0
    for mat in adjacency_matrices:
        epsilon = np.exp(epsilons[i])
        if (i % 1000 == 0):
            print(i)
        i += 1

        k = mat
        print("k ", k)
        print("epsilon, ", epsilon)
        # if verbose>0:
            # print("Construct graph Laplacian")

        L = np.exp( np.divide(k, -epsilon) )
        # normalization
        D = np.matrix(L).sum(1)
        one_over_D = sp.sparse.diags(1/np.squeeze(np.asarray(D)), format="csc")
        La = one_over_D @ L @ one_over_D

        # second normalization to recover Markov operator
        Da = np.matrix(La).sum(1)
        D_normalizer = sp.sparse.diags(np.asarray(np.transpose(np.sqrt(1/Da)))[0], format="csc")
        M = D_normalizer @ La @ D_normalizer

        # if verbose>0:
            # print("Eigendecomposition")

        w, v = sp.linalg.eig(M)
        # n = x.shape[0]
        # d = x.shape[1]
        # PhiJ = np.zeros((num_neighbors,20), dtype="complex128")
        # # nystrom extender
        # xLocal = x[indices[0]]
        # for i in range(20):
        #     print("x shape, n, d", x.shape, n, d)
        #     xID = np.random.randint(n, size=1)
        #     xSample = x[xID, :]
        #     kInput = np.power(np.linalg.norm(xLocal-xSample, ord=2, axis=1), 2)
        #     ke = np.exp(np.divide(-kInput, 2 * epsilon))
        #     print("kernel ", ke)
        #     print("shapes eigvec xl", v[0].shape, xLocal.shape)
        #     # might need this first np.multiply(v,xLocal)
        #     inner = np.divide(np.multiply(v, ke), np.sum(ke))
        #     print("inner", inner)
        #     print("in out shapes", inner.shape, np.divide(1, 1 - epsilon * w).shape)
        #     # print("phi j ", np.multiply(inner, np.divide(1, 1 - epsilon * w)))
        #     # PhiJ[:,i] = np.multiply(inner, np.divide(1, 1 - epsilon * w))
        #     print("phi j ", inner.dot(np.divide(1, 1 - epsilon * w)))
        #     PhiJ[:,i] += inner.dot(np.divide(1, 1 - epsilon * w))
        # print("after phi j ", PhiJ)


        # nystrom extension my guess
        n = x.shape[0]
        d = x.shape[1]
        Phi = np.zeros((num_neighbors,num_neighbors), dtype="complex128")
        # nystrom extender
        xLocal = x[indices[0]]
        numSamples = 40
        for i in range(numSamples):
            # print("x shape, n, d", x.shape, n, d)
            xID = np.random.randint(n, size=1)
            xSample = x[xID, :]
            kInput = np.power(np.linalg.norm(xLocal-xSample, ord=2, axis=1), 2)
            ke = np.exp(np.divide(-kInput, 2 * epsilon))
            # print("kernel ", ke)
            # print("shapes eigvec xl", v[0].shape, xLocal.shape)
            # might need this first np.multiply(v,xLocal)
            myGuess = v * ke[:, np.newaxis]
            # print("phij * ke ", myGuess)
            inner = np.divide(myGuess, np.sum(ke))
            # print("inner", inner)
            # print("in out shapes", inner.shape, np.divide(1, 1 - epsilon * w).shape)
            # print("phi j ", np.multiply(inner, np.divide(1, 1 - epsilon * w)))
            # PhiJ[:,i] = np.multiply(inner, np.divide(1, 1 - epsilon * w))
            # print("phi j ", inner.dot(np.divide(1, 1 - epsilon * w)))
            # PhiJ[:,i] += inner.dot(np.divide(1, 1 - epsilon * w))
            # print("phi j ", inner * np.divide(1, 1 - epsilon * w)[:, np.newaxis])
            Phi[:,:] += inner * np.divide(1, 1 - epsilon * w)[:, np.newaxis]
        Phi /= numSamples
        # print("after phi j ", Phi)
        Phis.append(Phi)
        lambdas.append(w)
        phis.append(v)



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
            
        Udts.append(Udt)
        D_normalizers.append(D_normalizer)
        # print("Udt", Udt)
        # print("sqrt in", (4*np.abs(1-w))/epsilon)
        # print("top", (4*np.abs(1-w)))
        # print("bottem", epsilon)
        # print("sqrt out", np.sqrt((4*np.abs(1-w))/epsilon))
        # quit()


    return Udts, D_normalizers, nearest_neighbors_matrix, epsilons, hs, Phis, lambdas, phis

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


def propagate(pt, qml_params, h, Npts, Uss, x, nearest_neighbors):
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

    # if verbose, output progress
    if verbose:
        if np.mod(pt, 100)==0:
            print("Propagating " + str(pt) + "/" + str(Npts))

    num_neighbors = qml_params['clusterSize']
    adj_matrix = np.zeros((num_neighbors, num_neighbors), dtype=float)
    # print("x shape", x.shape)
    xLocal = np.zeros((num_neighbors,x.shape[1]))
    jj = 0
    for j in nearest_neighbors[pt,:]:
        # print("nearest neighbor", j)
        xLocal[jj,:] = x[j,:]
        # kk = jj
        # for k2 in nearest_neighbors[pt,jj:num_neighbors]:
        #     adj_matrix[jj, kk] = k[j,k2]
        #     adj_matrix[kk, jj] = k[j,k2]
        #     # print("k[], i, j, k, jj, kk, a", k[j,k2], pt, j, k2, jj, kk, adj_matrix[jj, kk])
        #     kk += 1
        jj += 1
    # get Us from list according to starting point
    Us = Uss[pt]
    Npts = num_neighbors
    x = xLocal
    # k = adj_matrix
    k = spatial.distance.squareform(spatial.distance.pdist(xLocal, 'sqeuclidean'))
    ptSave = pt
    pt = 0

    # container to store the destination points after propagation
    # we do nColl propagations (each with a different momentum vector), for nProp time steps
    idx_store = np.zeros([nProp, nColl], dtype=int)
    Idx_store = np.zeros([nProp, nColl], dtype=int)


    # container for initial states (each initial state is a column in this matrix)
    psi0_coll = np.zeros([Npts,nColl],dtype=complex)

    # sort points according to Euclidean distance from starting point (pt)
    sorted_idx = np.squeeze(np.argsort(k[pt,]))
    # print("k", k)
    # print(pt, sorted_idx)



    # take the nColl closest points
    closest_pts = sorted_idx[1:nColl+1]
    # print("closest", closest_pts)

    # for each of the nColl initial states, set the momentum to be a (normalized) vector from starting point (pt) to
    # one of the closest points to it
    # print("closest dists ", np.linalg.norm(x[closest_pts,:] - x[pt,:], axis=0))
    # print("closest dists ", np.linalg.norm(x[closest_pts,:] - x[pt,:], axis=1))
    # print("initial states ", x[closest_pts,:] - x[pt,:])
    p0 = x[closest_pts,:] - x[pt,:]
    p0 = np.transpose(p0)/np.linalg.norm(p0, axis=1)

    # coherent state elements
    psi0_coll = np.transpose(np.multiply(np.exp( -k[:,pt]/(2*h) ), np.transpose(np.exp((-1j/h) * ((x - x[pt,:]) @ p0)))))

    # normalize coherent state
    psi0_coll = psi0_coll / np.linalg.norm(psi0_coll,axis=0)
    # print("initial psi ", psi0_coll)

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
            ind = np.argmax(values,axis=0)
            ind = nearest_neighbors[ptSave,ind]
            idx_store[pn,:] = ind
            # print("point, timestep, max", ptSave, pn, ind)
        else:
            ind, dist = pick_closest_to_mean(x, values, prob_thresh)
            ind = nearest_neighbors[ptSave,ind]
            idx_store[pn,:] = ind
            # print("point, timestep, max", ptSave, pn, ind, dist)

    return idx_store


def propagate_nystrom(pt, qml_params, h, Npts, Uss, x, nearest_neighbors, NPhi, lambdas, phi, dt):
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

    # if verbose, output progress
    if verbose:
        if np.mod(pt, 100)==0:
            print("Propagating " + str(pt) + "/" + str(Npts))

    num_neighbors = qml_params['clusterSize']
    adj_matrix = np.zeros((num_neighbors, num_neighbors), dtype=float)
    # print("x shape", x.shape)
    xLocal = np.zeros((num_neighbors,x.shape[1]))
    jj = 0
    for j in nearest_neighbors[pt,:]:
        print("nearest neighbor", j)
        xLocal[jj,:] = x[j,:]
        # kk = jj
        # for k2 in nearest_neighbors[pt,jj:num_neighbors]:
        #     adj_matrix[jj, kk] = k[j,k2]
        #     adj_matrix[kk, jj] = k[j,k2]
        #     # print("k[], i, j, k, jj, kk, a", k[j,k2], pt, j, k2, jj, kk, adj_matrix[jj, kk])
        #     kk += 1
        jj += 1
    # get Us from list according to starting point
    Us = Uss[pt]
    Npts = num_neighbors
    x = xLocal
    # k = adj_matrix
    k = spatial.distance.squareform(spatial.distance.pdist(xLocal, 'sqeuclidean'))
    ptSave = pt
    pt = 0

    # container to store the destination points after propagation
    # we do nColl propagations (each with a different momentum vector), for nProp time steps
    idx_store = np.zeros([nProp, nColl], dtype=int)
    Idx_store = np.zeros([nProp, nColl], dtype=int)


    # container for initial states (each initial state is a column in this matrix)
    psi0_coll = np.zeros([Npts,nColl],dtype=complex)

    # sort points according to Euclidean distance from starting point (pt)
    sorted_idx = np.squeeze(np.argsort(k[pt,]))
    # print("k", k)
    # print(pt, sorted_idx)



    # take the nColl closest points
    closest_pts = sorted_idx[1:nColl+1]
    print("closest", closest_pts)

    # for each of the nColl initial states, set the momentum to be a (normalized) vector from starting point (pt) to
    # one of the closest points to it
    # print("closest dists ", np.linalg.norm(x[closest_pts,:] - x[pt,:], axis=0))
    print("closest dists ", np.linalg.norm(x[closest_pts,:] - x[pt,:], axis=1))
    # print("initial states ", x[closest_pts,:] - x[pt,:])
    p0 = x[closest_pts,:] - x[pt,:]
    p0 = np.transpose(p0)/np.linalg.norm(p0, axis=1)

    # coherent state elements
    psi0_coll = np.transpose(np.multiply(np.exp( -k[:,pt]/(2*h) ), np.transpose(np.exp((-1j/h) * ((x - x[pt,:]) @ p0)))))
    
    # my guess for nystrom
    print("state comp shapes ", np.exp(1j * dt * np.sqrt(lambdas)).shape, psi0_coll.shape, phi.shape, NPhi.shape)
    # lambdas = lambdas.reshape((1,-1))
    leftSide = psi0_coll * np.exp(1j * dt * np.sqrt(np.transpose(lambdas)))[:, np.newaxis]
    psi0_coll = np.transpose(leftSide) @ phi @ NPhi
    psi0_coll = np.transpose(psi0_coll)
    print("state compu shape", psi0_coll.shape)

    # normalize coherent state
    psi0_coll = psi0_coll / np.linalg.norm(psi0_coll,axis=0)
    print("initial psi ", psi0_coll)

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
            ind = np.argmax(values,axis=0)
            ind = nearest_neighbors[ptSave,ind]
            idx_store[pn,:] = ind
            print("point, timestep, max", ptSave, pn, ind)
        else:
            ind, dist = pick_closest_to_mean(x, values, prob_thresh)
            ind = nearest_neighbors[ptSave,ind]
            idx_store[pn,:] = ind
            print("point, timestep, max", ptSave, pn, ind, dist)

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
    num_neighbors = qml_params["clusterSize"]
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

    if qml_params['colorfile']!=False:
        try:
            # colors = np.genfromtxt(qml_params['colorfile'], delimiter=',')
            colors = read_in_matrix(qml_params['colorfile'], qml_params['verbose'])
            # colors = np.loadtxt(open(qml_params['colorfile'], "rb"), delimiter=",", dtype=str)
            colors = np.array(colors)
        except:
            print("Cannot open color file: " + qml_params['colorfile'] + "... Exiting.")
            raise Exception("Cannot open color file")
    # # use this code for inputing a graph
    # with open(qml_params['datafile'], 'rb') as file:
    #     data = pickle.load(file)
    # result = umap.UMAP(n_components = 10)
    # result._densmap_kwds = {
    #         "lambda": np.max(result.dens_lambda),
    #         "frac": np.max(result.dens_frac),
    #         "var_shift": np.max(result.dens_var_shift),
    #         "n_neighbors": np.max(result.n_neighbors),
    #     }
    # result._populate_combined_params(self, other)
    # rng = np.random.default_rng()
    # n = data.shape[0]
    # embedding, aux_data = umap.simplicial_set_embedding(rng.standard_normal((n,4)), data, np.min(result.n_components),
    #         np.min(result.learning_rate),
    #         0.2, #np.mean(result._a),
    #         0.2, #np.mean(result._b),
    #         np.mean(result.repulsion_strength),
    #         np.mean(result.negative_sample_rate),
    #         100,
    #         'random',
    #         check_random_state(42),
    #         "euclidean",
    #         {},
    #         result.densmap,
    #         result._densmap_kwds,
    #         result.output_dens,
    #         parallel=False,
    #         verbose=bool(np.max(result.verbose)))
    # print("embed shape", embedding.shape)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2], c=colors)
    # plt.show()
    # x = embedding

    # load data
    # testNum = 1000
    try:
        print("Reading data")
        # x = np.genfromtxt(qml_params['datafile'], delimiter=',')
        # x = read_in_matrix(qml_params['datafile'], verbose)
        # downsample
        print("x shape", x.shape)
        # x = x[0:testNum,:]
    except:
        print("Cannot open data file: " + qml_params['datafile'] + "... Exiting.")
        raise Exception("Cannot open data file")
    else:
        # Npts is the number of data points
        Npts = np.shape(x)[0]

        # compute Euclidean squared distance matrix
        # print("All pairs distance comp")
        # k = spatial.distance.squareform(spatial.distance.pdist(x, 'sqeuclidean'))

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
        # Udts, D_normalizers, nearest_neighbors, epsisons, hs = qmaniGetU_nnGL(x, dt, epsilon, num_neighbors, verbose, trunc=0 )
        Udts, D_normalizers, nearest_neighbors, epsisons, hs, Phis, lambdas, phis = qmaniGetU_Nystrom(x, dt, epsilon, num_neighbors, verbose, trunc=0 )
        Uss = []
        ii = 0
        for Udt, D_normalizer in zip(Udts, D_normalizers):
            if (ii % 1000 == 0):
                print("append ", ii)
            ii += 1
            D_normalizer_inv = spinv(D_normalizer)
            Us = D_normalizer @ Udt @ (D_normalizer_inv)
            Uss.append(Us)

# Propagate
        # container to store destination points after propagation
        peak_idxs = dict()

        # propagate from each point in dataset, and store destination points
        if PCA:
            for pt in range(Npts):
                h = np.exp(hs[pt])
                peak_idxs[pt] = propagate_PCA(pt, qml_params, h, Npts, Uss, PCA_map, x, k)
        else:
            for pt in range(Npts):
                h = np.exp(hs[pt])
                # peak_idxs[pt] = propagate(pt, qml_params, h, Npts, Uss, x, nearest_neighbors)
                peak_idxs[pt] = propagate_nystrom(pt, qml_params, h, Npts, Uss, x, nearest_neighbors, Phis[pt], lambdas[pt], phis[pt], dt)


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
    logeps_v = np.arange(-10,6,0.5)
    logh_v =  np.arange(-10,6,0.5)
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
        k = spatial.distance.squareform(spatial.distance.pdist(x, 'sqeuclidean'))

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
            H = get_hamiltonian(k, epsilon)

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
                    p0 = x[sorted_idx[1],:] - x[pt,:]
                    p0 = p0/np.linalg.norm(p0)

                    # formulate coherent state (in extrinsic coordinates)
                    psi0 = np.multiply( np.exp( -k[:,pt]/(2*h) ), np.exp((-1j/h) * ((x - x[pt,:]) @ np.transpose(p0))) )
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



def qmaniGetU_nnGL_single( k, dt, epsilon, verbose=0, trunc=0 ):
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
    # for i in range(v.shape[1]):
    #     if (v[0,i] < 0):
    #         v[:,i] *= -1
    #         w[i] *= -1
    vSave = v
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

    Delta = (4/epsilon) * (np.identity(M.shape[0]) - M)
    w, vSave = sp.linalg.eig(Delta)
    return Udt, D_normalizer, vSave, M, Delta, w

def qmaniGetU_nnGLSingle_nystrom( k, dt, x, n, epsilon, verbose=0, trunc=0 ):
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

    # epsilon = 0.05
    print("square dists", k)
    L = np.exp( np.divide(k, -epsilon) )
    # normalization
    D = np.matrix(L).sum(1)
    one_over_D = sp.sparse.diags(1/np.squeeze(np.asarray(D)), format="csc")
    La = one_over_D @ L @ one_over_D

    # second normalization to recover Markov operator
    Da = np.matrix(La).sum(1)
    D_normalizer = sp.sparse.diags(np.asarray(np.transpose(np.sqrt(1/Da)))[0], format="csc")
    M = D_normalizer @ La @ D_normalizer

    # delta computation for nystrom
    D = np.sum(L,1)
    one_over_D = sp.sparse.diags(1.0/np.squeeze(np.asarray(D)), format="csc") # sqrt the D
    La_ = one_over_D * L
    Delta_a = (4/epsilon)*(np.eye(La_.shape[0]) - La_)

    if verbose>0:
        print("Eigendecomposition")


    w, v = sp.linalg.eig(Delta_a, left=False, right=True)
    vSave = v
    wSave = w
    print("w before", w)
    # w = (4*np.abs(1-w))/epsilon
    # w = (4*(1-w))/epsilon

    # idx = np.argsort(w) # sorted in ascending order
    # idx = idx[::-1] # reverse order to get descending eigenvalues
    # w = np.real(w[idx])
    # v = v[:,idx]
    # v_inv = v.conj().T
    # standardize the eigenvectors to leading positive values
    # for i in range(v.shape[1]):
        # if (v[0,i] < 0):
        #     v[:,i] *= -1
        #     w[i] *= -1
        # if (v[i,0] < 0):
        #     v[i,:] *= -1
        #     w[i] *= -1

    print("w, v shapes: ", w.shape, v.shape)
    # nystrom propagator
    # n = 100 # repolace with input parameter to full dataset
    # x = np.zeros((1000,1000))
    nLimit = k.shape[0]
    Phi = np.zeros((n, nLimit), dtype='complex128')
    U = np.zeros((n,n), dtype='complex128')
    Phi[0:nLimit,0:nLimit] = v
    for i in range(nLimit,n):
            # compute k_epi for each x < (0,N) and current x_i
            # nLimit is N from the email while n is Nbar
            print("x shape ", x.shape)
            kInput = np.exp(np.power(np.linalg.norm(x[0:nLimit,:]-x[i,:], ord=2, axis=1), 2) / (-epsilon))
            # kInput = np.exp(spatial.distance.cdist(x[0:nLimit,:], x[i,:], 'sqeuclidean') / (-epsilon))
            # kInput = np.exp(np.power(np.linalg.norm(x[0:nLimit,:]-x[i,:], ord=2, axis=1), 1) / (2*epsilon))
            # kInput = np.power(np.linalg.norm(x[0:nLimit,:]-x[i,:], ord=2, axis=1), 1)
            print("kInput shape ", kInput.shape)
            # Scale each k_epi by its respective dimensions for every eigenvector
            numerator = np.dot(kInput.T, v.T)
            print("numerator shape ", numerator.shape)
            # divide by sum of k_epi
            rightSide = np.divide(numerator, np.sum(kInput))
            print("rightSide shape ", rightSide.shape)
            # Multiply by scaled eigenvalue
            productFull = (1 / (1 - (epsilon * w * 0.25))) * rightSide
            print("productFull shape ", productFull.shape)
            Phi[i,:] = productFull
    for j in range(nLimit):
        # print("exp dtype", np.exp(1j * dt * np.sqrt(w[j])).dtype)
        # print("out phi", np.outer(Phi[:,j], Phi[:,j]).dtype)
        # print("out phi", np.outer(Phi[:,j], Phi[:,j]).astype(np.complex128).dtype)
        # U += np.exp(1j * dt * np.sqrt(w[j])) * np.outer(Phi[:,j], Phi[:,j]).astype(np.complex128)
        # U += np.exp( 1j*dt*np.real(np.sqrt((4*np.abs(1-w[j]))/epsilon)) ) * np.outer(Phi[:,j], Phi[:,j]).astype(np.complex128)
        # U += np.exp( 1j*dt*np.sqrt((4*np.abs(1-w[j]))/epsilon) ) * np.outer(Phi[:,j], Phi[:,j]).astype(np.complex128)
        U += np.exp( 1j*dt*np.sqrt(w[j])) * np.outer(Phi[:,j], Phi[:,j]).astype(np.complex128)
    print("U shape", U.shape)
    # print("w ", w)
    # quit()

    # # normalized nystrom extention
    # print("w, v shapes: ", w.shape, v.shape)
    # print("v ", v)
    # print("w ", w)
    # # nystrom propagator
    # # n = 100 # repolace with input parameter to full dataset
    # # x = np.zeros((1000,1000))
    # nLimit = k.shape[0]
    # Phi = np.zeros((n, nLimit), dtype='complex128')
    # U = np.zeros((n,n), dtype='complex128')
    # Phi[0:nLimit,0:nLimit] = v
    # for i in range(0,n): # range(nLimit,n)
    #         print("\n i", i)
    #         # print("data", x.shape, x)
    #         # print("dists to new point", np.power(np.linalg.norm(x[0:nLimit,:]-x[i,:], ord=2, axis=1), 2))
    #         kL_overN = np.exp(np.power(np.linalg.norm(x[0:nLimit,:]-x[i,:], ord=2, axis=1), 2) / (-epsilon))
    #         print("kL_overN ", kL_overN.shape, kL_overN)
    #         # print("epsilon", epsilon)
    #         kL_sum_sqrt = np.sqrt(np.divide(1,np.sum(kL_overN)))
    #         print("kL_sum_sqrt ", kL_sum_sqrt.shape, kL_sum_sqrt)
    #         # print("all pairs norm", spatial.distance.squareform(spatial.distance.pdist(x[0:nLimit,:], 'sqeuclidean')).shape, spatial.distance.squareform(spatial.distance.pdist(x[0:nLimit,:], 'sqeuclidean')))
    #         # print("Ex all pairs norm", spatial.distance.squareform(spatial.distance.pdist(x, 'sqeuclidean')).shape, spatial.distance.squareform(spatial.distance.pdist(x, 'sqeuclidean')))
    #         k_xl = np.exp(spatial.distance.squareform(spatial.distance.pdist(x[0:nLimit,:], 'sqeuclidean')) / (-epsilon))
    #         kL_xl = np.sqrt(np.sum(k_xl, axis=1))
    #         # kL_xl = np.sqrt(np.sum(k_xl - np.trace(k_xl), axis=1))
    #         print("k_xl", k_xl)
    #         print("kL_xl", kL_xl.shape, kL_xl)
    #         # print("kl_xl before exp", spatial.distance.squareform(spatial.distance.pdist(x[0:nLimit,:], 'sqeuclidean')) / (-epsilon))
    #         adjusted_eigenvectors = np.divide(v.T, kL_xl).T
    #         # print("eigenvec", v.shape, v)
    #         print("adjusted eigenvec", adjusted_eigenvectors.shape, adjusted_eigenvectors)
    #         # print("before sum", (kL_overN * adjusted_eigenvectors).T )
    #         k_phi_sum = np.sum((kL_overN * adjusted_eigenvectors.T).T, axis=0)
    #         print("k_phi_sum", k_phi_sum.shape, k_phi_sum)
    #         # print("1 / epi * eigval", 1 / (1 - epsilon * w))
    #         # print("sumsqrt * k_phi_sum", np.multiply(kL_sum_sqrt, k_phi_sum))
    #         rightHandSide = np.multiply(np.multiply(1 / (1 - epsilon * w * 0.25), kL_sum_sqrt), k_phi_sum)
    #         print("rhs", rightHandSide.shape, rightHandSide)
    #         Phi[i,:] = rightHandSide
    # for j in range(nLimit):
    #     print("add U ", j)
    #     # U += np.exp( 1j*dt*np.real(np.sqrt((4*np.abs(1-w[j]))/epsilon)) ) * np.outer(Phi[:,j], Phi[:,j]).astype(np.complex128)
    #     # U += np.exp( 1j*dt*np.sqrt((4*np.abs(1-w[j]))/epsilon) ) * np.outer(Phi[:,j], Phi[:,j]).astype(np.complex128)
    #     U += np.exp( 1j*dt*np.sqrt(w[j])) * np.outer(Phi[j,:], Phi[j,:]).astype(np.complex128)
    # print("U shape", U.shape)
    # print("w ", w)
    # quit()
    # D_normalizer_inv = spinv(D_normalizer)
    # U = D_normalizer_inv * U * D_normalizer

    # normalized nystrom extention
    print("w, v shapes: ", w.shape, v.shape)
    print("v ", v)
    print("w ", w)
    # nystrom propagator
    # n = 100 # repolace with input parameter to full dataset
    # x = np.zeros((1000,1000))
    nLimit = k.shape[0]
    Phi = np.zeros((n, nLimit), dtype='complex128')
    U = np.zeros((n,n), dtype='complex128')
    # Phi[0:nLimit,0:nLimit] = v
    for i in range(0,n): # range(nLimit,n)
            print("\n i", i)
            # print("data", x.shape, x)
            # print("dists to new point", np.power(np.linalg.norm(x[0:nLimit,:]-x[i,:], ord=2, axis=1), 2))
            kL_overN = np.exp(np.power(np.linalg.norm(x[0:nLimit,:]-x[i,:], ord=2, axis=1), 2) / (-epsilon))
            # print("kL_overN ", kL_overN.shape, kL_overN)
            # print("epsilon", epsilon)
            # print("all pairs norm", spatial.distance.squareform(spatial.distance.pdist(x[0:nLimit,:], 'sqeuclidean')).shape, spatial.distance.squareform(spatial.distance.pdist(x[0:nLimit,:], 'sqeuclidean')))
            # print("Ex all pairs norm", spatial.distance.squareform(spatial.distance.pdist(x, 'sqeuclidean')).shape, spatial.distance.squareform(spatial.distance.pdist(x, 'sqeuclidean')))
            # kL_xl = np.sqrt(np.sum(k_xl - np.trace(k_xl), axis=1))
            # print("kl_xl before exp", spatial.distance.squareform(spatial.distance.pdist(x[0:nLimit,:], 'sqeuclidean')) / (-epsilon))
            adjusted_eigenvectors = np.multiply(v.T, kL_overN).T
            k_phi_sum = np.sum(adjusted_eigenvectors,0)
            # print("eigenvec", v.shape, v)
            # print("adjusted eigenvec", adjusted_eigenvectors.shape, adjusted_eigenvectors)
            # print("before sum", (kL_overN * adjusted_eigenvectors).T )
            # k_phi_sum = np.sum((kL_overN * adjusted_eigenvectors.T).T, axis=0)
            # print("k_phi_sum", k_phi_sum.shape, k_phi_sum)
            # print("1 / epi * eigval", 1 / (1 - epsilon * w))
            # print("sumsqrt * k_phi_sum", np.multiply(kL_sum_sqrt, k_phi_sum))
            kL_sum_inv = np.divide(1,np.sum(kL_overN, 0))
            # print("kl sum inv", kL_sum_inv.shape, kL_sum_inv)
            rightHandSide = np.multiply(np.multiply(1 / (1 - epsilon * w * 0.25), kL_sum_inv), k_phi_sum)
            # print("rhs", rightHandSide.shape, rightHandSide)
            Phi[i,:] = rightHandSide
    for j in range(nLimit):
        print("add U ", j)
        # U += np.exp( 1j*dt*np.real(np.sqrt((4*np.abs(1-w[j]))/epsilon)) ) * np.outer(Phi[:,j], Phi[:,j]).astype(np.complex128)
        # U += np.exp( 1j*dt*np.sqrt((4*np.abs(1-w[j]))/epsilon) ) * np.outer(Phi[:,j], Phi[:,j]).astype(np.complex128)
        # U += np.exp( 1j*dt*np.real(np.sqrt(w[j]))) * np.outer(Phi[:,j], Phi[:,j]).astype(np.complex128)
        # U += np.exp( 1j*dt*np.sqrt((0.25*np.abs(1-w[j]))*epsilon) ) * np.outer(Phi[:,j], Phi[:,j]).astype(np.complex128)
        # U += np.exp( 1j*dt*np.real(np.sqrt(w[j]))) * np.outer(Phi[j,:], Phi[j,:]).astype(np.complex128)
        # U += np.exp( 1j*dt*np.sqrt((4*np.abs(1-w[j]))/epsilon) ) * np.outer(Phi[j,:], Phi[j,:]).astype(np.complex128)
        # U += np.exp( 1j*dt*np.sqrt((4*np.abs(1-wSave[j]))/epsilon) ) * np.outer(Phi[j,:], Phi[j,:]).astype(np.complex128)
        # U += np.exp( 1j*dt*np.sqrt((4*np.abs(1-wSave[j]))/epsilon) ) * np.outer(Phi[:,j], Phi[:,j]).astype(np.complex128)
        U += np.exp( 1j*dt*np.sqrt((4*np.abs(1-wSave[j]))/epsilon) ) * np.outer(Phi[j,:], Phi[j,:]).astype(np.complex128)

    print("U shape", U.shape)
    # print("w ", w)
    # quit()
    # D_normalizer_inv = spinv(D_normalizer)
    # U = D_normalizer_inv * U * D_normalizer

    # v = vSave
    # idx = np.argsort(w) # sorted in ascending order
    # idx = idx[::-1] # reverse order to get descending eigenvalues
    # w = np.real(w[idx])
    # v = v[:,idx]
    # v_inv = v.conj().T


    # if trunc>0:
    #     print("Doing spectral truncation to", trunc )
    #     wt = w[:trunc]
    #     vt = v[:,:trunc]
    #     v_invt = v_inv[:trunc,:]

    #     M_new = sp.sparse.diags( np.exp( 1j*dt*np.real(np.sqrt((4*(1-wt))/epsilon)) ) )

    #     Udt = vt @ M_new @ v_invt
    # else:
    #     M_new = sp.sparse.diags( np.exp( 1j*dt*np.real(np.sqrt((4*np.abs(1-w))/epsilon)) ), format="csc" )

    #     Udt = v @ M_new @ v_inv

    # return Udt, D_normalizer, epsilon, hParam
    return U, Phi, v




def qmaniGetU_nnGL_nystrom_first_expression( k, dt, x, n, epsilon, nLimit, verbose=0, trunc=0 ):
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

    # testing params
    # epsilon = 0.5
    # dt = 1
        
    # nLimit = k.shape[1]

    print("square dists", k.shape, k)
    T = np.exp( np.divide(k[0:nLimit,0:nLimit], -epsilon) )
    T_e = np.exp( np.divide(k, -epsilon) )
    print("T ", T.shape, T)
    print("T_e ", T_e.shape, T_e)
    # normalization
    D = np.matrix(T).sum(1)
    one_over_D = sp.sparse.diags(1/np.squeeze(np.asarray(D)), format="csc")
    M_a = one_over_D @ T
    print("M_s ", M_a)

    if verbose>0:
        print("Eigendecomposition")

    Delta_a = (4/epsilon) * (np.identity(M_a.shape[0]) - M_a)
    w, v = sp.linalg.eig(Delta_a, left=False, right=True)
    # w, v = sp.linalg.eig(Delta_s, left=True, right=False)
    index = w.argsort()[::-1]
    index = np.flip(index)
    w = w[index]
    v = v[:,index]
    wDiag = np.diag(w)
    vSave = v
    wSave = w
    print("w before", w)

    w_ma, v_ma = sp.linalg.eig(M_a, left=False, right=True)
    print("v_ma ", v_ma)
    print("w_ma ", w_ma)

    # normalized nystrom extention
    print("w, v shapes: ", w.shape, v.shape)
    print("v ", v)
    print("w ", w)
    # nystrom propagator
    phi_tilde = np.zeros((n, nLimit), dtype='complex128')
    U = np.zeros((n,n), dtype='complex128')
    for i in range(0,nLimit): # range(nLimit,n)
        print("\n i", i)
        phi = v[:,i]
        lamb = w[i]
        if ((1 - epsilon * 0.25 * lamb) < 1e-14):
            print("problem eigenvector", phi)
            phi = 0.0
            lamb = 0.0
        # print("(1 - epsilon * 0.25 * lamb)", (1 - epsilon * 0.25 * lamb))
        for j in range(0,n):
            # right side adjustments
            right_adjust = np.sum(T_e[j,:],0)
            # print("right_adjust ", right_adjust)
            right_sum = np.sum((np.transpose(T_e[j,:]) * phi) / right_adjust, 0)
            # print("T_e j", T_e[j,:])
            # print("phi", phi)
            print("right_sum ", right_sum)
            if (j < nLimit):
                phi_tilde[j,i] = (1.0 / (1.0 - epsilon * 0.25 * lamb)) * right_sum
            else:
                # phi_tilde[j,i] = (1.0 / n * (1.0 - epsilon * 0.25 * lamb)) * right_sum
                phi_tilde[j,i] = (1.0 / (1.0 - epsilon * 0.25 * lamb)) * right_sum
                if ((1 - epsilon * 0.25 * lamb) < 1e-9):
                    print("x", x.shape)
                    distances = np.linalg.norm(x[0:nLimit,:] - x[j,:], axis=1)
                    closest_index = np.argmin(distances)
                    print("closest", closest_index, j)
                    phi_tilde[j,i] = phi_tilde[closest_index,i]
            print("phi_tilde[j,i]", phi_tilde[j,i], j, i)
            print("denom", 1 - epsilon * 0.25 * lamb)
    # for i in range(nLimit,n):
    #     phi_tilde[i,:] = phi_tilde[i,:] / np.linalg.norm(phi_tilde[i,:])
    # print("U comp shapes", phi_tilde.shape, (np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon))).shape, np.transpose(np.conj(phi_tilde)).shape)
    print("middle comp", np.diag(np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon))))
    print("mid vec", np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon)))
    U = phi_tilde @ np.diag(np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon))) @ np.transpose(np.conj(phi_tilde))

    print("phi norms", np.linalg.norm(phi_tilde, 2, axis=1))
    print("T norms", np.linalg.norm(T_e, 2, axis=1))
    print("T norms cols", np.linalg.norm(T_e, 2, axis=0))
    print("v norm row", np.linalg.norm(v, 2, axis=1))
    print("v norm col", np.linalg.norm(v, 2, axis=0))
    print("v sum row", np.linalg.norm(v, 1, axis=1))
    print("v sum col", np.linalg.norm(v, 1, axis=0))

    print("v", v)
    print("phi_tilde", phi_tilde)
    print("phi diff", v - phi_tilde[0:nLimit,0:nLimit])
    print("phi max diff", np.max(v - phi_tilde[0:nLimit,0:nLimit]))
    print("U ", U.shape, U)

    return U, phi_tilde, v

def qmaniGetU_nnGL_nystrom_first_expression_full( k, dt, x, n, epsilon, nLimit, verbose=0, trunc=0 ):
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

    # testing params
    # epsilon = 0.5
    # dt = 1
        
    # nLimit = k.shape[1]

    print("square dists", k.shape, k)
    T = np.exp( np.divide(k[0:nLimit,0:nLimit], -epsilon) )
    T_e = np.exp( np.divide(k, -epsilon) )
    print("T ", T.shape, T)
    print("T_e ", T_e.shape, T_e)
    # normalization
    D = np.matrix(T).sum(1)
    one_over_D = sp.sparse.diags(1/np.squeeze(np.asarray(D)), format="csc")
    M_a = one_over_D @ T
    print("M_s ", M_a)

    if verbose>0:
        print("Eigendecomposition")

    Delta_a = (4/epsilon) * (np.identity(M_a.shape[0]) - M_a)
    w, v = sp.linalg.eig(Delta_a, left=False, right=True)
    # w, v = sp.linalg.eig(Delta_s, left=True, right=False)
    index = w.argsort()[::-1]
    index = np.flip(index)
    w = w[index]
    v = v[:,index]
    wDiag = np.diag(w)
    vSave = v
    wSave = w
    print("w before", w)

    # normalized nystrom extention
    print("w, v shapes: ", w.shape, v.shape)
    print("v ", v)
    print("w ", w)
    # nystrom propagator
    phi_tilde = np.zeros((n, nLimit), dtype='complex128')
    U = np.zeros((n,n), dtype='complex128')
    for i in range(0,nLimit): # range(nLimit,n)
        print("\n i", i)
        phi = v[:,i]
        lamb = w[i]
        if ((1 - epsilon * 0.25 * lamb) < 1e-15):
            phi = 0.0
            lamb = 0.0
        # print("(1 - epsilon * 0.25 * lamb)", (1 - epsilon * 0.25 * lamb))
        for j in range(0,n):
            # right side adjustments
            right_adjust = np.sum(T_e[j,:],0)
            # print("right_adjust ", right_adjust)
            right_sum = np.sum(np.transpose(T_e[j,:]) * (phi / right_adjust), 0)
            # print("T_e j", T_e[j,:])
            # print("phi", phi)
            # print("right_sum ", right_sum)
            phi_tilde[j,i] = (1.0 / (1.0 - epsilon * 0.25 * lamb)) * right_sum
            # print("phi_tilde[j,i]", phi_tilde[j,i], j, i)
            # print("denom", 1 - epsilon * 0.25 * lamb)
    # for i in range(nLimit,n):
    #     phi_tilde[i,:] = phi_tilde[i,:] / np.linalg.norm(phi_tilde[i,:])
    # print("U comp shapes", phi_tilde.shape, (np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon))).shape, np.transpose(np.conj(phi_tilde)).shape)
    print("middle comp", np.diag(np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon))))
    print("mid vec", np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon)))
    U = phi_tilde @ np.diag(np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon))) @ np.transpose(np.conj(phi_tilde))

    print("phi norms", np.linalg.norm(phi_tilde, 2, axis=1))
    print("T norms", np.linalg.norm(T_e, 2, axis=1))
    print("T norms cols", np.linalg.norm(T_e, 2, axis=0))
    print("v norm row", np.linalg.norm(v, 2, axis=1))
    print("v norm col", np.linalg.norm(v, 2, axis=0))
    print("v sum row", np.linalg.norm(v, 1, axis=1))
    print("v sum col", np.linalg.norm(v, 1, axis=0))

    print("v", v)
    print("phi_tilde", phi_tilde)
    print("phi diff", v - phi_tilde[0:nLimit,0:nLimit])
    print("U ", U.shape, U)

    return U, phi_tilde, v






def qmaniGetU_nnGL_nystrom_first_normalization( k, dt, x, n, epsilon, nLimit, verbose=0, trunc=0 ):
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

    # testing params
    # epsilon = 0.5
    # dt = 1
        
    # nLimit = k.shape[1]

    print("square dists", k.shape, k)
    T_e = np.exp( np.divide(k, -epsilon) )
    T = np.exp( np.divide(k[0:nLimit,0:nLimit], -epsilon) )
    print("T ", T.shape, T)
    print("T_e ", T_e.shape, T_e)
    # normalization
    D = np.matrix(T).sum(1)
    one_over_D = sp.sparse.diags(1/np.squeeze(np.asarray(D)), format="csc")
    M_s = np.sqrt(one_over_D) @ T @ np.sqrt(one_over_D)
    print("M_s ", M_s)

    if verbose>0:
        print("Eigendecomposition")

    Delta_s = (4/epsilon) * (np.identity(M_s.shape[0]) - M_s)
    w, v = sp.linalg.eig(Delta_s, left=False, right=True)
    # w, v = sp.linalg.eig(Delta_s, left=True, right=False)
    index = w.argsort()[::-1]
    index = np.flip(index)
    w = w[index]
    v = v[:,index]
    wDiag = np.diag(w)
    vSave = v
    wSave = w
    print("w before", w)

    # check for zero eigenvalues
    w_ms, v_ms = sp.linalg.eig(M_s, left=False, right=True)
    print("w_ms", w_ms)

    # normalized nystrom extention
    print("w, v shapes: ", w.shape, v.shape)
    print("v ", v)
    print("w ", w)
    # nystrom propagator
    phi_tilde = np.zeros((n, nLimit), dtype='complex128')
    U = np.zeros((n,n), dtype='complex128')
    for i in range(0,nLimit): # range(nLimit,n)
        print("\n i", i)
        phi = v[:,i]
        lamb = w[i]
        if ((1 - epsilon * 0.25 * lamb) < 1e-16):
            phi = 0.0
            lamb = 0.0
        # print("(1 - epsilon * 0.25 * lamb)", (1 - epsilon * 0.25 * lamb))
        for j in range(0,n):
            # left side adjustments
            sqrt_1overT_epsi = np.sqrt(1 / np.sum(T_e[j,:], 0))
            # print("sqrt_1overT_epsi ", sqrt_1overT_epsi)
            sqrt_1overM_s = 1 #np.sqrt(1 / np.sum(M_s_e[j,:], 0))
            # print("sqrt_1overM_s ", sqrt_1overM_s)
            left_adjust = sqrt_1overT_epsi * sqrt_1overM_s 
            # print("left_adjust ", left_adjust)
            # right side adjustments
            right_adjust = np.sqrt(np.sum(T,1) * 1) #np.sqrt(np.sum(M_s,1))
            # right_adjust = np.sum(T_ex,0) * np.sqrt(np.sum(M_s_ex,0))
            # print("right_adjust ", right_adjust)
            right_sum = np.sum(np.transpose(T_e[j,:]) * (phi / right_adjust), 0)
            # print("T_e j", T_e[j,:])
            # print("phi", phi)
            # print("lamb ", lamb)
            # print("right_sum ", right_sum)
            phi_tilde[j,i] = (1.0 / (1.0 - epsilon * 0.25 * lamb)) * left_adjust * right_sum
            # print("phi_tilde[j,i]", phi_tilde[j,i], j, i)
            # print("denom", 1 - epsilon * 0.25 * lamb)
            if (j >= nLimit):
                # print("j nLimit", j, nLimit)
                if ((1 - epsilon * 0.25 * lamb) < 1e-2):
                    print("x", x.shape)
                    distances = np.linalg.norm(x[0:nLimit,:] - x[j,:], axis=1)
                    closest_index = np.argmin(distances)
                    print("closest", closest_index, j)
                    phi_tilde[j,i] = phi_tilde[closest_index,i]
    # for i in range(nLimit,n):
    #     phi_tilde[i,:] = phi_tilde[i,:] / np.linalg.norm(phi_tilde[i,:])
    # print("U comp shapes", phi_tilde.shape, (np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon))).shape, np.transpose(np.conj(phi_tilde)).shape)
    print("middle comp", np.diag(np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon))))
    print("mid vec", np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon)))
    U = phi_tilde @ np.diag(np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon))) @ np.transpose(np.conj(phi_tilde))

    print("phi norms", np.linalg.norm(phi_tilde, 2, axis=1))
    print("T norms", np.linalg.norm(T_e, 2, axis=1))
    print("T norms cols", np.linalg.norm(T_e, 2, axis=0))
    print("v norm row", np.linalg.norm(v, 2, axis=1))
    print("v norm col", np.linalg.norm(v, 2, axis=0))
    print("v sum row", np.linalg.norm(v, 1, axis=1))
    print("v sum col", np.linalg.norm(v, 1, axis=0))

    print("v", v)
    print("phi_tilde", phi_tilde)
    print("phi diff", v - phi_tilde[0:nLimit,0:nLimit])
    print("U ", U.shape, U)

    return U, phi_tilde, v

def qmaniGetU_nnGL_nystrom_first_normalization_full( k, dt, x, n, epsilon, nLimit, verbose=0, trunc=0 ):
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

    # testing params
    # epsilon = 0.5
    # dt = 1
        
    # nLimit = k.shape[1]

    print("square dists", k.shape, k)
    T_e = np.exp( np.divide(k, -epsilon) )
    T = np.exp( np.divide(k[0:nLimit,0:nLimit], -epsilon) )
    print("T ", T.shape, T)
    print("T_e ", T_e.shape, T_e)
    # normalization
    D = np.matrix(T).sum(1)
    one_over_D = sp.sparse.diags(1/np.squeeze(np.asarray(D)), format="csc")
    M_s = np.sqrt(one_over_D) @ T @ np.sqrt(one_over_D)
    print("M_s ", M_s)

    if verbose>0:
        print("Eigendecomposition")

    Delta_s = (4/epsilon) * (np.identity(M_s.shape[0]) - M_s)
    w, v = sp.linalg.eig(Delta_s, left=False, right=True)
    # w, v = sp.linalg.eig(Delta_s, left=True, right=False)
    index = w.argsort()[::-1]
    index = np.flip(index)
    w = w[index]
    v = v[:,index]
    wDiag = np.diag(w)
    vSave = v
    wSave = w
    print("w before", w)

    # check for zero eigenvalues
    w_ms, v_ms = sp.linalg.eig(M_s, left=False, right=True)
    print("w_ms", w_ms)

    # normalized nystrom extention
    print("w, v shapes: ", w.shape, v.shape)
    print("v ", v)
    print("w ", w)
    # nystrom propagator
    phi_tilde = np.zeros((n, nLimit), dtype='complex128')
    U = np.zeros((n,n), dtype='complex128')
    for i in range(0,nLimit): # range(nLimit,n)
        print("\n i", i)
        phi = v[:,i]
        lamb = w[i]
        if ((1 - epsilon * 0.25 * lamb) < 1e-15):
            phi = 0.0
            lamb = 0.0
        # print("(1 - epsilon * 0.25 * lamb)", (1 - epsilon * 0.25 * lamb))
        for j in range(0,n):
            # left side adjustments
            sqrt_1overT_epsi = np.sqrt(1 / np.sum(T_e[j,:], 0))
            # print("sqrt_1overT_epsi ", sqrt_1overT_epsi)
            sqrt_1overM_s = 1 #np.sqrt(1 / np.sum(M_s_e[j,:], 0))
            # print("sqrt_1overM_s ", sqrt_1overM_s)
            left_adjust = sqrt_1overT_epsi * sqrt_1overM_s 
            # print("left_adjust ", left_adjust)
            # right side adjustments
            right_adjust = np.sqrt(np.sum(T,1) * 1) #np.sqrt(np.sum(M_s,1))
            # right_adjust = np.sum(T_ex,0) * np.sqrt(np.sum(M_s_ex,0))
            # print("right_adjust ", right_adjust)
            right_sum = np.sum(np.transpose(T_e[j,:]) * (phi / right_adjust), 0)
            # print("T_e j", T_e[j,:])
            # print("phi", phi)
            # print("right_sum ", right_sum)
            phi_tilde[j,i] = (1.0 / (1.0 - epsilon * 0.25 * lamb)) * left_adjust * right_sum
            # print("phi_tilde[j,i]", phi_tilde[j,i], j, i)
            # print("denom", 1 - epsilon * 0.25 * lamb)
    # for i in range(nLimit,n):
    #     phi_tilde[i,:] = phi_tilde[i,:] / np.linalg.norm(phi_tilde[i,:])
    # print("U comp shapes", phi_tilde.shape, (np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon))).shape, np.transpose(np.conj(phi_tilde)).shape)
    print("middle comp", np.diag(np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon))))
    print("mid vec", np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon)))
    U = phi_tilde @ np.diag(np.exp(1j * dt * np.real(np.sqrt((4*np.abs(1-w))/epsilon)))) @ np.transpose(np.conj(phi_tilde))
    # for i in range(nLimit):
    #     U += np.exp(1j * dt * np.sqrt((4*np.abs(1-w[i]))/epsilon)) * np.outer(phi_tilde[:,i],phi_tilde[:i])

    print("phi norms", np.linalg.norm(phi_tilde, 2, axis=1))
    print("T norms", np.linalg.norm(T_e, 2, axis=1))
    print("T norms cols", np.linalg.norm(T_e, 2, axis=0))
    print("v norm row", np.linalg.norm(v, 2, axis=1))
    print("v norm col", np.linalg.norm(v, 2, axis=0))
    print("v sum row", np.linalg.norm(v, 1, axis=1))
    print("v sum col", np.linalg.norm(v, 1, axis=0))

    print("v", v)
    print("phi_tilde", phi_tilde)
    print("phi diff", v - phi_tilde[0:nLimit,0:nLimit])
    print("U ", U.shape, U)

    return U, phi_tilde, v

def qmaniGetU_nnGL_nystrom_nonDelta( k, dt, x, n, epsilon, smallEpsilon, nLimit, verbose=0, trunc=0 ):
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

    # testing params
    # epsilon = 0.5
    # dt = 1

    # testing for scaled epsilon based on nLimit
    # epsilon /= 5 * float(nLimit) / float(n);
        
    # nLimit = k.shape[1]

    print("square dists", k.shape, k)
    T_e = np.exp( np.divide(k, -epsilon) )
    T = np.exp( np.divide(k[0:nLimit,0:nLimit], -epsilon) )
    print("T ", T.shape, T)
    print("T_e ", T_e.shape, T_e)
    # normalization
    D = np.matrix(T).sum(1)
    D_e = np.matrix(T_e).sum(1)
    one_over_D = sp.sparse.diags(1/np.squeeze(np.asarray(D)), format="csc")
    one_over_D_e = sp.sparse.diags(1/np.squeeze(np.asarray(D_e)), format="csc")
    M_s = np.sqrt(one_over_D) @ T @ np.sqrt(one_over_D)
    M_s_e = np.sqrt(one_over_D_e) @ T_e @ np.sqrt(one_over_D)
    M_s = one_over_D @ T @ one_over_D
    M_s_e = one_over_D_e @ T_e @ one_over_D
    # M_s_e = np.sqrt(one_over_D_e) @ T_e @ np.sqrt(one_over_D_e)
    print("M_s ", M_s)
    print("M_s_e", M_s_e)


    # second normalization to recover Markov operator
    N = np.matrix(M_s).sum(1)
    print("N ", N)
    D_normalizer = sp.sparse.diags(np.asarray(np.transpose(np.sqrt(1/N)))[0], format="csc")
    one_over_N = sp.sparse.diags(1/np.squeeze(np.asarray(N)), format="csc")
    one_over_N_sqrt = sp.sparse.diags(np.squeeze(np.asarray(np.sqrt(1/N))), format="csc")
    M_a = one_over_N_sqrt @ one_over_D @ T @ one_over_D @ one_over_N_sqrt
    # M_a = one_over_N_sqrt @ np.sqrt(one_over_D) @ T @ np.sqrt(one_over_D) @ one_over_N_sqrt
    Delta_a = (4/epsilon) * (np.identity(M_a.shape[0]) - M_a)
    print("Delta_a ", Delta_a)

    if verbose>0:
        print("Eigendecomposition")

    # see if there should be a smaller epsilon for extended values
    epsilon = smallEpsilon

    w, v = sp.linalg.eig(M_a, left=False, right=True)

    # for i in range(v.shape[1]):
    #     if (v[0,i] < 0):
    #         v[:,i] *= -1
    #         w[i] *= -1
    # vSave = v
    # idx = np.argsort(w) # sorted in ascending order
    # idx = idx[::-1] # reverse order to get descending eigenvalues
    # w = np.real(w[idx])
    # v = v[:,idx]
    # v_inv = v.conj().T

    # # w, v = sp.linalg.eig(Delta_a, left=True, right=False)
    # index = w.argsort()[::-1]
    # index = np.flip(index)
    # w = w[index]
    # v = v[:,index]
    # wDiag = np.diag(w)
    # vSave = v
    # wSave = w
    # print("w before", w)

    # normalized nystrom extention
    print("w, v shapes: ", w.shape, v.shape)
    print("v ", v)
    print("w ", w)
    print("w adjust", 1.0 - epsilon * 0.25 * w)
    # nystrom propagator
    phi_tilde = np.zeros((n, nLimit), dtype='complex128')
    U = np.zeros((n,n), dtype='complex128')
    for i in range(0,nLimit): # range(nLimit,n)
        print("\n i", i)
        phi = v[:,i]
        lamb = w[i]
        print("lambda", lamb)
        # if (np.abs(1 - epsilon * 0.25 * lamb) < 1e-6):
        if (np.abs(lamb) < 1e-9):
        # if (lamb < 1e-18):
            phi = 0.0
            lamb = 1.0
            # print("lambda", lamb)

        # print("(1 - epsilon * 0.25 * lamb)", (1 - epsilon * 0.25 * lamb))
        for j in range(0,n):
            # left side adjustments
            sqrt_1overT_epsi = 1 / np.sum(T_e[j,:], 0)
            # print("sqrt_1overT_epsi ", sqrt_1overT_epsi)
            sqrt_1overM_s = np.sqrt(1 / np.sum(M_s_e[j,:], 0))
            # print("sqrt_1overM_s ", sqrt_1overM_s)
            left_adjust = sqrt_1overT_epsi * sqrt_1overM_s
            # print("left_adjust ", left_adjust)
            # right side adjustments
            right_adjust = np.sum(T,1) * np.sqrt(np.sum(M_s,1))
            # right_adjust = np.sum(T_ex,0) * np.sqrt(np.sum(M_s_ex,0))
            # print("right_adjust ", right_adjust)
            right_sum = np.sum(np.transpose(T_e[j,0:nLimit]) * (phi / right_adjust), 0)
            # print("right_sum ", right_sum)
            # phi_tilde[j,i] = (1.0 / (1.0 - epsilon * 0.25 * lamb)) * left_adjust * right_sum
            phi_tilde[j,i] = (1.0 / (lamb)) * left_adjust * right_sum
            # print("phi_tilde[j,i]", phi_tilde[j,i], j, i)
            # print("denom", 1.0 - epsilon * 0.25 * lamb)
            # if (j >= nLimit):
            #     # print("j nLimit", j, nLimit)
            #     if ((1 - epsilon * 0.25 * lamb) < 1e-12):
            #         print("x", x.shape)
            #         distances = np.linalg.norm(x[0:nLimit,:] - x[j,:], axis=1)
            #         closest_index = np.argmin(distances)
            #         print("closest", closest_index, j)
            #         phi_tilde[j,i] = phi_tilde[closest_index,i]
        # print("phi",phi_tilde[:,i])
    # for i in range(0,n):
    #     phi_tilde[i,:] = phi_tilde[i,:] / np.linalg.norm(phi_tilde[i,:])
    #     # phi_tilde[:,i] = phi_tilde[:,i] / np.linalg.norm(phi_tilde[:,i])

    # print("U comp shapes", phi_tilde.shape, (np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon))).shape, np.transpose(np.conj(phi_tilde)).shape)
    print("middle comp", np.diag(np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon))))
    print("mid vec", np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon)))
    U = phi_tilde @ np.diag(np.exp(1j * dt * np.real(np.sqrt((4*np.abs(1-w))/epsilon)))) @ np.transpose(np.conj(phi_tilde))
    # U = v @ np.diag(np.exp(1j * dt * np.real(np.sqrt((4*np.abs(1-w))/epsilon)))) @ np.transpose(np.conj(v))
    # U = phi_tilde @ np.diag(np.exp(1j * dt * np.real(np.sqrt(np.abs(w))))) @ np.transpose(np.conj(phi_tilde))
    # for i in range(nLimit):
    #     # print(np.exp(1j * dt * np.sqrt((4*np.abs(1-w[i]))/epsilon)).shape)
    #     # print(np.outer(phi_tilde[:,i],phi_tilde[:,i]).shape)
    #     # print(phi_tilde[:,i].shape)
    #     U += np.exp(1j * dt * np.real(np.sqrt((4*np.abs(1-w[i]))/epsilon))) * np.outer(phi_tilde[:,i],phi_tilde[:,i])

    # print("v norms", np.linalg.norm(v, 2, axis=1))
    # print("phi norms", np.linalg.norm(phi_tilde, 2, axis=1))
    # print("T norms", np.linalg.norm(T_e, 2, axis=1))
    # print("Ms norms", np.linalg.norm(M_s_e, 2, axis=1))

    # np.set_printoptions(threshold=sys.maxsize)
    print("v", v.shape, v)
    print("phi_tilde", phi_tilde.shape, phi_tilde)
    print("phi diff", v - phi_tilde[0:nLimit,0:nLimit])
    print("phi norm", np.linalg.norm(v - phi_tilde[0:nLimit,0:nLimit]))
    print("n nLimit", n, nLimit)
    # print("U ", U.shape, U)

    return U, phi_tilde, v, D_normalizer, M_a


def filter_symmetric_matrix_by_eigenvalues(A, threshold=1e-3, verbose=1): 
    """ Removes rows and columns contributing to near-zero eigenvalues 
    from a symmetric matrix, keeping it symmetric. 
    Parameters: A (numpy.ndarray): Input symmetric matrix. 
    threshold (float): Threshold to identify near-zero eigenvalues (default: 1e-5). 
    Returns: A_filtered (numpy.ndarray): Symmetric matrix with identified rows and columns removed.
    indices_to_remove (list): Indices of rows and columns that were removed. """ 
    # Ensure the input matrix is square 
    if A.shape[0] != A.shape[1]: 
        raise ValueError("Input matrix A must be square.") 
    
    # Ensure the input matrix is symmetric 
    if not np.allclose(A, A.T, atol=1e-3): 
        raise ValueError("Input matrix A must be symmetric.") 
    
    # Calculate eigenvalues and eigenvectors 
    eigenvalues, eigenvectors = np.linalg.eig(A) 
    if (verbose):
        print("M_a eigenvalues", eigenvalues)

    # Find indices of eigenvalues close to zero 
    small_indices = np.where(np.abs(eigenvalues) < threshold)[0] 

    # Initialize a list to keep track of indices to remove 
    indices_to_remove = set() 

    # Identify indices contributing to small eigenvalues 
    for i in small_indices: 
        # Find the index of the dominant component in the eigenvector 
        max_idx = np.argmax(np.abs(eigenvectors[:, i])) 
        # Add the identified index to the set 
        indices_to_remove.add(max_idx) 

    # Convert set to a sorted list of unique indices 
    indices_to_remove = sorted(indices_to_remove) 

    # Create a new matrix excluding the identified rows and columns 
    A_filtered = np.delete(A, indices_to_remove, axis=0) # Remove rows 
    A_filtered = np.delete(A_filtered, indices_to_remove, axis=1) # Remove columns 

    # Display the results 
    # print('Indices identified for removal:') 
    # print(indices_to_remove) 
    # print('Filtered symmetric matrix:') 
    # print(A_filtered) 
    return A_filtered, indices_to_remove 
def reduce_to_nonSingular( k, dt, x, n, epsilon, smallEpsilon, nLimit, verbose=1):
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
    # if verbose>0:
    #     print("Construct graph Laplacian")

    # testing params
    # epsilon = 0.5
    # dt = 1

    # testing for scaled epsilon based on nLimit
    # epsilon /= 5 * float(nLimit) / float(n);
        
    # nLimit = k.shape[1]


    if (verbose):
        print("square dists", k.shape, k)
    T_e = np.exp( np.divide(k, -epsilon) )
    T = np.exp( np.divide(k[0:nLimit,0:nLimit], -epsilon) )
    if (verbose):
        print("T ", T.shape, T)
        print("T_e ", T_e.shape, T_e)
    # normalization
    D = np.matrix(T).sum(1)
    D_e = np.matrix(T_e).sum(1)
    one_over_D = sp.sparse.diags(1/np.squeeze(np.asarray(D)), format="csc")
    one_over_D_e = sp.sparse.diags(1/np.squeeze(np.asarray(D_e)), format="csc")
    M_s = np.sqrt(one_over_D) @ T @ np.sqrt(one_over_D)
    M_s_e = np.sqrt(one_over_D_e) @ T_e @ np.sqrt(one_over_D)
    M_s = one_over_D @ T @ one_over_D
    M_s_e = one_over_D_e @ T_e @ one_over_D
    # M_s_e = np.sqrt(one_over_D_e) @ T_e @ np.sqrt(one_over_D_e)
    if (verbose):
        print("M_s ", M_s)
        print("M_s_e", M_s_e)


    # second normalization to recover Markov operator
    N = np.matrix(M_s).sum(1)
    if (verbose):
        print("N ", N)
    D_normalizer = sp.sparse.diags(np.asarray(np.transpose(np.sqrt(1/N)))[0], format="csc")
    one_over_N = sp.sparse.diags(1/np.squeeze(np.asarray(N)), format="csc")
    one_over_N_sqrt = sp.sparse.diags(np.squeeze(np.asarray(np.sqrt(1/N))), format="csc")
    M_a = one_over_N_sqrt @ one_over_D @ T @ one_over_D @ one_over_N_sqrt
    # M_a = one_over_N_sqrt @ np.sqrt(one_over_D) @ T @ np.sqrt(one_over_D) @ one_over_N_sqrt
    Delta_a = (4/epsilon) * (np.identity(M_a.shape[0]) - M_a)
    if (verbose):
        print("Delta_a ", Delta_a)

    # Example of calling the function 
    A = np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]]) # Example symmetric matrix 
    M_a_filtered, indices_to_remove = filter_symmetric_matrix_by_eigenvalues(M_a, 1e-6, 1) 

    # x_filtered = np.delete(x, indices_to_remove, axis=0)
    # x_filtered = np.delete(x_filtered, indices_to_remove, axis=1)

    if (verbose):
        print("x", x)
    n = x.shape[0]
    indices_to_keep = [i for i in range(n) if i not in indices_to_remove]
    new_order = indices_to_keep + indices_to_remove
    x_rearranged = x[new_order, :]

    if (verbose):
        print("x_rearranged", x_rearranged)
        print("indices_to_keep", indices_to_keep)
        print("indices_to_remove", indices_to_remove)

    # print('Filtered Symmetric Matrix:') 
    # print(A_filtered) 
    # print('Indices Removed:') 
    # print(indices_to_remove) 
    n = len(indices_to_keep)
    return x_rearranged, n



    if verbose>0:
        print("Eigendecomposition")

    # see if there should be a smaller epsilon for extended values
    epsilon = smallEpsilon

    w, v = sp.linalg.eig(M_a, left=False, right=True)

    for i in range(v.shape[1]):
        if (v[0,i] < 0):
            v[:,i] *= -1
            w[i] *= -1
    vSave = v
    idx = np.argsort(w) # sorted in ascending order
    idx = idx[::-1] # reverse order to get descending eigenvalues
    w = np.real(w[idx])
    v = v[:,idx]
    v_inv = v.conj().T

    # # w, v = sp.linalg.eig(Delta_a, left=True, right=False)
    # index = w.argsort()[::-1]
    # index = np.flip(index)
    # w = w[index]
    # v = v[:,index]
    # wDiag = np.diag(w)
    # vSave = v
    # wSave = w
    # print("w before", w)

    # normalized nystrom extention
    print("w, v shapes: ", w.shape, v.shape)
    print("v ", v)
    print("w ", w)
    print("w adjust", 1.0 - epsilon * 0.25 * w)
    # nystrom propagator
    phi_tilde = np.zeros((n, nLimit), dtype='complex128')
    U = np.zeros((n,n), dtype='complex128')
    for i in range(0,nLimit): # range(nLimit,n)
        print("\n i", i)
        phi = v[:,i]
        lamb = w[i]
        print("lambda", lamb)
        # if (np.abs(1 - epsilon * 0.25 * lamb) < 1e-6):
        if (np.abs(lamb) < 1e-9):
        # if (lamb < 1e-18):
            phi = 0.0
            lamb = 1.0
            # print("lambda", lamb)

        # print("(1 - epsilon * 0.25 * lamb)", (1 - epsilon * 0.25 * lamb))
        for j in range(0,n):
            # left side adjustments
            sqrt_1overT_epsi = 1 / np.sum(T_e[j,:], 0)
            # print("sqrt_1overT_epsi ", sqrt_1overT_epsi)
            sqrt_1overM_s = np.sqrt(1 / np.sum(M_s_e[j,:], 0))
            # print("sqrt_1overM_s ", sqrt_1overM_s)
            left_adjust = sqrt_1overT_epsi * sqrt_1overM_s
            # print("left_adjust ", left_adjust)
            # right side adjustments
            right_adjust = np.sum(T,1) * np.sqrt(np.sum(M_s,1))
            # right_adjust = np.sum(T_ex,0) * np.sqrt(np.sum(M_s_ex,0))
            # print("right_adjust ", right_adjust)
            right_sum = np.sum(np.transpose(T_e[j,0:nLimit]) * (phi / right_adjust), 0)
            # print("right_sum ", right_sum)
            # phi_tilde[j,i] = (1.0 / (1.0 - epsilon * 0.25 * lamb)) * left_adjust * right_sum
            phi_tilde[j,i] = (1.0 / (lamb)) * left_adjust * right_sum
            # print("phi_tilde[j,i]", phi_tilde[j,i], j, i)
            # print("denom", 1.0 - epsilon * 0.25 * lamb)
            # if (j >= nLimit):
            #     # print("j nLimit", j, nLimit)
            #     if ((1 - epsilon * 0.25 * lamb) < 1e-12):
            #         print("x", x.shape)
            #         distances = np.linalg.norm(x[0:nLimit,:] - x[j,:], axis=1)
            #         closest_index = np.argmin(distances)
            #         print("closest", closest_index, j)
            #         phi_tilde[j,i] = phi_tilde[closest_index,i]
        # print("phi",phi_tilde[:,i])
    # for i in range(0,n):
    #     phi_tilde[i,:] = phi_tilde[i,:] / np.linalg.norm(phi_tilde[i,:])
    #     # phi_tilde[:,i] = phi_tilde[:,i] / np.linalg.norm(phi_tilde[:,i])

    # print("U comp shapes", phi_tilde.shape, (np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon))).shape, np.transpose(np.conj(phi_tilde)).shape)
    print("middle comp", np.diag(np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon))))
    print("mid vec", np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon)))
    U = phi_tilde @ np.diag(np.exp(1j * dt * np.real(np.sqrt((4*np.abs(1-w))/epsilon)))) @ np.transpose(np.conj(phi_tilde))
    # U = v @ np.diag(np.exp(1j * dt * np.real(np.sqrt((4*np.abs(1-w))/epsilon)))) @ np.transpose(np.conj(v))
    # U = phi_tilde @ np.diag(np.exp(1j * dt * np.real(np.sqrt(np.abs(w))))) @ np.transpose(np.conj(phi_tilde))
    # for i in range(nLimit):
    #     # print(np.exp(1j * dt * np.sqrt((4*np.abs(1-w[i]))/epsilon)).shape)
    #     # print(np.outer(phi_tilde[:,i],phi_tilde[:,i]).shape)
    #     # print(phi_tilde[:,i].shape)
    #     U += np.exp(1j * dt * np.real(np.sqrt((4*np.abs(1-w[i]))/epsilon))) * np.outer(phi_tilde[:,i],phi_tilde[:,i])

    # print("v norms", np.linalg.norm(v, 2, axis=1))
    # print("phi norms", np.linalg.norm(phi_tilde, 2, axis=1))
    # print("T norms", np.linalg.norm(T_e, 2, axis=1))
    # print("Ms norms", np.linalg.norm(M_s_e, 2, axis=1))

    # print("v", v)
    # print("phi_tilde", phi_tilde)
    # print("phi diff", v - phi_tilde[0:nLimit,0:nLimit])
    # print("U ", U.shape, U)

    return U, phi_tilde, v, D_normalizer, M_a



# try new expression for extending full normalization
def qmaniGetU_nnGL_nystrom( k, dt, x, n, epsilon, nLimit, verbose=0, trunc=0 ):
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

    # testing params
    # epsilon = 0.5
    # dt = 1
        
    # nLimit = k.shape[1]

    print("square dists", k.shape, k)
    T_e = np.exp( np.divide(k, -epsilon) )
    T = np.exp( np.divide(k[0:nLimit,0:nLimit], -epsilon) )
    print("T ", T.shape, T)
    print("T_e ", T_e.shape, T_e)
    # normalization
    D = np.matrix(T).sum(1)
    D_e = np.matrix(T_e).sum(1)
    one_over_D = sp.sparse.diags(1/np.squeeze(np.asarray(D)), format="csc")
    one_over_D_e = sp.sparse.diags(1/np.squeeze(np.asarray(D_e)), format="csc")
    # M_s = np.sqrt(one_over_D) @ T @ np.sqrt(one_over_D)
    # M_s_e = np.sqrt(one_over_D_e) @ T_e @ np.sqrt(one_over_D)
    M_s = one_over_D @ T @ one_over_D
    M_s_e = one_over_D_e @ T_e @ one_over_D
    # M_s_e = np.sqrt(one_over_D_e) @ T_e @ np.sqrt(one_over_D_e)
    print("M_s ", M_s)
    print("M_s_e", M_s_e)


    # second normalization to recover Markov operator
    N = np.matrix(M_s).sum(1)
    print("N ", N)
    D_normalizer = sp.sparse.diags(np.asarray(np.transpose(np.sqrt(1/N)))[0], format="csc")
    one_over_N = sp.sparse.diags(1/np.squeeze(np.asarray(N)), format="csc")
    M_a = np.sqrt(one_over_N) @ one_over_D @ T @ one_over_D @ np.sqrt(one_over_N)
    Delta_a = (4/epsilon) * (np.identity(M_a.shape[0]) - M_a)
    print("Delta_a ", Delta_a)

    if verbose>0:
        print("Eigendecomposition")

    w, v = sp.linalg.eig(Delta_a, left=False, right=True)
    # w, v = sp.linalg.eig(Delta_a, left=True, right=False)
    index = w.argsort()[::-1]
    index = np.flip(index)
    w = w[index]
    v = v[:,index]
    wDiag = np.diag(w)
    vSave = v
    wSave = w
    print("w before", w)

    # normalized nystrom extention
    print("w, v shapes: ", w.shape, v.shape)
    print("v ", v)
    print("w ", w)
    print("w adjust", 1.0 - epsilon * 0.25 * w)
    # nystrom propagator
    phi_tilde = np.zeros((n, nLimit), dtype='complex128')
    U = np.zeros((n,n), dtype='complex128')
    for i in range(0,nLimit): # range(nLimit,n)
        print("\n i", i)
        phi = v[:,i]
        lamb = w[i]
        if ((1 - epsilon * 0.25 * lamb) < 1e-3):
            phi = 0.0
            lamb = 0.0

        # print("(1 - epsilon * 0.25 * lamb)", (1 - epsilon * 0.25 * lamb))
        for j in range(0,n):
            # left side adjustments
            sqrt_1overT_epsi = 1 / np.sum(T_e[j,:], 0)
            # print("sqrt_1overT_epsi ", sqrt_1overT_epsi)
            sqrt_1overM_s = np.sqrt(1 / np.sum(M_s_e[j,:], 0))
            # print("sqrt_1overM_s ", sqrt_1overM_s)
            left_adjust = sqrt_1overT_epsi * sqrt_1overM_s
            # print("left_adjust ", left_adjust)
            # right side adjustments
            right_adjust = np.sum(T,1) * np.sqrt(np.sum(M_s,1))
            # right_adjust = np.sum(T_ex,0) * np.sqrt(np.sum(M_s_ex,0))
            # print("right_adjust ", right_adjust)
            right_sum = np.sum(np.transpose(T_e[j,0:nLimit]) * (phi / right_adjust), 0)
            # print("right_sum ", right_sum)
            phi_tilde[j,i] = (1.0 / (1.0 - epsilon * 0.25 * lamb)) * left_adjust * right_sum
            # print("phi_tilde[j,i]", phi_tilde[j,i], j, i)
            print("denom", 1.0 - epsilon * 0.25 * lamb)
            if (j >= nLimit):
                # print("j nLimit", j, nLimit)
                if ((1 - epsilon * 0.25 * lamb) < 1e-12):
                    print("x", x.shape)
                    distances = np.linalg.norm(x[0:nLimit,:] - x[j,:], axis=1)
                    closest_index = np.argmin(distances)
                    print("closest", closest_index, j)
                    phi_tilde[j,i] = phi_tilde[closest_index,i]
    # for i in range(0,n):
    #     phi_tilde[i,:] = phi_tilde[i,:] / np.linalg.norm(phi_tilde[i,:])
    #     # phi_tilde[:,i] = phi_tilde[:,i] / np.linalg.norm(phi_tilde[:,i])
    # print("U comp shapes", phi_tilde.shape, (np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon))).shape, np.transpose(np.conj(phi_tilde)).shape)
    print("middle comp", np.diag(np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon))))
    print("mid vec", np.exp(1j * dt * np.sqrt((4*np.abs(1-w))/epsilon)))
    U = phi_tilde @ np.diag(np.exp(1j * dt * np.real(np.sqrt((4*np.abs(1-w))/epsilon)))) @ np.transpose(np.conj(phi_tilde))
    # U = v @ np.diag(np.exp(1j * dt * np.real(np.sqrt((4*np.abs(1-w))/epsilon)))) @ np.transpose(np.conj(v))
    # U = phi_tilde @ np.diag(np.exp(1j * dt * np.real(np.sqrt(np.abs(w))))) @ np.transpose(np.conj(phi_tilde))
    # for i in range(nLimit):
    #     # print(np.exp(1j * dt * np.sqrt((4*np.abs(1-w[i]))/epsilon)).shape)
    #     # print(np.outer(phi_tilde[:,i],phi_tilde[:,i]).shape)
    #     # print(phi_tilde[:,i].shape)
    #     U += np.exp(1j * dt * np.real(np.sqrt((4*np.abs(1-w[i]))/epsilon))) * np.outer(phi_tilde[:,i],phi_tilde[:,i])

    print("v norms", np.linalg.norm(v, 2, axis=1))
    print("phi norms", np.linalg.norm(phi_tilde, 2, axis=1))
    print("T norms", np.linalg.norm(T_e, 2, axis=1))
    print("Ms norms", np.linalg.norm(M_s_e, 2, axis=1))

    print("v", v)
    print("phi_tilde", phi_tilde)
    print("phi diff", v - phi_tilde[0:nLimit,0:nLimit])
    print("U ", U.shape, U)

    return U, phi_tilde, v, D_normalizer, Delta_a, w



def propagateSingle(pt, qml_params, h, Npts, Us, x, k):
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
    psi_store = np.zeros([x.shape[0],nProp], dtype=complex)
    psi_count = 0

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
    # print("shapes, k, x, p0", k.shape, x.shape, p0.shape)
    psi0_coll = np.transpose(np.multiply(np.exp( -k[:,pt]/(2*h) ), np.transpose(np.exp((-1j/h) * ((x - x[pt,:]) @ p0)))))
    # print("initial psi ", psi0_coll)

    # normalize coherent state
    psi0_coll = psi0_coll / np.linalg.norm(psi0_coll,axis=0)

    # propagate each of the initial states
    psi_coll = psi0_coll
    for pn in range(nProp):

        # propagate by one timestep (dt)
        psi_coll = Us @ psi_coll
        # print("state", pn, psi_coll)

        if (psi_count < nProp):
            psi_store[:,psi_count] = psi_coll[:,0]
            psi_count += 1

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
            # print("Single point, timestep, max", pt, pn, ind, dist)

    return idx_store, psi_store


def runSingle(qml_params):
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
    NLimit = 99
    # epsilon = 0.5
    try:
        # x = np.genfromtxt(qml_params['datafile'], delimiter=',')
        x = read_in_matrix(qml_params['datafile'], verbose)
        # x = x[0:400,:]
        # x = np.array([[0.1, 0.2, 0.3],
        #                 [0.4, 0.5, 0.6],
        #                 [0.7, 0.8, 0.9]])
        # x = np.array([[0.01, 0.2, 0.3],
        #                 [0.4, 0.5, 0.6],
        #                 [0.7, -0.1, 1.0]])
        # Get the number of rows in the matrix
        num_rows = x.shape[0]

        row_indices = range(0,num_rows)

        # randomize data positions
        # Generate an array of indices representing the rows
        row_indices = np.arange(num_rows)
        # Shuffle the row indices randomly
        np.random.shuffle(row_indices)
        # Use the shuffled indices to reorder the rows of the matrix
        x = x[row_indices]
        xFull = x
        Npts = np.shape(x)[0]
        logsmallEpsilon = -2
        smallEpsilon = np.exp(logsmallEpsilon)
        kFull = spatial.distance.squareform(spatial.distance.pdist(x, 'sqeuclidean'))
        # np.set_printoptions(threshold=sys.maxsize)
        print("data x full ", x.shape, x)
        for t in range(10):
            k = kFull[:,0:NLimit]
            x, new_n = reduce_to_nonSingular(k, dt, x, Npts, epsilon, smallEpsilon, NLimit, 0)
            x = x[0:new_n,:]
            Npts = new_n
            if (new_n < NLimit):
                NLimit = new_n
        print("data x reduced ", x.shape, x)
    except:
        print("Cannot open data file: " + qml_params['datafile'] + "... Exiting.")
        raise Exception("Cannot open data file")
    else:
        # Npts is the number of data points
        Npts = np.shape(x)[0]
        # compute Euclidean squared distance matrix
        # k = spatial.distance.squareform(spatial.distance.pdist(x[0:NLimit,:], 'sqeuclidean'))
        # make rectanular k for extension
        kFull = spatial.distance.squareform(spatial.distance.pdist(x, 'sqeuclidean'))
        k = kFull[:,0:NLimit]
        # k = kFull
        # x = xFull
        rows_kept = np.array([row for row in xFull if any((row == x).all(1))])
        rows_lost = np.array([row for row in xFull if not any((row == x).all(1))])
        x = np.vstack((rows_kept, rows_lost))
        print("reordered x", x)
        Npts = np.shape(x)[0]
        kFull = spatial.distance.squareform(spatial.distance.pdist(x, 'sqeuclidean'))
        k = kFull[:,0:NLimit]
        quit()

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

        # get e and h parameters from testing
        # epsilon, h = param_hamiltonian_test(qml_params, x[0:NLimit,:], k)
        # epsilonFull, hFull = param_hamiltonian_test(qml_params, x, kFull)
# QPROP
        # compute quantum propagator
        Udt, D_normalizer, v, M, Delta, w = qmaniGetU_nnGL_single( kFull, dt, epsilon, verbose, trunc=0 )
        # # Udt, D_normalizer = qmaniGetU_nnGLSingle_nystrom( k, dt, epsilon, verbose, trunc=0 )
        # Uf, Phif, v, D_normalizer_n = qmaniGetU_nnGL_nystrom(kFull, dt, x, Npts, epsilon, Npts, verbose=0, trunc=0 )
        # U, Phi, v_n, D_normalizer_no, Delta_n, w_n = qmaniGetU_nnGL_nystrom(k, dt, x, Npts, epsilon, NLimit, verbose=0, trunc=0 )

        # Delta_re = np.zeros((Npts,Npts), dtype=np.complex128)
        # Delta_n_re = np.zeros((Npts,Npts), dtype=np.complex128)
        # for i in range(Npts):
        #     if (np.abs(w[i]) >  1e-3):
        #         Delta_re += w[i] * np.outer(v[:,i], v[:,i])
        #     if (np.abs(w_n[i]) >  1e-3):
        #         Delta_n_re += w_n[i] * np.outer(v_n[:,i], v_n[:,i])
        #         # Delta_re += w[i] * np.outer(v[i,:], v[i,:])
        #         # Delta_n_re += w_n[i] * np.outer(v_n[i,:], v_n[i,:])
        # print("Delta", Delta)
        # print("Delta_re", Delta_re)
        # print("Delta_n", Delta_n)
        # print("Delta_n_re", Delta_n_re)
        # print("Delta - Delta_n", Delta - Delta_n)
        # print("Delta - Delta_re", Delta - Delta_re)
        # print("Delta_n - Delta_n_re", Delta_n - Delta_n_re)
        # print("Delta_re - Delta_n_re", Delta_re - Delta_n_re)

        # w_diag = sp.sparse.diags(1/np.squeeze(np.asarray(w_n)), format="csc")
        # Delta_try = v_n * w_diag * np.linalg.inv(v_n)
        # print("Delta_try", Delta_try)
        # print("Delta_n - Delta_try", Delta_n - Delta_try)

        # try a version of nystrom that operates on m and not delta
        logsmallEpsilon = -2
        smallEpsilon = np.exp(logsmallEpsilon)
        U, Phi, v_n, D_normalizer_no, M_a = qmaniGetU_nnGL_nystrom_nonDelta(k, dt, x, Npts, epsilon, smallEpsilon, NLimit, verbose=0, trunc=0 )
        # U, Phi, v_n, D_normalizer_no, M_a = qmaniGetU_nystrom_nonSingularReduce(k, dt, x, Npts, epsilon, smallEpsilon, NLimit, verbose=0, trunc=0 )
        # U = Uf # compare original to full nystrom
        # Phi = Phif
        # NLimit = Npts

        # # comparisons of non extended matricies
        # # Uf, Phi, v = qmaniGetU_nnGL_nystrom_first_normalization_full(kFull, dt, x, Npts, epsilon, Npts, verbose=0, trunc=0 )
        # # U, Phi, v_n = qmaniGetU_nnGL_nystrom_first_normalization(k, dt, x, Npts, epsilon, NLimit, verbose=0, trunc=0 )
        # # Uf, Phi, v = qmaniGetU_nnGL_nystrom_first_expression_full(kFull, dt, x, Npts, epsilon, Npts, verbose=0, trunc=0 )
        # # U, Phi, v_n = qmaniGetU_nnGL_nystrom_first_expression(k, dt, x, Npts, epsilon, NLimit, verbose=0, trunc=0 )
        # # print("M", M)
        # # print("M_a", M_a)
        # # print("M diff", M - M_a)
        # print("eigenvector\n", v.shape, v)
        # print("phi\n", Phi.shape, Phi)
        # # Phi = Phi[0:NLimit,0:NLimit] # compare against values present in each
        # diff =  np.abs(v[:,0:NLimit] - Phi)
        # diffA = np.abs(v[:,0:NLimit]) - np.abs(Phi)
        # print("diff\n", np.linalg.norm(v[:,0:NLimit] - Phi))
        # # print("phi / diff", np.divide(Phi, v[:,0:NLimit] - Phi))
        # # print("eig / diff", np.divide(v[:,0:NLimit], v[:,0:NLimit] - Phi))
        # print("max difference ", diff[np.unravel_index(diff.argmax(), diff.shape)])
        # print("max diff index ", np.unravel_index(diff.argmax(), diff.shape))
        # print("values ", v[np.unravel_index(diff.argmax(), diff.shape)], Phi[np.unravel_index(diff.argmax(), diff.shape)])
        # print("diffA\n", diffA)
        # print("max differenceA ", diffA[np.unravel_index(diffA.argmax(), diffA.shape)])
        # print("max diffA index ", np.unravel_index(diffA.argmax(), diffA.shape))
        # print("values ", v[np.unravel_index(diffA.argmax(), diffA.shape)], Phi[np.unravel_index(diffA.argmax(), diffA.shape)])
        
        D_normalizer_inv = spinv(D_normalizer)
        D_normalizer_inv_no = spinv(D_normalizer_no)
        # U = D_normalizer @ U @ (D_normalizer_inv) # dimension mismatch for extended case
        Us = D_normalizer @ Udt @ (D_normalizer_inv)
        print("U original", Us.shape)
        print("U nystrom", U.shape)

        # # comparison of propagators for nonextended case
        # # U = D_normalizer_no @ U @ (D_normalizer_inv_no)
        # # Us = Udt
        # # print("D_normalizer", D_normalizer)
        # # print("Us", Us)
        # # print("Udt", Udt)
        # # D_normalizer_inv_n = spinv(D_normalizer_n)
        # # U = D_normalizer_n @ Uf @ (D_normalizer_inv_n)
        # print("U nystrom \n", U)
        # print("Us        \n", Us)
        # print("difference\n", Us - U)
        # print("U difference norm", np.linalg.norm(U-Us))
        # # largest sinulgar value, 2 norm
        # diff = np.abs(Us - U)
        # print("max difference ", diff[np.unravel_index(diff.argmax(), diff.shape)])
        # print("max diffU index ", np.unravel_index(diff.argmax(), diff.shape))
        # print("values ", Us[np.unravel_index(diff.argmax(), diff.shape)], U[np.unravel_index(diff.argmax(), diff.shape)])
        # diffU = np.abs(Us) - np.abs(U)
        # # print("diffUA\n", diffU)
        # print("max differenceUA ", diffU[np.unravel_index(diffU.argmax(), diffU.shape)])
        # print("max diffUA index ", np.unravel_index(diffU.argmax(), diffU.shape))
        # print("values ", Us[np.unravel_index(diffU.argmax(), diffU.shape)], U[np.unravel_index(diffU.argmax(), diffU.shape)])
        
        # Us = U
        Uf = U
        # U = Us # compare full nystrom to original propagator
        # quit()

        # fix later, construct psi better
        k = spatial.distance.squareform(spatial.distance.pdist(x, 'sqeuclidean'))

# Propagate
        # container to store destination points after propagation
        peak_idxs = dict()
        peak_idxs_n = dict()

        # propagate from each point in dataset, and store destination points
        if PCA:
            for pt in range(Npts):
                peak_idxs[pt] = propagate_PCA(pt, qml_params, h, Npts, Us, PCA_map, x, k)
        else:
            for pt in range(1):
                # print("pt", pt)
                peak_idxs_n[pt], psi_store_n = propagateSingle(pt, qml_params, h, Npts, Us, x, k)
            for pt in range(1):
                # print("pt", pt)
                peak_idxs[pt], psi_store = propagateSingle(pt, qml_params, h, Npts, U, x, k)

            # print("psi_normal", psi_store)
            # print("psi_nystrom", psi_store_n)
            print("psi_diff", psi_store-psi_store_n)
            print("Psi frobenius", np.linalg.norm(psi_store-psi_store_n))
            print("Psi largest SV", np.linalg.norm(psi_store-psi_store_n, 2))
            print("Psi shape", psi_store.shape)
            
            # plot difference of nystrom propagation and original
            timeSteps = np.ones((psi_store.shape[1]))
            for i in range(psi_store.shape[1]):
                timeSteps[i] = timeSteps[i] * dt * (i+1)
            psi_store_difs = np.linalg.norm(psi_store-psi_store_n, axis=0)
            fig, ax = plt.subplots(1)
            ax.plot(timeSteps, psi_store_difs, label="2norm(Nystrom-Original)")
            # ax[0].plot(timeSteps, distances[:,2], label="Original")
            # ax[0].plot(distances[:,1], distances[:,3], label="Djisktra")
            # ax[0].set_title('Geodesic Distance Methods')
            ax.legend()
            ax.set_xlabel('Time')
            ax.set_ylabel('Difference in Propagated State')
            ax.set_title('Propagated States with Nystrom Extension of ' + str(NLimit) + ' to ' + str(Npts) + ' Points')
            plt.savefig('Nystrom_propagateState_test.png')
            plt.show()

            for ii in range(10):
                print("Psi col ", ii, "frobenius", np.linalg.norm(psi_store[:,ii]-psi_store_n[:,ii]))

            for pt in range(Npts):
                # print("pt", pt)
                peak_idxs[pt], psi_store = propagateSingle(pt, qml_params, h, Npts, U, x, k)

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

        return D, row_indices


def param_hamiltonian_test(qml_params, x, k):
    """
    Test data-driven Hamiltonian with various values of epsilon and h.
    Funciton plots error and asks user to choose log(epsilon) and log(h) to proceed with.

    Inputs:
        qml_params: QML parameters
    Outputs:
        retval: a dictionary containing the user inputted log(epsilon) and log(h) values
    """

    # range of parameters to test over
    logeps_v = np.arange(-10.10,6,0.61)
    logh_v =  np.arange(-10.10,6,0.61)
    Neps = np.shape(logeps_v)[0]

    # number of states to evaluate expectation over
    avg = qml_params['H_test_avg']


    # Npts is the number of data points
    Npts = np.shape(x)[0]

    # container for storing devitations/errors
    devs = np.zeros([len(logeps_v), len(logh_v)])

    # loop over parameters and evaluate error in expectations value of data-driven Hamiltonian under coherent state

    # loop over epsilon
    for le_i, le in enumerate(logeps_v):

        # if qml_params['verbose']:
        #     if np.mod(le_i, 10)==0:
        #         print("log epsilon = {} ({}/{})".format(le, le_i, Neps))

        epsilon = np.exp(le)

        # calculate Hamiltonian
        H = get_hamiltonian(k, epsilon)

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
                p0 = x[sorted_idx[1],:] - x[pt,:]
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

    # # plot
    loge, logh = np.meshgrid(logeps_v, logh_v, indexing='ij')
    # fig, ax = plt.subplots()
    # im = ax.pcolormesh(loge, logh, devs, norm=LogNorm())
    # # im = ax.pcolormesh(np.transpose(logh), np.transpose(loge), devs)
    # fig.colorbar(im)

    # ax.set_xlabel('log(eps)')
    # ax.set_ylabel('log(h)')
    # ax.set_title('Deviation -- choose log(epsilon) and log(h) values')


    min_index = np.argmin(devs)
    min_row, min_col = np.unravel_index(min_index, devs.shape)
    newE = loge[min_row, min_col]
    newH = logh[min_row, min_col]
    # print("e, h ", newE, newH)
    # newA = newE / newH - 2
    # qml_params['logepsilon'] = newE
    # qml_params['alpha'] = newE / newH - 2

    # plt.show()

    return newE, newH


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

    # testNum = 1000

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
        D, row_indices = runSingle(qml_params)
        # D = run(qml_params)
        print("edge count", np.count_nonzero(D))
        print("nodes with connection", np.count_nonzero(np.count_nonzero(D, axis=0)))
        fname = "{}.out".format(sys.argv[1])
        # Convert NumPy matrix to SciPy sparse matrix (CSR format)
        sparse_matrix = csr_matrix(D)
        # Iterate over the edges in the sparse matrix
        with open(fname, "w") as file:
            for i in range(sparse_matrix.shape[0]):
                for j in sparse_matrix.indices[sparse_matrix.indptr[i]:sparse_matrix.indptr[i + 1]]:
                    # i and j represent the indices of the connected nodes
                    # print(f"Edge: ({i}, {j}) - Value: {sparse_matrix[i, j]}")
                    print(i, j, sparse_matrix[i, j], file=file)
        # with open(fname, "w") as file:
        #     for i in range(D.shape[0]):
        #         for j in range(D.shape[1]):
        #             # if (not np.isinf(data[i,j])):
        #             if (not D[i,j] == 0.0):
        #                 print(i, j, D[i,j], file=file)

        # save geodesic distance matrix to file
        # fname = "{}.out".format(sys.argv[1])
        # np.savetxt(fname, D, fmt='%.10f', delimiter=',')

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

        if qml_params['SHOW_EMBEDDING']==3: ######## TODO change back to print
            print("Computing 3D embedding using geodesic distance matrix ...")
            sources, targets = D.nonzero()
            edgelist = zip(sources.tolist(), targets.tolist())
            g = ig.Graph(edgelist)
            # g = ig.Graph.Weighted_Adjacency(D)
            fig = plt.figure(figsize=(6,6))

            # lyout3d = g.layout_umap()
            lyout3d = g.layout_fruchterman_reingold_3d()
            # g.layout_
            ax = fig.add_subplot(111, projection='3d')
            ed = np.array(lyout3d.coords)
            # load color map
            if qml_params['colorfile']!=False:
                try:
                    # colors = np.genfromtxt(qml_params['colorfile'], delimiter=',')
                    colors = read_in_matrix(qml_params['colorfile'], qml_params['verbose'])
                    # colors = np.loadtxt(open(qml_params['colorfile'], "rb"), delimiter=",", dtype=str)
                    colors = np.array(colors)
                    colors = colors[row_indices]
                except:
                    print("Cannot open color file: " + qml_params['colorfile'] + "... Exiting.")
                    raise Exception("Cannot open color file")
                else:
            #         if qml_params['labelfile']!=False:
            #             try:
            #                 # colors = np.genfromtxt(qml_params['colorfile'], delimiter=',')
            #                 labels = np.loadtxt(open(qml_params['labelfile'], "rb"), delimiter=",", dtype=str)
            #             except:
            #                 print("Cannot open label file: " + qml_params['labelfile'] + "... Exiting.")
            #                 raise Exception("Cannot open label file")
            #             else:
            #                 # legend_handles = []
            #                 # legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color))
                            
            #                 # print("label size ", labels.size)
            #                 # print(labels)
            #                 # print("color size ", colors.size)
            #                 # print(colors)
            #                 # print("embed size", ed.size)
            #                 # print("n ", ed[:,0].size)

                            
            #                 unique_classes = np.unique(np.array(colors))
            #                 # print("unique_classes", unique_classes)
            #                 # Generate a colormap with a different color for each class
            #                 num_classes = len(unique_classes)
            #                 classSizes = np.zeros((num_classes,1))
            #                 # base_cmap = plt.get_cmap('tab20')
            #                 # # Create a new colormap with 99 colors by replicating the base_cmap
            #                 # num_colors = num_classes
            #                 # new_colors = np.concatenate([base_cmap(i * np.ones(5)) for i in range(5)])
            #                 # # Trim the colormap to have the desired number of colors
            #                 # cmap = ListedColormap(new_colors[:num_colors], name='custom_cmap', N=num_colors)
            #                 # print("num_classes", num_classes)
            #                 cmap = plt.get_cmap('hsv', num_classes) # viridis Spectral
            #                 # print("cmap ", cmap)
            #                 print("colors", colors)

            #                 # Get multiple qualitative colormaps
            #                 cmaps = ['tab20b', 'tab20c', 'Set1', 'Set3', 'Dark2', 'Accent']

            #                 # Combine colormaps to create a new colormap
            #                 num_colors = num_classes
            #                 new_colors = []
            #                 print("cmap size", len(cmaps))
            #                 for cmap in cmaps:
            #                     base_cmap = plt.get_cmap(cmap)
            #                     new_colors.extend(base_cmap(np.arange(base_cmap.N)))
            #                     # new_colors.extend(base_cmap(np.linspace(0, 1, num_colors // len(cmaps))))

            #                 # Trim the colormap to have the desired number of colors
            #                 cmap = ListedColormap(new_colors[:num_colors], name='custom_cmap', N=num_colors)


            #                 labelColorMapping = {}
            #                 for i, label in enumerate(labels):
            #                     if label not in labelColorMapping:
            #                         # print("label color, ", label, colors[i], i)
            #                         labelColorMapping[label] = colors[i]
            #                 i = 0
            #                 for label, color in labelColorMapping.items():
            #                     # print("label color", label, color)
            #                     # print("colors==color", colors==color)
            #                     # print("colorsingle", np.where(unique_classes == color))
            #                     tempColors = colors[(colors==color).reshape(-1)]
            #                     tempX = ed[(colors==color).reshape(-1),:]
            #                     tempLabels = labels[(labels==label)]
            #                     # print("tempX size", tempX.size)
            #                     # print("tempColors size", tempColors.size)
            #                     # print("tempLabels size", tempLabels.size)
            #                     # print("color, ", color.size, color, color[0])
            #                     colorSingle = cmap(np.where(unique_classes == color))
            #                     # print("colorSingle ", colorSingle, color)
            #                     colorPrint = np.full((tempX[:,0].shape[0],4), colorSingle)
            #                     if (tempX.size > 0):
            #                         ax.scatter(tempX[:,0], tempX[:,1], tempX[:,2], c=colorPrint, label=label)
            #                     # ax.scatter(tempX[:,0], tempX[:,1], tempX[:,2], label=label)
            #                     classSizes[i] = tempX.size
            #                     i += 1
            #                 plt.legend(ncol=num_classes/25, fontsize="4")

            #                 # legend_entries = {}
            #                 # # Create the scatter plot with unique labels and their respective colors
            #                 # for i, label in enumerate(labels):
            #                 #     if label not in legend_entries:
            #                 #         legend_entries[label] = ax.scatter(ed[i,0], ed[i,1], ed[i,2], c=colors[i], label=label, cmap=plt.cm.Spectral)
            #                 #     else:
            #                 #         ax.scatter(ed[i,0], ed[i,1], ed[i,2], c=colors[i], cmap=plt.cm.Spectral)
            #                 # # Create a custom legend based on the unique labels and colors
            #                 # handles = [legend_entries[label] for label in legend_entries]
            #                 # plt.legend(handles=handles)
                            
                            
            #                 # ax.scatter(ed[:,0], ed[:,1], ed[:,2], c=colors, cmap=plt.cm.Spectral, label=labels)
            #                 # plt.legend(loc='upper left')
            #         # ax.scatter(ed[:,0], ed[:,1], ed[:,2], c=colors, cmap=plt.cm.Spectral)
            # else:
                    # ax.scatter(ed[:,0], ed[:,1], ed[:,2], c=colors)
                    print(lyout3d)
                    print(ed.shape)
                    # ax.scatter(ed[:,0], ed[:,1], c=colors[1:1000])
                    # ax.scatter(ed[:,0], ed[:,1], ed[:,2], c=colors[0:testNum])
                    ax.scatter(ed[:,0], ed[:,1], ed[:,2], c=colors, cmap=plt.cm.Spectral)
                    ax.axis('off')
                    ax.set_title('3D embedding')

                    plt.show()

        # np.savetxt("{}.csv".format(sys.argv[1]), ed, delimiter=",")

        # score = silhouette_score(ed, colors)
        # print("silhouette score: ", score)
        # scores = np.zeros(unique_classes.shape)
        # i = 0
        # # colors = colors.reshape(-1)
        # # print("colors", colors, colors.shape)
        # # print("unique classes", unique_classes)
        # for i, clas in enumerate(unique_classes):
        #     # print("clas", clas)
        #     # tempColor = labelColorMapping[clas]
        #     tempColors = np.ones(colors.shape) * 2
        #     tempColors[colors == clas] = 1
        #     # tempColors = tempColors.reshape((tempColors.size, 1))
        #     # print("tempcolors", tempColors, tempColors.shape)
        #     scores[i] = silhouette_score(ed, tempColors)
        # # print("class sizes", classSizes.flatten())
        # indices = np.argsort(classSizes.flatten())
        # # print("indices", indices)
        # print("class sizes", classSizes[indices].flatten())
        # scores = scores[indices]
        # np.set_printoptions(precision=4)
        # print("single class silhouette scores", scores)


        
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

        # distanceList = []
        # ids, distances, xScale = runDistanceCompare(qml_params, 1000, 0.0)
        # print("first distance compare")
        # epsilonSave = qml_params['logepsilon']
        # for i in range(20,101,10):
        #     for j in range(-10,10,1):
        #         print("i,j", i, j)
        #         adjust = float(j) / 10.0
        #         # qml_params['logepsilon'] = j + epsilonSave
        #         ids, distances, xScale = runDistanceCompare(qml_params, i, j)
        #         distanceList.append(distances)

        # count = 0
        # sumDiff = []
        # for i in distanceList:
        #     print(count)
        #     count += 1
        #     print(i[:,0] - i[:,1])
        #     sumDiff.append(np.sum(i[:,0] - i[:,1]))
        # print("sum of differences", sumDiff)

        # firstDiff = []
        # for i in distanceList:
        #     firstDiff.append(i[0,0] - i[0,1])
        # print("first point difference", firstDiff)




        # test like h test

        # logeps_v = np.arange(-10, 0, 1)
        # logh_v =  np.arange(-10, 0, 1)
        # logeps_s = np.arange(-10, 0, 1)
        # Neps = np.shape(logeps_v)[0]
        # Nhs = np.shape(logh_v)[0]
        # Ness = np.shape(logeps_s)[0]
        # distanceQML = np.zeros((Neps,Nhs,Ness))
        # distanceCir = np.zeros((Neps,Nhs,Ness))
        # distanceRMS = np.zeros((Neps,Nhs,Ness))
        # distanceCount = np.zeros((Neps,Nhs,Ness))

        # distanceList = []
        # ids, distances, xScale = runDistanceCompare(qml_params, 100, 0.0, 0.0, 0.0)
        # print("first distance compare")
        # epsilonSave = qml_params['logepsilon']
        # for i in range(80,101,100):
        #     for idj, j in enumerate(logeps_v):
        #         for idk, k in enumerate(logh_v):
        #             for idl, l in enumerate(logeps_s):
        #                 print("i,j", i, j)
        #                 adjust = float(j) / 10.0
        #                 # qml_params['logepsilon'] = j + epsilonSave
        #                 ids, distances, xScale = runDistanceCompare(qml_params, i, j, k, l)
        #                 # distanceList.append(distances)
        #                 if distances.size == 0:
        #                     distanceQML[idj,idk,idl] = -np.pi
        #                     distanceCir[idj,idk,idl] = 0.0
        #                     distanceRMS[idj,idk,idl] = np.pi
        #                 else:
        #                     distanceQML[idj,idk,idl] = distances[0,0]
        #                     distanceCir[idj,idk,idl] = distances[0,1]
        #                     distanceRMS[idj,idk,idl] = np.sqrt( (1/distances[:,1].size) * np.sum(np.power(distances[:,0]-distances[:,1],2)) )
        #                     distanceCount[idj,idk,idl] = distances[:,1].size

        logeps_v = np.arange(-8, 2, 0.5)
        logh_v =  np.arange(-8, 2, 0.5)
        logeps_s = np.arange(-10, 0, 1)
        Nes = np.shape(logeps_v)[0]
        Nhs = np.shape(logh_v)[0]
        Ness = np.shape(logeps_s)[0]
        hTestDifs = np.zeros((Nes,Nhs,Ness))
        hTestSuSe = np.zeros((Nes,Nhs))
        hTestDifs_s = np.zeros((Nes,Nhs))
        # h test
        nLimit = 99
        for i in range(nLimit,101,100):
            for idl, l in enumerate(logeps_s):
                print("i,j", i, j)
                adjust = float(j) / 10.0
                # qml_params['logepsilon'] = j + epsilonSave
                # ids, distances, xScale = runDistanceCompare(qml_params, i, j, k, l)
                hTestDifs[:,:,idl], hTestDifs_s = perform_hamiltonian_test_nystrom(qml_params, i, l)


        # Subplots are organized in a Rows x Cols Grid
        # Tot and Cols are known
        Tot = Ness
        Cols = 4
        # Compute Rows required
        Rows = Tot // Cols 
        #     EDIT for correct number of rows:
        #     If one additional row is necessary -> add one:
        if Tot % Cols != 0:
            Rows += 1
        # Create a Position index
        Position = range(1,Tot + 1)






        # plotting of geodesic distance testing
        # # # plot first error
        # # loge, logh = np.meshgrid(logeps_v, logh_v, indexing='ij')
        # # fig, ax = plt.subplots()
        # # print("loge", loge)
        # # print("logh", logh)
        # # print("devs", distanceQML - distanceCir)
        # # im = ax.pcolormesh(loge, logh, distanceQML - distanceCir)
        # # # im = ax.pcolormesh(np.transpose(logh), np.transpose(loge), devs)
        # # fig.colorbar(im)

        # # ax.set_xlabel('log(eps)')
        # # ax.set_ylabel('log(h)')
        # # ax.set_title('First Point Found Error')

        # # plt.savefig('Nystrom_firstError_test.png')
        # # plt.show()

        # # plot rms
        # loge, logh = np.meshgrid(logeps_v, logh_v, indexing='ij')
        # # print("1d min", np.minimum(distanceRMS))
        # # print("2d min", np.minimum(np.minimum(distanceRMS)))
        # # print("3d min", np.minimum(np.minimum(np.minimum(distanceRMS))))
        # # colorMin = np.minimum(np.minimum(np.minimum(distanceRMS)))
        # colorMin = np.min(distanceRMS)
        # # Create main figure
        # fig = plt.figure(1)
        # for k in range(Tot):
        #     # add every single subplot to the figure with a for loop
        #     ax = fig.add_subplot(Rows,Cols,Position[k])

        #     # fig, ax = plt.subplots()
        #     # print("loge", loge)
        #     # print("logh", logh)
        #     # print("devs", distanceQML - distanceCir)

        #     im = ax.pcolormesh(loge, logh, distanceRMS[:,:,k], norm="log", vmin=colorMin, vmax=np.pi)
        #     # im = ax.pcolormesh(loge, logh, distanceRMS[:,:,k], vmin=colorMin, vmax=np.pi)
        #     # im = ax.pcolormesh(np.transpose(logh), np.transpose(loge), devs)
        #     fig.colorbar(im)

        #     ax.set_xlabel('log(eps)')
        #     ax.set_ylabel('log(h)')
        #     ax.set_title('RMSE small Epsilon: ' + str(logeps_s[k]))

        # plt.savefig('Nystrom_RMSE_test.png')
        # plt.show()

        # ids = np.unravel_index(np.argmin(distanceRMS, axis=None), distanceRMS.shape)
        # print("min RMS value", colorMin)
        # print("# of edges found", distanceCount[ids])












        # plot h test
        loge, logh = np.meshgrid(logeps_v, logh_v, indexing='ij')
        # print("1d min", np.minimum(distanceRMS))
        # print("2d min", np.minimum(np.minimum(distanceRMS)))
        # print("3d min", np.minimum(np.minimum(np.minimum(distanceRMS))))
        # colorMin = np.minimum(np.minimum(np.minimum(distanceRMS)))
        hTestDifs = np.nan_to_num(hTestDifs, nan=10.0)
        hTestDifs = np.abs(hTestDifs)
        print(hTestDifs)
        colorMin = np.min(hTestDifs)
        colorMax = np.max(hTestDifs)
        if (colorMax > 10):
            colorMax = 10
        # Create main figure
        fig = plt.figure(1)
        for k in range(Tot):
            # add every single subplot to the figure with a for loop
            ax = fig.add_subplot(Rows,Cols,Position[k])

            # fig, ax = plt.subplots()
            # print("loge", loge)
            # print("logh", logh)
            # print("devs", distanceQML - distanceCir)

            im = ax.pcolormesh(loge, logh, hTestDifs[:,:,k], norm="log", vmin=colorMin, vmax=colorMax)
            # im = ax.pcolormesh(loge, logh, hTestDifs[:,:,k], vmin=colorMin, vmax=colorMax)
            # im = ax.pcolormesh(np.transpose(logh), np.transpose(loge), devs)
            fig.colorbar(im)

            ax.set_xlabel('log(eps)')
            ax.set_ylabel('log(h)')
            ax.set_title('Small Epsilon: ' + str(logeps_s[k]))

        ax = fig.add_subplot(Rows,Cols,Position[k]+1)
        im = ax.pcolormesh(loge, logh, hTestDifs_s, norm="log", vmin=colorMin, vmax=colorMax)
        fig.colorbar(im)
        ax.set_xlabel('log(eps)')
        ax.set_ylabel('log(h)')
        ax.set_title('H Test Non-Extended: ' + str(nLimit) + " points")

        plt.savefig('H_test_nystrom.png')
        plt.show()

        ids = np.unravel_index(np.argmin(hTestDifs, axis=None), hTestDifs.shape)
        print("min h test value", colorMin)