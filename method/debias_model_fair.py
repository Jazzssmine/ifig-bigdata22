import utils

import numpy as np
import networkx as nx
import sklearn.preprocessing as skpp

from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh


class DebiasModel:
    """
    debiasing the mining model
    """
    def __init__(self):
        return

    def fit(self):
        return

    @staticmethod
    def spectral_clustering(adj, sim, alpha, lmbda, ncluster=10, niteration=10, v0=None):
        """
        individually fair spectral clustering
        :param adj: adjacency matrix
        :param sim: similarity matrix
        :param alpha: regularization parameter
        :param lambda: intra view parameter 
        :param ncluster: number of clusters
        :param v0: starting vector for eigen-decomposition
        :return: soft cluster membership matrix of fair spectral clustering
        """
        matrices = {0: {}}
        matrices[0]['L'] = {}
        matrices[0]['U'] = {}
        views = np.array([v0, v0])

        L = {}
        U = {}
        for i in range(len(adj)):
            L[i] = laplacian(adj[i]) + alpha[i] * laplacian(sim[i])
            L[i] *= -1
            _, U[i] = eigsh(L[i], which='LM', k=ncluster, sigma=1.0, v0=v0)

            matrices[0]['U'][i] = U[i]


        for i in range(len(adj)):
            matrices[i+1] = {}
            matrices[i+1]['L'] = {}
            matrices[i+1]['U'] = {}
            matrices[i+1]['KU'] = {}


            for j in range(len(adj)):
                KU = sum(np.matmul(eigenvectors, eigenvectors.T) for k, eigenvectors in enumerate(U) if k != j)
                lap = L[j] + lmbda[j] * KU

                _, U[j] = eigsh(L[j], which='LM', k=ncluster, sigma=1.0, v0=v0)
                L[j] = lap

                matrices[i+1]['L'][j] = L[j]
                matrices[i+1]['U'][j] = U[j]
                matrices[i+1]['KU'][j] = KU

            print('Iteration: ', i+1)

            # Measure of disagreement (to be minimized)
            if not len(adj) == 1:
                print("Disagreement: ",
                    - np.linalg.multi_dot(
                        [np.matmul(eigenvectors, eigenvectors.T) for m, eigenvectors in U.items()]).trace())

        return matrices

    
