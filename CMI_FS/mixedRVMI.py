# coding=utf-8
from scipy.special import digamma
from scipy.spatial import cKDTree
import numpy as np

def MIEstimate(X,Y,k=5,estimate='digamma'):
    'MI Estimator based on Mixed Random Variable Mutual Information Estimator - Gao et al.'
    nSamples = len(X)
    if X.ndim == 1:
        X = X.reshape(-1,1)
    if Y.ndim == 1:
        Y = Y.reshape(-1,1)
    dataset = np.concatenate((X,Y), axis=1) # concatenate Y to X as a column

    # kdtree per trovare più rapidamente i k nearest neighbors
    tree_xy = cKDTree(dataset) 
    tree_x = cKDTree(X)
    tree_y = cKDTree(Y)
    
    # rho
    Knn_dists = [tree_xy.query(sample, k+1, p=float('inf'))[0][k] for sample in dataset]
    
    res = 0
    
    for i in range(nSamples):
        k_hat, n_xi, n_yi = k, k, k
        #print(Knn_dists[i])
        if Knn_dists[i] <= 1e-15:
            # punti a distanza inferiore uguale a (quasi) 0 
            k_hat = len(tree_xy.query_ball_point(dataset[i], 1e-15, p=float('inf')))
            n_xi = len(tree_x.query_ball_point(X[i], 1e-15, p=float('inf')))
            n_yi = len(tree_y.query_ball_point(Y[i], 1e-15, p=float('inf')))
        else:
            k_hat = k 
            # punti a distanza inferiore uguale a rho
            n_xi = len(tree_x.query_ball_point(X[i], Knn_dists[i]-1e-15, p=float('inf')))
            n_yi = len(tree_y.query_ball_point(Y[i], Knn_dists[i]-1e-15, p=float('inf')))
        #print(k_hat)
        #print(n_xi)
        #print(n_yi)
        if estimate=='digamma':
            res+=(digamma(k_hat) + np.log(nSamples) - digamma(n_xi) - digamma(n_yi))/nSamples
        else:
            res+=(digamma(k_hat) + np.log(nSamples) - np.log(n_xi+1) - np.log(n_yi+1))/nSamples
    return res

def CMIEstimate(X,Y,Z,k=5,estimate='digamma'):
    """
    I(X;Y|Z) = I(X,Z; Y) - I(Z; Y)
    """
    if X.ndim == 1:
        X = X.reshape(-1,1)
    if Y.ndim == 1:
        Y = Y.reshape(-1,1)
    if Z.ndim == 1:
        Z = Z.reshape(-1,1) 
    
    XZ = np.hstack((X,Z)) # concateno X,Z in colonne

    return (MIEstimate(XZ,Y,k,estimate)-MIEstimate(Z,Y,k,estimate))


def estimateAllMI(X,Y,k):
    'returns a list of MI scores, each one is the MI between one feature and the target'
    scores = np.zeros(X.shape[1])

    for col in range(X.shape[1]):
        scores[col] = MIEstimate(X[:, col], Y, k)
    
    return list(zip(range(len(scores)), scores))