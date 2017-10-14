import numpy as np
from sklearn.neighbors import NearestNeighbors


def fit_transformation(source, target):
    assert source.shape == target.shape
    center_source = np.mean(source, axis=0)
    center_target = np.mean(target, axis=0)
    m = source.shape[1]
    source_zeromean = source - center_source
    target_zeromean = target - center_target
    W = np.dot(source_zeromean.T, target_zeromean)
    U, S, Vt = np.linalg.svd(W)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = center_target.T - np.dot(R, center_source.T)

    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t
    return T, R, t


def nearest_neightbor(source, target):
    assert source.shape == target.shape
    neighbor_finder = NearestNeighbors(n_neighbors=1)
    neighbor_finder.fit(target)
    distances, indices = neighbor_finder.kneighbors(source, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(source, target, max_iter = 100, tol=0.001):
    pass
