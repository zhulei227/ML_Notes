"""
非负矩阵分解,NMF
"""

import numpy as np


class NMF(object):
    def __init__(self, n_components=None, epochs=100, tol=1e-3):
        """
        :param n_components: 降维数
        :param epochs:最大迭代次数
        :param tol:最大误差
        """
        self.n_components = n_components
        if self.n_components is None:
            self.n_components = 2
        self.epochs = epochs
        self.tol = tol

    def fit_transform(self, X):
        m, n = X.shape
        W = np.abs(np.random.random(size=(m, self.n_components)))
        H = np.abs(np.random.random(size=(self.n_components, n)))
        # update
        for _ in range(self.epochs):
            W_ratio = (X @ H.T) / (W @ H @ H.T)
            H_ratio = (W.T @ X) / (W.T @ W @ H)
            W_u = W * W_ratio
            H_u = H * H_ratio
            if np.mean(np.abs(W - W_u)) < self.tol:
                return W_u
            W = W_u
            H = H_u
        return W
