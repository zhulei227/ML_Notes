"""
谱聚类实现
"""

import numpy as np
from ml_models.cluster import KMeans


class Spectral(object):
    def __init__(self, n_clusters=None, n_components=None, gamma=None):
        """
        :param n_clusters: 聚类数量
        :param n_components: 降维数量
        :param gamma: rbf函数的超参数
        """
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.gamma = gamma
        if self.n_components is None:
            self.n_components = 10
        if self.gamma is None:
            self.gamma = 1
        if self.n_clusters is None:
            self.n_clusters = 3

    def fit_transform(self, X):
        rows, cols = X.shape
        # 1.构建拉普拉斯矩阵
        W = np.zeros(shape=(rows, rows))
        for i in range(0, rows):
            for j in range(i, rows):
                w = np.exp(-1 * np.sum(np.power(X[i] - X[j], 2)) / (2 * self.gamma * self.gamma))
                W[i, j] = w
                W[j, i] = w
        D = np.diag(np.sum(W, axis=0))
        L = D - W
        # 2.对拉普拉斯矩阵特征分解
        eig_vals, eig_vecs = np.linalg.eig(L)
        sorted_indice = np.argsort(eig_vals)  # 默认升序排序
        eig_vecs[:] = eig_vecs[:, sorted_indice]
        return eig_vecs[:, 0:self.n_components].real

    def fit_predict(self, X):
        # 3.对特征矩阵进行聚类
        transform_matrix = self.fit_transform(X)
        transform_matrix = transform_matrix / np.sqrt(np.sum(np.power(transform_matrix, 2), axis=1, keepdims=True))
        kmeans = KMeans(k=self.n_clusters)
        kmeans.fit(transform_matrix)
        return kmeans.predict(transform_matrix)