"""
线性判别分析的代码实现
"""
import numpy as np


class LDA(object):
    def __init__(self, n_components=None):
        """
        :param n_components: 主成分数量，原则上只需要（类别数-1）即可
        """
        self.n_components = n_components
        self.trans_matrix = None  # 保存前n_components个特征向量

    def fit(self, X, y):

        x_mean = np.mean(X, axis=0)
        k = np.max(y) + 1  # 类别
        dims = len(x_mean)  # 数据维度
        if self.n_components is None:
            self.n_components = dims
        S_b = np.zeros(shape=(dims, dims))
        S_w = np.zeros(shape=(dims, dims))
        for j in range(0, k):
            idx = np.argwhere(y == j).reshape(-1)
            N_j = len(idx)
            X_j = X[idx]
            x_mean_j = np.mean(X_j, axis=0)
            S_b += N_j * ((x_mean - x_mean_j).reshape(-1, 1).dot((x_mean - x_mean_j).reshape(1, -1)))
            S_w += (X_j - x_mean_j).T.dot(X_j - x_mean_j)
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_w).dot(S_b))
        sorted_indice = np.argsort(-1 * eig_vals)  # 默认升序排序
        eig_vecs[:] = eig_vecs[:, sorted_indice]
        self.trans_matrix = eig_vecs[:, 0:self.n_components]

    def transform(self, X):
        return X.dot(self.trans_matrix)
