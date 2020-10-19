"""
主成分分析
"""
import numpy as np


class PCA(object):
    def __init__(self, n_components=None):
        """
        :param n_components: 主成分数量
        """
        self.n_components = n_components
        self.mean_info = None  # 保存均值信息
        self.trans_matrix = None  # 保存前n_components个主成分矩阵

    def fit(self, X):
        self.mean_info = np.mean(X, axis=0, keepdims=True)
        if self.n_components is None:
            self.n_components = X.shape[1]
        X_ = X - self.mean_info
        xTx = X_.T.dot(X_)
        eig_vals, eig_vecs = np.linalg.eig(xTx)
        sorted_indice = np.argsort(-1 * eig_vals)  # 默认升序排序
        eig_vecs[:] = eig_vecs[:, sorted_indice]
        self.trans_matrix = eig_vecs[:, 0:self.n_components]

    def transform(self, X):
        X_ = X - self.mean_info
        return X_.dot(self.trans_matrix)