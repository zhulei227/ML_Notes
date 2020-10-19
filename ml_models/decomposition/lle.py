"""
局部线性嵌入LLE
"""
import numpy as np


class LLE(object):
    def __init__(self, k=5, n_components=3):
        """
        :param k: 最近邻数
        :param n_components: 降维后的维度
        """
        self.k = k
        self.n_components = n_components

    def fit_transform(self, data):
        # 构建距离矩阵
        m = data.shape[0]
        D = np.zeros(shape=(m, m))
        for i in range(0, m):
            for j in range(i, m):
                D[i, j] = np.sqrt(np.sum(np.power(data[i] - data[j], 2)))
                D[j, i] = D[i, j]

        # 保留最近的k个坐标
        idx = []
        for i in range(0, m):
            idx.append(np.argsort(D[i])[1:self.k + 1].tolist())
        # 构建权重矩阵W
        C = np.zeros(shape=(m, self.k, self.k))  # 辅助计算W
        W = np.zeros(shape=(m, m))
        for i in range(0, m):
            nn_idx = idx[i]  # Q_i
            for cj1 in range(0, self.k):
                for cj2 in range(cj1, self.k):
                    j1 = nn_idx[cj1]
                    j2 = nn_idx[cj2]
                    c = np.dot(data[i] - data[j1], data[i] - data[j2])
                    c = 1.0 / c
                    C[i, cj1, cj2] = c
                    C[i, cj2, cj1] = c
        C = np.sum(C, axis=1)
        C = C / np.sum(C, axis=1, keepdims=True)
        for i in range(0, m):
            nn_idx = idx[i]
            for cj in range(0, self.k):
                W[i, nn_idx[cj]] = C[i, cj]
        # 对M特征分解
        M = (np.eye(m) - W).T.dot(np.eye(m) - W)
        eig_vals, eig_vecs = np.linalg.eig(M)
        sorted_indice = np.argsort(eig_vals.real)
        eig_vals = eig_vals[sorted_indice]
        eig_vecs[:] = eig_vecs[:, sorted_indice]
        # 保留前n_compnent维
        return eig_vecs.real[:, :self.n_components]
