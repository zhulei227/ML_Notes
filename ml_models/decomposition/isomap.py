"""
等度量映射实现
"""
import numpy as np


class Isomap(object):
    def __init__(self, n_components=None, epsilon=1e-3):
        self.n_components = n_components
        self.epsilon = epsilon

    @staticmethod
    def floyd(dist_matrix):
        vex_num = len(dist_matrix)
        for k in range(vex_num):
            for i in range(vex_num):
                for j in range(vex_num):
                    if dist_matrix[i][k] == np.inf or dist_matrix[k][j] == np.inf:
                        temp = np.inf
                    else:
                        temp = dist_matrix[i][k] + dist_matrix[k][j]
                    if dist_matrix[i][j] > temp:
                        dist_matrix[i][j] = temp
        return dist_matrix

    def fit_transform(self, data=None, D=None):

        if D is None:
            m = data.shape[0]
            D = np.zeros(shape=(m, m))
            # 初始化D
            for i in range(0, m):
                for j in range(i, m):
                    D[i, j] = np.sqrt(np.sum(np.power(data[i] - data[j], 2)))
                    D[j, i] = D[i, j]
        else:
            m = D.shape[0]

        # floyd更新
        D = np.where(D < self.epsilon, D, np.inf)
        D = self.floyd(D)
        D_i = np.sum(np.power(D, 2), axis=0) / m
        D_j = np.sum(np.power(D, 2), axis=1) / m
        D_2 = np.sum(np.power(D, 2)) / (m * m)

        # 计算B
        B = np.zeros(shape=(m, m))
        for i in range(0, m):
            for j in range(i, m):
                B[i, j] = -0.5 * (D[i, j] * D[i, j] - D_i[i] - D_j[j] + D_2)
                B[j, i] = B[i, j]
        # 求Z
        eig_vals, eig_vecs = np.linalg.eig(B)
        sorted_indice = np.argsort(-1 * eig_vals)
        eig_vals = eig_vals[sorted_indice]
        eig_vecs[:] = eig_vecs[:, sorted_indice]
        # 简化：取前n_components维
        Lambda = np.diag(eig_vals.real[:self.n_components])
        vecs = eig_vecs.real[:, :self.n_components]
        return vecs.dot(np.sqrt(Lambda))