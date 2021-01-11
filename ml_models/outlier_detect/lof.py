"""
Local Outlier Factor实现
"""
import numpy as np


class LOF(object):
    def __init__(self, n_neighbors=5):
        """
        :param n_neighbors: 考虑最近的样本量
        """
        self.n_neighbors = n_neighbors

    def fit_transform(self, X):
        # 构建距离矩阵，这里默认使用欧式距离
        m = X.shape[0]
        D = np.zeros(shape=(m, m))
        for i in range(0, m):
            for j in range(i, m):
                D[i, j] = np.sqrt(np.sum(np.power(X[i] - X[j], 2)))
                D[j, i] = D[i, j]
        # 对每个样本，求最近的n_neighbors距离（非0）
        d_k = []
        for i in range(0, m):
            d = D[i]
            d = d[d > 0]
            d.sort()
            d_k.append(d[min(self.n_neighbors, len(d)) - 1])
        # 计算局部可达密度
        lrd = []
        for i in range(0, m):
            d = D[i]
            indices = d.argsort()
            k = 0
            neighbor_distances = []
            for idx in indices:
                if k == self.n_neighbors:
                    break
                if D[i, idx] > 0:
                    neighbor_distances.append(max(D[i, idx], d_k[idx]))
                    k += 1
            lrd.append(len(neighbor_distances) / np.sum(neighbor_distances))
        # 计算lof
        lof = []
        for i in range(0, m):
            d = D[i]
            indices = d.argsort()
            k = 0
            neighbor_lrd = []
            for idx in indices:
                if k == self.n_neighbors:
                    break
                if D[i, idx] > 0:
                    neighbor_lrd.append(lrd[idx])
                    k += 1
            lof.append(np.sum(neighbor_lrd) / (len(neighbor_lrd) * lrd[i]))
        return np.asarray(lof)
