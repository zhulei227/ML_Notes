import numpy as np


class KNN(object):
    def __init__(self, n_neighbors=3):
        """
        :param n_neighbors: 最近的样本量
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
        # 对每个样本，求最近的n_neighbors个非0距离之和
        rst = []
        for i in range(0, m):
            d = D[i]
            d = d[d > 0]
            d.sort()
            rst.append(np.sum(d[:min(self.n_neighbors, len(d))]))
        return np.asarray(rst)
