"""
kmeans聚类实现
"""

import numpy as np


class KMeans(object):
    def __init__(self, k=3, epochs=100, tol=1e-3, dist_method=None):
        """
        :param k: 聚类簇数量
        :param epochs: 最大迭代次数
        :param tol: 终止条件
        :param dist_method:距离函数，默认欧氏距离
        """
        self.k = k
        self.epochs = epochs
        self.tol = tol
        self.dist_method = dist_method
        if self.dist_method is None:
            self.dist_method = lambda x, y: np.sqrt(np.sum(np.power(x - y, 2)))
        self.cluster_centers_ = {}  # 记录簇中心坐标

    def fit(self, X):
        m = X.shape[0]
        # 初始化
        for idx, data_idx in enumerate(np.random.choice(list(range(m)), self.k, replace=False).tolist()):
            self.cluster_centers_[idx] = X[data_idx]
        # 迭代
        for _ in range(self.epochs):
            C = {}
            for idx in range(self.k):
                C[idx] = []
            for j in range(m):
                best_k = None
                min_dist = np.infty
                for idx in range(self.k):
                    dist = self.dist_method(self.cluster_centers_[idx], X[j])
                    if dist < min_dist:
                        min_dist = dist
                        best_k = idx
                C[best_k].append(j)
            # 更新
            eps = 0
            for idx in range(self.k):
                vec_k = np.mean(X[C[idx]], axis=0)
                eps += self.dist_method(vec_k, self.cluster_centers_[idx])
                self.cluster_centers_[idx] = vec_k
            # 判断终止条件
            if eps < self.tol:
                break

    def predict(self, X):
        m = X.shape[0]
        rst = []
        for i in range(m):
            vec = X[i]
            best_k = None
            min_dist = np.infty
            for idx in range(self.k):
                dist = self.dist_method(self.cluster_centers_[idx], vec)
                if dist < min_dist:
                    min_dist = dist
                    best_k = idx
            rst.append(best_k)
        return np.asarray(rst)