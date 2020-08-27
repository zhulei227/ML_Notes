"""
层次聚类：AGNES的实现
"""

import numpy as np


# 定义默认的距离函数
def euclidean_average_dist(Gi, Gj):
    return np.sum(np.power(np.mean(Gi, axis=0) - np.mean(Gj, axis=0), 2))


class AGNES(object):
    def __init__(self, k=3, dist_method=None):
        """
        :param k: 聚类数量
        :param dist_method: 距离函数定义
        """
        self.k = k
        self.dist_method = dist_method
        if self.dist_method is None:
            self.dist_method = euclidean_average_dist
        self.G = None
        self.cluster_center = {}  # 记录聚类中心点

    def fit(self, X):
        m, _ = X.shape
        # 初始化簇
        G = {}
        for row in range(m):
            G[row] = X[[row]]
        # 计算簇间距离
        M = np.zeros(shape=(m, m))
        for i in range(0, m):
            for j in range(0, m):
                M[i, j] = self.dist_method(G[i], G[j])
                M[j, i] = M[i, j]
        q = m
        while q > self.k:
            # 寻找最近的簇
            min_dist = np.infty
            i_ = None
            j_ = None
            for i in range(0, q - 1):
                for j in range(i + 1, q):
                    if M[i, j] < min_dist:
                        i_ = i
                        j_ = j
                        min_dist = M[i, j]
            # 合并
            G[i_] = np.concatenate([G[i_], G[j_]])
            # 重编号
            for j in range(j_ + 1, q):
                G[j - 1] = G[j]
            # 删除G[q]
            del G[q-1]
            # 删除
            M = np.delete(M, j_, axis=0)
            M = np.delete(M, j_, axis=1)
            # 更新距离
            for j in range(q - 1):
                M[i_, j] = self.dist_method(G[i_], G[j])
                M[j, i_] = M[i_, j]
            # 更新q
            q = q - 1
        # self.G = G
        for idx in G:
            self.cluster_center[idx] = np.mean(G[idx], axis=0)

    def predict(self, X):
        rst = []
        rows, _ = X.shape
        for row in range(rows):
            # vec = X[[row]]
            vec = X[row]
            min_dist = np.infty
            bst_label = None
            for idx in self.cluster_center:
                # dist = self.dist_method(vec, self.G[idx]) < min_dist
                dist = np.sum(np.power(vec - self.cluster_center[idx], 2))
                if dist < min_dist:
                    bst_label = idx
                    min_dist = dist
            rst.append(bst_label)
        return np.asarray(rst)