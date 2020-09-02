"""
原型聚类LVQ的实现
"""

import numpy as np
import copy


class LVQ(object):
    def __init__(self, class_label=None, epochs=100, eta=1e-3, tol=1e-3, dist_method=None):
        """
        :param class_label: 原型向量类别标记
        :param epochs: 最大迭代次数
        :param eta:学习率
        :param tol: 终止条件
        :param dist_method:距离函数，默认欧氏距离
        """
        self.class_label = class_label
        self.epochs = epochs
        self.eta = eta
        self.tol = tol
        self.dist_method = dist_method
        if self.dist_method is None:
            self.dist_method = lambda x, y: np.sqrt(np.sum(np.power(x - y, 2)))
        self.cluster_centers_ = {}  # 记录簇中心坐标

    def fit(self, X, y):
        m = X.shape[0]
        # 随机初始化一组原型向量
        for idx, random_idx in enumerate(np.random.choice(list(range(m)), len(self.class_label), replace=False)):
            self.cluster_centers_[idx] = X[random_idx]
        # 更新
        for _ in range(self.epochs):
            eps = 0
            cluster_centers_old = copy.deepcopy(self.cluster_centers_)
            idxs = list(range(m))
            np.random.shuffle(idxs)
            # 随机选择样本点
            for idx in idxs:
                vec = X[idx]
                yi = y[idx]
                bst_distance = np.infty
                bst_cid = None
                for cid in range(len(self.class_label)):
                    center_vec = self.cluster_centers_[cid]
                    if self.dist_method(vec, center_vec) < bst_distance:
                        bst_distance = self.dist_method(vec, center_vec)
                        bst_cid = cid
                # 更新
                if yi == self.class_label[bst_cid]:
                    self.cluster_centers_[bst_cid] = (1-self.eta)*self.cluster_centers_[bst_cid] + self.eta * vec
                else:
                    self.cluster_centers_[bst_cid] = self.cluster_centers_[bst_cid] - self.eta * (vec - self.cluster_centers_[bst_cid])
            # 判断终止条件
            for key in self.cluster_centers_:
                eps += self.dist_method(cluster_centers_old[key], self.cluster_centers_[key])
            eps /= len(self.cluster_centers_)
            if eps < self.tol:
                break

    def predict(self, X):
        m = X.shape[0]
        rst = []
        for i in range(m):
            vec = X[i]
            best_k = None
            min_dist = np.infty
            for idx in range(len(self.cluster_centers_)):
                dist = self.dist_method(self.cluster_centers_[idx], vec)
                if dist < min_dist:
                    min_dist = dist
                    best_k = idx
            rst.append(best_k)
        return np.asarray(rst)
