"""
利用GMMCluster实现分类
"""
from ml_models.em import GMMCluster
import numpy as np


class GMMClassifier(object):
    def __init__(self, cluster_each_class=1, n_iter=100, tol=1e-3, shr_cov=False):
        """
        :param cluster_each_class:每个类需要几个高斯模型去拟合，默认1个
        :param n_iter:迭代次数
        :param tol: -log likehold增益<tol时，停止训练
        :param shr_cov:是否共享协方差矩阵
        """
        self.cluster_each_class = cluster_each_class
        self.n_iter = n_iter
        self.tol = tol
        self.shr_cov = shr_cov
        self.models = []

    def fit(self, X, y):
        for y_index in range(y.max() + 1):
            new_X = X[y == y_index]
            cluster = GMMCluster(n_components=self.cluster_each_class, tol=self.tol, n_iter=self.n_iter)
            cluster.fit(new_X)
            self.models.append(cluster)
        if self.shr_cov:
            # 获取所有的协方差矩阵
            sigmas = []
            for model in self.models:
                params = model.params
                for param in params:
                    sigmas.append(param[2])
            # 求平均
            ave_sigma = np.mean(sigmas, axis=0)
            # 更新
            for model in self.models:
                params = model.params
                for param in params:
                    param[2] = ave_sigma

    def predict_proba(self, X):
        W = np.asarray([model.predict_sample_generate_proba(X) for model in self.models]).T
        W = W / np.sum(W, axis=1, keepdims=True)
        return W

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
