"""
bagging回归实现
"""
from ml_models.tree import CARTRegressor
import copy
import numpy as np


class BaggingRegressor(object):
    def __init__(self, base_estimator=None, n_estimators=10):
        """
        :param base_estimator: 基学习器，允许异质；异质的情况下使用列表传入比如[estimator1,estimator2,...,estimator10],这时n_estimators会失效；
                                同质的情况，单个estimator会被copy成n_estimators份
        :param n_estimators: 基学习器迭代数量
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        if self.base_estimator is None:
            # 默认使用决策树
            self.base_estimator = CARTRegressor()
        # 同质
        if type(base_estimator) != list:
            estimator = self.base_estimator
            self.base_estimator = [copy.deepcopy(estimator) for _ in range(0, self.n_estimators)]
        # 异质
        else:
            self.n_estimators = len(self.base_estimator)

    def fit(self, x, y):
        # TODO:并行优化
        n_sample = x.shape[0]
        for estimator in self.base_estimator:
            # 重采样训练集
            indices = np.random.choice(n_sample, n_sample, replace=True)
            x_bootstrap = x[indices]
            y_bootstrap = y[indices]
            estimator.fit(x_bootstrap, y_bootstrap)

    def predict(self, x):
        # TODO:并行优化
        preds = []
        for estimator in self.base_estimator:
            preds.append(estimator.predict(x))

        return np.mean(preds, axis=0)
