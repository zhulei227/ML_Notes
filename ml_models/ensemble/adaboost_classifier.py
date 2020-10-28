"""
AdaBoost分类器的实现
"""

from ml_models.tree import CARTClassifier
import copy
import numpy as np


class AdaBoostClassifier(object):
    def __init__(self, base_estimator=None, n_estimators=10, learning_rate=1.0):
        """
        :param base_estimator: 基分类器，允许异质；异质的情况下使用列表传入比如[estimator1,estimator2,...,estimator10],这时n_estimators会失效；
                                同质的情况，单个estimator会被copy成n_estimators份
        :param n_estimators: 基分类器迭代数量
        :param learning_rate: 学习率，降低后续基分类器的权重，避免过拟合
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        if self.base_estimator is None:
            # 默认使用决策树桩
            self.base_estimator = CARTClassifier(max_depth=2)
        # 同质分类器
        if type(base_estimator) != list:
            estimator = self.base_estimator
            self.base_estimator = [copy.deepcopy(estimator) for _ in range(0, self.n_estimators)]
        # 异质分类器
        else:
            self.n_estimators = len(self.base_estimator)

        # 记录estimator权重
        self.estimator_weights = []

    def fit(self, x, y):
        n_sample = x.shape[0]
        sample_weights = np.asarray([1.0] * n_sample)
        for index in range(0, self.n_estimators):
            self.base_estimator[index].fit(x, y, sample_weight=sample_weights)

            indicates = (self.base_estimator[index].predict(x) == y).astype(int)
            # 计算误分率
            error_rate = np.sum([sample_weights[j] * (1.0 - indicates[j]) for j in range(0, n_sample)]) / n_sample

            # 计算权重系数
            alpha_rate = 1.0 / 2.0 * np.log((1 - error_rate) / (error_rate + 1e-7))
            alpha_rate = min(10.0, alpha_rate)
            self.estimator_weights.append(alpha_rate)

            # 更新样本权重
            for j in range(0, n_sample):
                sample_weights[j] = sample_weights[j] * np.exp(-1.0 * alpha_rate * np.power(-1.0, 1 - indicates[j]))
            sample_weights = sample_weights / np.sum(sample_weights) * n_sample
        # 更新estimator权重
        for i in range(0, self.n_estimators):
            self.estimator_weights[i] *= np.power(self.learning_rate, i)

    def predict_proba(self, x):
        # TODO:并行优化
        result = np.sum(
            [self.base_estimator[j].predict_proba(x) * self.estimator_weights[j] for j in
             range(0, self.n_estimators)],
            axis=0)
        return result / result.sum(axis=1, keepdims=True)

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)
