"""
梯度提升回归树的实现（ps,其实也不一定非要决策树，所以代码实现上还是支持异质的情况）
"""
from ml_models.tree import CARTRegressor
import copy
import numpy as np


class GradientBoostingRegressor(object):
    def __init__(self, base_estimator=None, n_estimators=10, learning_rate=1.0, loss='ls', huber_threshold=1e-1,
                 quantile_threshold=0.5):
        """
        :param base_estimator: 基学习器，允许异质；异质的情况下使用列表传入比如[estimator1,estimator2,...,estimator10],这时n_estimators会失效；
                                同质的情况，单个estimator会被copy成n_estimators份
        :param n_estimators: 基学习器迭代数量
        :param learning_rate: 学习率，降低后续基学习器的权重，避免过拟合
        :param loss:表示损失函数ls表示平方误差,lae表示绝对误差,huber表示huber损失,quantile表示分位数损失
        :param huber_threshold:huber损失阈值，只有在loss=huber时生效
        :param quantile_threshold损失阈值，只有在loss=quantile时生效
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        if self.base_estimator is None:
            # 默认使用决策树桩
            self.base_estimator = CARTRegressor(max_depth=2)
        # 同质分类器
        if type(base_estimator) != list:
            estimator = self.base_estimator
            self.base_estimator = [copy.deepcopy(estimator) for _ in range(0, self.n_estimators)]
        # 异质分类器
        else:
            self.n_estimators = len(self.base_estimator)
        self.loss = loss
        self.huber_threshold = huber_threshold
        self.quantile_threshold = quantile_threshold

    def _get_gradient(self, y, y_pred):
        if self.loss == 'ls':
            return y - y_pred
        elif self.loss == 'lae':
            return (y - y_pred > 0).astype(int) * 2 - 1
        elif self.loss == 'huber':
            return np.where(np.abs(y - y_pred) > self.huber_threshold,
                            self.huber_threshold * ((y - y_pred > 0).astype(int) * 2 - 1), y - y_pred)
        elif self.loss == "quantile":
            return np.where(y - y_pred > 0, self.quantile_threshold, self.quantile_threshold-1)

    def fit(self, x, y):
        # 拟合第一个模型
        self.base_estimator[0].fit(x, y)
        y_pred = self.base_estimator[0].predict(x)
        new_y = self._get_gradient(y, y_pred)
        for index in range(1, self.n_estimators):
            self.base_estimator[index].fit(x, new_y)
            y_pred += self.base_estimator[index].predict(x) * self.learning_rate
            new_y = self._get_gradient(y, y_pred)

    def predict(self, x):
        return np.sum(
            [self.base_estimator[0].predict(x)] +
            [self.learning_rate * self.base_estimator[i].predict(x) for i in
             range(1, self.n_estimators - 1)] +
            [self.base_estimator[self.n_estimators - 1].predict(x)]
            , axis=0)
