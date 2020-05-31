"""
xgboost回归树的实现
"""
from ml_models.ensemble import XGBoostBaseTree
from ml_models import utils
import copy
import numpy as np


class XGBoostRegressor(object):
    def __init__(self, base_estimator=None, n_estimators=10, learning_rate=1.0, loss='squarederror', p=2.5):
        """
        :param base_estimator: 基学习器
        :param n_estimators: 基学习器迭代数量
        :param learning_rate: 学习率，降低后续基学习器的权重，避免过拟合
        :param loss:损失函数，支持squarederror、logistic、poisson,gamma,tweedie
        :param p:对tweedie回归生效
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        if self.base_estimator is None:
            # 默认使用决策树桩
            self.base_estimator = XGBoostBaseTree()
        # 同质分类器
        if type(base_estimator) != list:
            estimator = self.base_estimator
            self.base_estimator = [copy.deepcopy(estimator) for _ in range(0, self.n_estimators)]
        # 异质分类器
        else:
            self.n_estimators = len(self.base_estimator)
        self.loss = loss
        self.p = p

    def _get_gradient_hess(self, y, y_pred):
        """
        获取一阶、二阶导数信息
        :param y:真实值
        :param y_pred:预测值
        :return:
        """
        if self.loss == 'squarederror':
            return y_pred - y, np.ones_like(y)
        elif self.loss == 'logistic':
            return utils.sigmoid(y_pred) - utils.sigmoid(y), utils.sigmoid(y_pred) * (1 - utils.sigmoid(y_pred))
        elif self.loss == 'poisson':
            return np.exp(y_pred) - y, np.exp(y_pred)
        elif self.loss == 'gamma':
            return 1.0 - y * np.exp(-1.0 * y_pred), y * np.exp(-1.0 * y_pred)
        elif self.loss == 'tweedie':
            if self.p == 1:
                return np.exp(y_pred) - y, np.exp(y_pred)
            elif self.p == 2:
                return 1.0 - y * np.exp(-1.0 * y_pred), y * np.exp(-1.0 * y_pred)
            else:
                return np.exp(y_pred * (2.0 - self.p)) - y * np.exp(y_pred * (1.0 - self.p)), (2.0 - self.p) * np.exp(
                    y_pred * (2.0 - self.p)) - (1.0 - self.p) * y * np.exp(y_pred * (1.0 - self.p))

    def fit(self, x, y):
        y_pred = np.zeros_like(y)
        g, h = self._get_gradient_hess(y, y_pred)
        for index in range(0, self.n_estimators):
            self.base_estimator[index].fit(x, g, h)
            y_pred += self.base_estimator[index].predict(x) * self.learning_rate
            g, h = self._get_gradient_hess(y, y_pred)

    def predict(self, x):
        rst_np = np.sum(
            [self.base_estimator[0].predict(x)] +
            [self.learning_rate * self.base_estimator[i].predict(x) for i in
             range(1, self.n_estimators - 1)] +
            [self.base_estimator[self.n_estimators - 1].predict(x)]
            , axis=0)
        if self.loss in ["poisson", "gamma", "tweedie"]:
            return np.exp(rst_np)
        else:
            return rst_np
