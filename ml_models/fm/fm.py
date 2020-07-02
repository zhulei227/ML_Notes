"""
FM因子分解机的简单实现，只实现了损失函数为平方损失的回归任务，更多功能扩展请使用后续的FFM
"""
import numpy as np


class FM(object):
    def __init__(self, epochs=1, lr=1e-3, batch_size=2, hidden_dim=4, lamb=1e-3, normal=True):
        """

        :param epochs: 迭代轮数
        :param lr: 学习率
        :param batch_size:
        :param hidden_dim:隐变量维度
        :param lamb:l2正则化系数
        :param normal:是否归一化，默认用min-max归一化
        """
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.lamb = lamb
        # 初始化参数
        self.w = None  # w_0,w_i
        self.V = None  # v_{i,f}
        # 归一化
        self.normal = normal
        if normal:
            self.xmin = None
            self.xmax = None

    def _y(self, X):
        """
        实现y(x)的功能
        :param X:
        :return:
        """
        # 去掉第一列bias
        X_ = X[:, 1:]
        n_sample, n_feature = X_.shape
        pol = (X_.reshape((n_sample, n_feature, 1)) * X_.reshape((n_sample, 1, n_feature))) * (
            1 - np.eye(n_feature)) * (
                  self.V @ self.V.T)
        return X @ self.w.reshape(-1) + 0.5 * np.sum(pol, axis=(1, 2))

    def fit(self, X, y):
        if self.normal:
            self.xmin = X.min(axis=0)
            self.xmax = X.max(axis=0)
            X = (X - self.xmin) / self.xmax
        # 记录loss
        losses = []
        n_sample, n_feature = X.shape
        x_y = np.c_[np.ones(n_sample), X, y]
        # 初始化参数
        self.w = np.random.random((n_feature + 1, 1))
        self.V = np.random.random((n_feature, self.hidden_dim))
        # 更新参数
        count = 0
        for _ in range(self.epochs):
            np.random.shuffle(x_y)
            for index in range(x_y.shape[0] // self.batch_size):
                count += 1
                batch_x_y = x_y[self.batch_size * index:self.batch_size * (index + 1)]
                batch_x = batch_x_y[:, :-1]
                batch_y = batch_x_y[:, -1:]
                # 计算y(x)-t
                y_x_t = self._y(batch_x).reshape((-1, 1)) - batch_y
                # 更新w
                self.w = self.w - (self.lr * (np.sum(y_x_t * batch_x, axis=0) / self.batch_size).reshape(
                    (-1, 1)) + self.lamb * self.w)
                # 更新 V
                batch_x_ = batch_x[:, 1:]
                V_X = batch_x_ @ self.V
                X_V_X = batch_x_.reshape((batch_x_.shape[0], batch_x_.shape[1], 1)) * V_X.reshape(
                    (V_X.shape[0], 1, V_X.shape[1]))
                X_2 = batch_x_ * batch_x_
                V_X_2 = X_2.reshape((X_2.shape[0], X_2.shape[1], 1)) * self.V.reshape(
                    (1, self.V.shape[0], self.V.shape[1]))
                self.V = self.V - self.lr * (np.sum(y_x_t * (X_V_X - V_X_2),
                                                    axis=0) / self.batch_size + self.lamb * self.V)
                # 计算loss
                loss = np.sum(np.abs(y - self.predict(X))) / n_sample
                losses.append(loss)
        return losses

    def predict(self, X):
        """
        :param X:
        :return:
        """
        if self.normal:
            X = (X - self.xmin) / self.xmax
        n_sample, n_feature = X.shape
        pol = (X.reshape((n_sample, n_feature, 1)) * X.reshape((n_sample, 1, n_feature))) * (
            1 - np.eye(n_feature)) * (
                  self.V @ self.V.T)
        return np.c_[np.ones(n_sample), X] @ self.w.reshape(-1) + 0.5 * np.sum(pol, axis=(1, 2))
