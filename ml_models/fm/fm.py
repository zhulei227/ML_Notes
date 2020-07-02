"""
FM因子分解机的简单实现，只实现了损失函数为平方损失的回归任务，更多功能扩展请使用后续的FFM
"""
import numpy as np
from tqdm import tqdm


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
        X_V = X_ @ self.V
        X_V_2 = X_V * X_V
        X_2_V_2 = (X_ * X_) @ (self.V * self.V)
        pol = 0.5 * np.sum(X_V_2 - X_2_V_2, axis=1)
        return X @ self.w.reshape(-1) + pol

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
        self.w = np.random.random((n_feature + 1, 1))*1e-3
        self.V = np.random.random((n_feature, self.hidden_dim))*1e-3
        # 更新参数
        count = 0
        for _ in tqdm(range(self.epochs)):
            np.random.shuffle(x_y)
            for index in tqdm(range(x_y.shape[0] // self.batch_size)):
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
                X_2 = batch_x_ * batch_x_
                for i in range(self.V.shape[0]):
                    for f in range(self.V.shape[1]):
                        self.V[i, f] -= self.lr * (
                            np.sum(y_x_t.reshape(-1) * batch_x_[:, i] * V_X[:, f] - self.V[i, f] * X_2[:,
                                                                                                   i]) / self.batch_size + self.lamb *
                            self.V[i, f])
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
        X_V = X @ self.V
        X_V_2 = X_V * X_V
        X_2_V_2 = (X * X) @ (self.V * self.V)
        pol = 0.5 * np.sum(X_V_2 - X_2_V_2, axis=1)
        return np.c_[np.ones(n_sample), X] @ self.w.reshape(-1) + pol
