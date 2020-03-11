"""
感知机模型
"""
import numpy as np
import matplotlib.pyplot as plt
from .. import utils


class Perceptron(object):
    def __init__(self, epochs=10, eta=None, mode=None):
        self.w = None
        self.epochs = epochs
        self.eta = eta
        self.mode = mode

    def init_params(self, n_features):
        """
        初始化参数
        :return:
        """
        self.w = np.random.random(size=(n_features + 1, 1))

    def _dual_fit(self, x, y, sample_weight):
        """
        模型训练的对偶形式
        :param x:
        :param y:
        :return:
        """
        y = y.reshape(-1, 1)
        y[y == 0] = -1

        n_samples, n_features = x.shape

        # 初始化参数
        self.alpha = np.zeros(shape=(n_samples, 1))

        x = np.c_[x, np.ones(shape=(n_samples,))]

        for _ in range(self.epochs):
            error_sum = 0
            indices = list(range(0, n_samples))
            np.random.shuffle(indices)
            for index in indices:
                x_i = x[index, :]
                y_i = y[index]
                # 更新错分点的参数，（注意需要有等号，因为初始化的alpha全为0）
                if (x_i.dot(x.T.dot(self.alpha * y)) * y_i)[0] <= 0:
                    self.alpha[index] += self.eta * sample_weight[index]
                    error_sum += 1
            if error_sum == 0:
                break
        # 更新回w
        self.w = x.T.dot(self.alpha * y)

    def fit(self, x, y, sample_weight=None):
        """
        :param x: ndarray格式数据: m x n
        :param y: ndarray格式数据: m x 1
        :param sample_weight: mx1,样本权重
        :return:
        """
        n_sample = x.shape[0]
        if sample_weight is None:
            sample_weight = np.asarray([1.0] * n_sample)
        else:
            sample_weight = sample_weight
        # check sample_weight
        if len(sample_weight) != n_sample:
            raise Exception('sample_weight size error:', len(sample_weight))
        # 设置学习率
        if self.eta is None:
            self.eta = max(1e-2, 1.0 / np.sqrt(x.shape[0]))
        if self.mode == "dual":
            self._dual_fit(x, y, sample_weight)
            return
        y = y.reshape(-1, 1)
        y[y == 0] = -1
        # 初始化参数w,b
        n_samples, n_features = x.shape
        self.init_params(n_features)
        x = np.c_[x, np.ones(shape=(n_samples,))]
        x_y = np.c_[x, y]

        for _ in range(self.epochs):
            error_sum = 0
            np.random.shuffle(x_y)
            for index in range(0, n_samples):
                x_i = x_y[index, :-1]
                y_i = x_y[index, -1:]
                # 更新错分点的参数
                if (x_i.dot(self.w) * y_i)[0] < 0:
                    dw = (-x_i * y_i).reshape(-1, 1)
                    # 考虑sample_weight
                    dw = dw * sample_weight[index]
                    self.w = self.w - self.eta * dw
                    error_sum += 1
            if error_sum == 0:
                break

    def get_params(self):
        """
        输出原始的系数
        :return: w
        """

        return self.w

    def predict(self, x):
        """
        :param x:ndarray格式数据: m x n
        :return: m x 1
        """
        return np.argmax(self.predict_proba(x), axis=1)

    def predict_proba(self, x):
        """
        :param x:ndarray格式数据: m x n
        :return: m x 1
        """
        n_samples = x.shape[0]
        x = np.c_[x, np.ones(shape=(n_samples,))]
        return np.c_[1.0 - utils.sigmoid(x.dot(self.w)), utils.sigmoid(x.dot(self.w))]

    def plot_decision_boundary(self, x, y):
        """
        绘制前两个维度的决策边界
        :param x:
        :param y:
        :return:
        """
        weights = self.get_params()
        w1 = weights[0][0]
        w2 = weights[1][0]
        bias = weights[-1][0]
        x1 = np.arange(np.min(x), np.max(x), 0.1)
        x2 = -w1 / w2 * x1 - bias / w2
        plt.scatter(x[:, 0], x[:, 1], c=y, s=50)
        plt.plot(x1, x2, 'r')
        plt.show()
