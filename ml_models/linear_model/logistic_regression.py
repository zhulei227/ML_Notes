"""
Logistic回归
"""
import numpy as np
import matplotlib.pyplot as plt
from .. import utils
from .. import optimization


class LogisticRegression(object):
    def __init__(self, fit_intercept=True, solver='sgd', if_standard=True, l1_ratio=None, l2_ratio=None, epochs=10,
                 eta=None, batch_size=16):

        self.w = None
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.if_standard = if_standard
        if if_standard:
            self.feature_mean = None
            self.feature_std = None
        self.epochs = epochs
        self.eta = eta
        self.batch_size = batch_size
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        # 注册sign函数
        self.sign_func = np.vectorize(utils.sign)
        # 记录losses
        self.losses = []

    def init_params(self, n_features):
        """
        初始化参数
        :return:
        """
        self.w = np.random.random(size=(n_features, 1))

    def _fit_sgd(self, x, y, sample_weight):
        """
        随机梯度下降求解
        :param x:
        :param y:
        :return:
        """
        x_y = np.c_[x, y]
        count = 0
        for _ in range(self.epochs):
            np.random.shuffle(x_y)
            for index in range(x_y.shape[0] // self.batch_size):
                count += 1
                batch_x_y = x_y[self.batch_size * index:self.batch_size * (index + 1)]
                batch_x = batch_x_y[:, :-1]
                batch_y = batch_x_y[:, -1:]

                # 考虑sample_weight
                sample_weight_diag = np.diag(sample_weight[self.batch_size * index:self.batch_size * (index + 1)])
                sample_weight_mean = np.mean(sample_weight[self.batch_size * index:self.batch_size * (index + 1)])

                dw = -1 * (batch_y - utils.sigmoid(batch_x.dot(self.w))).T.dot(sample_weight_diag).dot(
                    batch_x) / self.batch_size
                dw = dw.T

                # 添加l1和l2的部分
                dw_reg = np.zeros(shape=(x.shape[1] - 1, 1))
                if self.l1_ratio is not None:
                    dw_reg += sample_weight_mean * self.l1_ratio * self.sign_func(self.w[:-1]) / self.batch_size
                if self.l2_ratio is not None:
                    dw_reg += 2 * sample_weight_mean * self.l2_ratio * self.w[:-1] / self.batch_size
                dw_reg = np.concatenate([dw_reg, np.asarray([[0]])], axis=0)

                dw += dw_reg

                if self.solver == 'dfp':
                    if self.dfp is None:
                        self.dfp = optimization.DFP(x0=self.w, g0=dw)
                    else:
                        # 更新一次拟牛顿矩阵
                        self.dfp.update_quasi_newton_matrix(self.w, dw)
                    # 调整梯度方向
                    dw = self.dfp.adjust_gradient(dw)

                if self.solver == 'bfgs':
                    if self.bfgs is None:
                        self.bfgs = optimization.BFGS(x0=self.w, g0=dw)
                    else:
                        # 更新一次拟牛顿矩阵
                        self.bfgs.update_quasi_newton_matrix(self.w, dw)
                    # 调整梯度方向
                    dw = self.bfgs.adjust_gradient(dw)
                self.w = self.w - self.eta * dw

            # 计算losses
            cost = -1 * np.sum(
                np.multiply(y, np.log(utils.sigmoid(x.dot(self.w)))) + np.multiply(1 - y, np.log(
                    1 - utils.sigmoid(x.dot(self.w)))))
            self.losses.append(cost)

    def fit(self, x, y2, sample_weight=None):
        """
        :param x: ndarray格式数据: m x n
        :param y2: ndarray格式数据: m
        :return:
        """
        n_sample = x.shape[0]
        if sample_weight is None:
            sample_weight = np.asarray([1.0] * n_sample)
        # check sample_weight
        if len(sample_weight) != n_sample:
            raise Exception('sample_weight size error:', len(sample_weight))
        y = y2.reshape(x.shape[0], 1)
        # 是否归一化feature
        if self.if_standard:
            self.feature_mean = np.mean(x, axis=0)
            self.feature_std = np.std(x, axis=0) + 1e-8
            x = (x - self.feature_mean) / self.feature_std
        # 是否训练bias
        if self.fit_intercept:
            x = np.c_[x, np.ones_like(y)]
        # 初始化参数
        self.init_params(x.shape[1])
        # 更新eta
        if self.eta is None:
            self.eta = self.batch_size / np.sqrt(x.shape[0])

        if self.solver == 'sgd':
            self._fit_sgd(x, y, sample_weight)
        elif self.solver == 'dfp':
            self.dfp = None
            self._fit_sgd(x, y, sample_weight)
        elif self.solver == 'bfgs':
            self.bfgs = None
            self._fit_sgd(x, y, sample_weight)

    def get_params(self):
        """
        输出原始的系数
        :return: w,b
        """
        if self.fit_intercept:
            w = self.w[:-1]
            b = self.w[-1]
        else:
            w = self.w
            b = 0
        if self.if_standard:
            w = w / self.feature_std.reshape(-1, 1)
            b = b - w.T.dot(self.feature_mean.reshape(-1, 1))
        return w.reshape(-1), b

    def predict_proba(self, x):
        """
        预测为y=1的概率
        :param x:ndarray格式数据: m x n
        :return: m x 2
        """
        if self.if_standard:
            x = (x - self.feature_mean) / self.feature_std
        if self.fit_intercept:
            x = np.c_[x, np.ones(x.shape[0])]
        return np.c_[1.0 - utils.sigmoid(x.dot(self.w)), utils.sigmoid(x.dot(self.w))]

    def predict(self, x):
        """
        预测类别，默认大于0.5的为1，小于0.5的为0
        :param x:
        :return:
        """
        return np.argmax(self.predict_proba(x), axis=1)

    def plot_decision_boundary(self, x, y):
        """
        绘制前两个维度的决策边界
        :param x:
        :param y:
        :return:
        """
        y = y.reshape(-1)
        weights, bias = self.get_params()
        w1 = weights[0]
        w2 = weights[1]
        bias = bias[0][0]
        x1 = np.arange(np.min(x), np.max(x), 0.1)
        x2 = -w1 / w2 * x1 - bias / w2
        plt.scatter(x[:, 0], x[:, 1], c=y, s=50)
        plt.plot(x1, x2, 'r')
        plt.show()

    def plot_losses(self):
        plt.plot(range(0, len(self.losses)), self.losses)
        plt.show()
