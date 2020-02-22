import numpy as np

"""
DPF拟牛顿法
"""


class DFP(object):
    def __init__(self, x0, g0):
        """

        :param x0: 初始的x
        :param g0: 初始x对应的梯度
        """
        self.x0 = x0
        self.g0 = g0
        # 初始化G0
        self.G0 = np.eye(len(x0))

    def update_quasi_newton_matrix(self, x1, g1):
        """
        更新拟牛顿矩阵
        :param x1:
        :param g1:
        :return:
        """
        # 进行一步更新
        y0 = g1 - self.g0
        delta0 = x1 - self.x0
        self.G0 = self.G0 + delta0.dot(delta0.T) / delta0.T.dot(y0)[0][0] - self.G0.dot(y0).dot(y0.T).dot(self.G0) / \
                                                                            y0.T.dot(self.G0).dot(y0)[0][0]

    def adjust_gradient(self, gradient):
        """
        对原始的梯度做调整
        :param gradient:
        :return:
        """
        return self.G0.dot(gradient)


"""
BFGS拟牛顿法
"""


class BFGS(object):
    def __init__(self, x0, g0):
        """

        :param x0: 初始的x
        :param g0: 初始x对应的梯度
        """
        self.x0 = x0
        self.g0 = g0
        # 初始化B0
        self.B0 = np.eye(len(x0))

    def update_quasi_newton_matrix(self, x1, g1):
        """
        更新拟牛顿矩阵
        :param x1:
        :param g1:
        :return:
        """
        # 进行一步更新
        y0 = g1 - self.g0
        delta0 = x1 - self.x0
        # 使用sherman-morrison公式不稳定
        # divide_value = delta0.T.dot(y0)[0][0]
        # tmp = np.eye(len(y0)) - delta0.dot(y0.T) / divide_value
        # self.G0 = self.G0 + tmp.dot(self.G0).dot(tmp.T) + delta0.dot(delta0.T) / divide_value
        self.B0 = self.B0 + y0.dot(y0.T) / y0.T.dot(delta0)[0][0] - self.B0.dot(delta0).dot(delta0.T).dot(self.B0) / \
                                                                    delta0.T.dot(self.B0).dot(delta0)[0][0]

    def adjust_gradient(self, gradient):
        """
        对原始的梯度做调整
        :param gradient:
        :return:
        """
        return np.linalg.pinv(self.B0).dot(gradient)
