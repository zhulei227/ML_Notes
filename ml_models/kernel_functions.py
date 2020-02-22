"""
定义一些常见的核函数
"""
import numpy as np


def linear():
    """
    线性核函数
    :return:linear function
    """

    def _linear(x, y):
        return np.dot(x, y)

    return _linear


def poly(p=2):
    """
    多项式核函数
    :param p:
    :return: poly function
    """

    def _poly(x, y):
        return np.power(np.dot(x, y) + 1, p)

    return _poly


def rbf(sigma=0.1):
    """
    径向基/高斯核函数
    :param sigma:
    :return:
    """

    def _rbf(x, y):
        np_x = np.asarray(x)
        if np_x.ndim <= 1:
            return np.exp((-1 * np.dot(x - y, x - y) / (2 * sigma * sigma)))
        else:
            return np.exp((-1 * np.multiply(x - y, x - y).sum(axis=1) / (2 * sigma * sigma)))

    return _rbf
