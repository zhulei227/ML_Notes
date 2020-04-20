"""
高斯朴素贝叶斯分类器实现
"""
import numpy as np
from .. import utils


class GaussianNBClassifier(object):
    def __init__(self):
        self.p_y = {}  # p(y)
        self.p_x_y = {}  # p(x | y)
        self.class_num = None

    def fit(self, x, y):
        # 参数估计
        self.class_num = y.max() + 1
        for y_index in range(0, self.class_num):
            # p(y)
            y_n_sample = np.sum(y == y_index)
            self.p_y[y_index] = np.log(y_n_sample / len(y))
            self.p_x_y[y_index] = {}
            # p(x | y)
            x_y = x[y == y_index]
            for i in range(0, x_y.shape[1]):
                u = np.mean(x_y[:, i])
                sigma = np.std(x_y[:, i])
                self.p_x_y[y_index][i] = [u, sigma]

    def predict_proba(self, x):
        rst = []
        for x_row in x:
            tmp = []
            for y_index in range(0, self.class_num):
                p_y_log = self.p_y[y_index]
                for j in range(0, len(x_row)):
                    xij = x_row[j]
                    p_y_log += np.log(utils.gaussian_1d(xij, self.p_x_y[y_index][j][0], self.p_x_y[y_index][j][1]))
                tmp.append(p_y_log)
            rst.append(tmp)
        return utils.softmax(np.asarray(rst))

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)
