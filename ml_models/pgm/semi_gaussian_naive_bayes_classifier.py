"""
半朴素贝叶斯分类器实现
"""
import numpy as np
from .. import utils


class SemiGaussianNBClassifier(object):
    def __init__(self, link_rulers=None):
        """
        :param link_rulers: 属性间的链接关系[(x1,x2),(x3,x4)]
        """
        self.p_y = {}  # p(y)
        self.p_x_y = {}  # p(x | y)
        self.class_num = None
        self.link_rulers = link_rulers
        # check link_rulers，由于某一个属性最多仅依赖于另一个属性，所以某一属性在尾部出现次数不可能大于1次
        self.tail_link_rulers = {}
        if self.link_rulers is not None and len(self.link_rulers) > 0:
            for x1, x2 in self.link_rulers:
                if x2 in self.tail_link_rulers:
                    raise Exception("属性依赖超过1次")
                self.tail_link_rulers[x2] = [x1, x2]

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
                # 如果i在link_rulers的尾部，则需要构建二维高斯分布
                if i in self.tail_link_rulers:
                    first_feature, second_feature = self.tail_link_rulers[i]
                    u = np.mean(x_y[:, [first_feature, second_feature]], axis=0)
                    sigma = np.cov(x_y[:, [first_feature, second_feature]].T)
                else:
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
                    if j in self.tail_link_rulers:
                        first_feature, second_feature = self.tail_link_rulers[j]
                        xij = x_row[[first_feature, second_feature]]
                        p_y_log += np.log(utils.gaussian_nd(xij, self.p_x_y[y_index][j][0], self.p_x_y[y_index][j][1]))
                    else:
                        xij = x_row[j]
                        p_y_log += np.log(utils.gaussian_1d(xij, self.p_x_y[y_index][j][0], self.p_x_y[y_index][j][1]))
                tmp.append(p_y_log)
            rst.append(tmp)
        return utils.softmax(np.asarray(rst)).reshape(x.shape[0], self.class_num)

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1).reshape(-1)
