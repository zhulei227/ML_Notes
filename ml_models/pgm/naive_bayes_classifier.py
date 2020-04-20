"""
朴素贝叶斯分类器实现
"""
import numpy as np
from .. import utils
from ..wrapper_models import DataBinWrapper


class NaiveBayesClassifier(object):
    def __init__(self, max_bins=10):
        """
        :param max_bins:为了方便，对x每维特征做分箱
        """
        self.dbw = DataBinWrapper(max_bins=max_bins)
        # 记录模型参数
        self.default_y_prob = None  # y的默认概率
        self.default_x_prob = {}  # x的默认概率
        self.p_y = {}  # p(y)
        self.p_x_y = {}  # p(x | y)
        self.class_num = None

    def fit(self, x, y):
        self.default_y_prob = np.log(0.5 / (y.max()+1))
        # 分箱
        self.dbw.fit(x)
        x_bins = self.dbw.transform(x)
        # 参数估计
        self.class_num = y.max() + 1
        for y_index in range(0, self.class_num):
            # p(y)
            y_n_sample = np.sum(y == y_index)
            self.default_x_prob[y_index] = np.log(0.5 / y_n_sample)
            x_y = x_bins[y == y_index]
            self.p_y[y_index] = np.log(y_n_sample / (self.class_num + len(y)))
            self.p_x_y[y_index] = {}
            # p(x | y)
            for i in range(0, x_y.shape[1]):
                self.p_x_y[y_index][i] = {}
                x_i_feature_set = set(x_y[:, i])
                for x_feature in x_i_feature_set:
                    self.p_x_y[y_index][i][x_feature] = np.log(
                        np.sum(x_y[:, i] == x_feature) / (y_n_sample + len(x_i_feature_set)))

    def predict_proba(self, x):
        x_bins = self.dbw.transform(x)
        rst = []
        for x_row in x_bins:
            tmp = []
            for y_index in range(0, self.class_num):
                try:
                    p_y_log = self.p_y[y_index]
                except:
                    p_y_log = self.default_y_prob
                for i,xij in enumerate(x_row):
                    try:
                        p_y_log += self.p_x_y[y_index][i][xij]
                    except:
                        p_y_log += self.default_x_prob[y_index]
                tmp.append(p_y_log)
            rst.append(tmp)
        return utils.softmax(np.asarray(rst))

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)
