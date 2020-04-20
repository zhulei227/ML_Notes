"""
使用EM算法进行NB聚类
"""
from ml_models.wrapper_models import DataBinWrapper
import numpy as np
from ml_models import utils


class NaiveBayesCluster(object):
    def __init__(self, n_components=1, tol=1e-5, n_iter=100, max_bins=10, verbose=False):
        """
        :param n_components: 朴素贝叶斯模型数量
        :param tol: log likehold增益<tol时，停止训练
        :param n_iter: 最多迭代次数
        :param verbose: 是否可视化训练过程
        """
        self.n_components = n_components
        self.tol = tol
        self.n_iter = n_iter
        self.verbose = verbose
        # 分箱
        self.dbw = DataBinWrapper(max_bins=max_bins)
        # 参数
        self.p_y = {}
        self.p_x_y = {}
        # 默认参数
        self.default_y_prob = None  # y的默认概率
        self.default_x_prob = {}  # x的默认概率

    def get_log_w(self, x_bins):
        """
        获取隐变量
        :param x_bins:
        :return:
        """
        W = []
        for x_row in x_bins:
            tmp = []
            for k in range(0, self.n_components):
                llh = self.p_y[k]
                for j, x_ij in enumerate(x_row):
                    llh += self.p_x_y[k][j][x_ij]
                tmp.append(llh)
            W.append(tmp)
        W = np.asarray(W)
        return W

    def fit(self, x):
        n_sample = x.shape[0]
        self.dbw.fit(x)
        x_bins = self.dbw.transform(x)
        # 初始化模型参数
        self.default_y_prob = np.log(0.5 / self.n_components)  # 默认p_y
        for y_label in range(0, self.n_components):
            self.p_x_y[y_label] = {}
            self.p_y[y_label] = np.log(1.0 / self.n_components)  # 初始p_y设置一样
            self.default_x_prob[y_label] = np.log(0.5 / n_sample)  # 默认p_x_y
            # 初始p_x_y设置一样
            for j in range(0, x_bins.shape[1]):
                self.p_x_y[y_label][j] = {}
                x_j_set = set(x_bins[:, j])
                for x_j in x_j_set:
                    # 随机抽样计算条件概率
                    sample_x_index = np.random.choice(n_sample, n_sample // self.n_components)
                    sample_x_bins = x_bins[sample_x_index]
                    p_x_y = (np.sum(sample_x_bins[:, j] == x_j) + 1) / (
                        sample_x_bins.shape[0] + len(x_j_set))
                    self.p_x_y[y_label][j][x_j] = np.log(p_x_y)
        # 计算隐变量
        W_log = self.get_log_w(x_bins)
        W = utils.softmax(W_log)
        W_gen = np.exp(W_log)
        current_log_loss = np.log(W_gen.sum(axis=1)).sum()
        # 迭代训练
        current_epoch = 0
        for _ in range(0, self.n_iter):
            if self.verbose is True:
                utils.plot_decision_function(x, self.predict(x), self)
                utils.plt.pause(0.1)
                utils.plt.clf()
            # 更新模型参数
            for k in range(0, self.n_components):
                self.p_y[k] = np.log(np.sum(W[:, k]) / n_sample)
                for j in range(0, x_bins.shape[1]):
                    x_j_set = set(x_bins[:, j])
                    for x_j in x_j_set:
                        self.p_x_y[k][j][x_j] = np.log(
                            1e-10 + np.sum(W[:, k] * (x_bins[:, j] == x_j)) / np.sum(W[:, k]))

            # 更新隐变量
            W_log = self.get_log_w(x_bins)
            W = utils.softmax(W_log)
            W_gen = np.exp(W_log)
            # 计算log like hold
            new_log_loss = np.log(W_gen.sum(axis=1)).sum()
            if new_log_loss - current_log_loss > self.tol:
                current_log_loss = new_log_loss
                current_epoch += 1
            else:
                print('total epochs:', current_epoch)
                break
        if self.verbose:
            utils.plot_decision_function(x, self.predict(x), self)
            utils.plt.show()

    def predict_proba(self, x):
        x_bins = self.dbw.transform(x)
        rst = []
        for x_row in x_bins:
            tmp = []
            for y_index in range(0, self.n_components):
                try:
                    p_y_log = self.p_y[y_index]
                except:
                    p_y_log = self.default_y_prob
                for i, xij in enumerate(x_row):
                    try:
                        p_y_log += self.p_x_y[y_index][i][xij]
                    except:
                        p_y_log += self.default_x_prob[y_index]
                tmp.append(p_y_log)
            rst.append(tmp)
        return utils.softmax(np.asarray(rst))

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)
