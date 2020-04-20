"""
使用EM算法进行GaussianNB聚类
"""
import numpy as np
from ml_models import utils


class GaussianNBCluster(object):
    def __init__(self, n_components=1, tol=1e-5, n_iter=100, verbose=False):
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

        # 参数
        self.p_y = {}
        self.p_x_y = {}
        # 默认参数
        self.default_y_prob = None  # y的默认概率

    def get_w(self, x):
        """
        获取隐变量
        :return:
        """
        W = []
        for k in range(0, self.n_components):
            tmp = []
            for j in range(0, x.shape[1]):
                tmp.append(np.log(utils.gaussian_1d(x[:, j], self.p_x_y[k][j][0], self.p_x_y[k][j][1])))
            W.append(np.sum(tmp, axis=0) + np.log(self.p_y[k]))
        W = np.asarray(W)
        return np.exp(W.T)

    def fit(self, x):
        n_sample = x.shape[0]

        # 初始化模型参数
        self.default_y_prob = np.log(0.5 / self.n_components)  # 默认p_y
        for y_label in range(0, self.n_components):
            self.p_x_y[y_label] = {}
            self.p_y[y_label] = 1.0 / self.n_components  # 初始p_y设置一样
            # 初始p_x_y设置一样
            for j in range(0, x.shape[1]):
                self.p_x_y[y_label][j] = {}
                u = np.mean(x[:, j], axis=0) + np.random.random() * (x[:, j].max() + x[:, j].min()) / 2
                sigma = np.std(x[:, j])
                self.p_x_y[y_label][j] = [u, sigma]

        # 计算隐变量
        W= self.get_w(x)
        current_log_loss = np.log(W.sum(axis=1)).sum()
        W = W / np.sum(W, axis=1, keepdims=True)
        # 迭代训练
        current_epoch = 0
        for _ in range(0, self.n_iter):
            if self.verbose is True:
                utils.plot_decision_function(x, self.predict(x), self)
                utils.plt.pause(0.1)
                utils.plt.clf()
            # 更新模型参数
            for k in range(0, self.n_components):
                self.p_y[k] = np.sum(W[:, k]) / n_sample
                for j in range(0, x.shape[1]):
                    x_j = x[:, j]
                    u = np.sum(x_j * W[:, k]) / np.sum(W[:, k])
                    sigma = np.sqrt(np.sum((x_j - u) * (x_j - u) * W[:, k]) / np.sum(W[:, k]))
                    self.p_x_y[k][j] = [u, sigma]

            # 更新隐变量
            W = self.get_w(x)
            new_log_loss = np.log(W.sum(axis=1)).sum()
            W = W / np.sum(W, axis=1, keepdims=True)
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
        rst = []
        for x_row in x:
            tmp = []
            for y_index in range(0, self.n_components):
                try:
                    p_y_log = self.p_y[y_index]
                except:
                    p_y_log = self.default_y_prob
                for i, xij in enumerate(x_row):
                    p_y_log += np.log(
                        1e-12 + utils.gaussian_1d(xij, self.p_x_y[y_index][i][0], self.p_x_y[y_index][i][1]))
                tmp.append(p_y_log)
            rst.append(tmp)
        return utils.softmax(np.asarray(rst))

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)
