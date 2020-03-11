"""
硬间隔支持向量机的smo实现
"""
import numpy as np
from .. import utils
import copy
import random
from .. import kernel_functions


class SVC(object):
    def __init__(self, epochs=100, C=1.0, tol=1e-3, kernel=None, degree=3, gamma=0.1):
        """
        :param epochs: 迭代次数上限
        :param C: C越小，对于误分类的惩罚越小
        :param tol:提前中止训练时的误差值上限
        :param kernel:核函数
        :param degree:kernel='poly'时生效
        :param gamma:kernel='rbf'时生效
        """
        self.b = None
        self.alpha = None
        self.E = None
        self.epochs = epochs
        self.C = C
        self.tol = tol
        # 定义核函数
        if kernel is None:
            self.kernel_function = kernel_functions.linear()
        elif kernel == 'poly':
            self.kernel_function = kernel_functions.poly(degree)
        elif kernel == 'rbf':
            self.kernel_function = kernel_functions.rbf(gamma)
        else:
            self.kernel_function = kernel_functions.linear()
        # 记录支持向量
        self.support_vectors = None
        # 记录支持向量的x
        self.support_vector_x = []
        # 记录支持向量的y
        self.support_vector_y = []
        # 记录支持向量的alpha
        self.support_vector_alpha = []

    def f(self, x):
        """
        :param x:
        :return: wx+b
        """
        x_np = np.asarray(x)
        if len(self.support_vector_x) == 0:
            if x_np.ndim <= 1:
                return 0
            else:
                return np.zeros((x_np.shape[:-1]))
        else:
            if x_np.ndim <= 1:
                wx = 0
            else:
                wx = np.zeros((x_np.shape[:-1]))
            for i in range(0, len(self.support_vector_x)):
                wx += self.kernel_function(x, self.support_vector_x[i]) * self.support_vector_alpha[i] * \
                      self.support_vector_y[i]
            return wx + self.b

    def init_params(self, X, y):
        """
        :param X: (n_samples,n_features)
        :param y: (n_samples,) y_i\in\{0,1\}
        :return:
        """
        n_samples, n_features = X.shape
        self.b = .0
        self.alpha = np.zeros(n_samples)
        self.E = np.zeros(n_samples)
        # 初始化E
        for i in range(0, n_samples):
            self.E[i] = self.f(X[i, :]) - y[i]

    def _select_j(self, best_i):
        """
        选择j
        :param best_i:
        :return:
        """
        valid_j_list = [i for i in range(0, len(self.alpha)) if self.alpha[i] > 0 and i != best_i]
        best_j = -1
        # 优先选择使得|E_i-E_j|最大的j
        if len(valid_j_list) > 0:
            max_e = 0
            for j in valid_j_list:
                current_e = np.abs(self.E[best_i] - self.E[j])
                if current_e > max_e:
                    best_j = j
                    max_e = current_e
        else:
            # 随机选择
            l = list(range(len(self.alpha)))
            seq = l[: best_i] + l[best_i + 1:]
            best_j = random.choice(seq)
        return best_j

    def _meet_kkt(self, x_i, y_i, alpha_i, weight):
        """
        判断是否满足KKT条件

        :param w:
        :param b:
        :param x_i:
        :param y_i:
        :param weight:样本weight
        :return:
        """
        if alpha_i < self.C * weight:
            return y_i * self.f(x_i) >= 1 - self.tol
        else:
            return y_i * self.f(x_i) <= 1 + self.tol

    def fit(self, X, y2, show_train_process=False, sample_weight=None):
        """

        :param X:
        :param y2:
        :param show_train_process: 显示训练过程
        :param sample_weight:样本权重
        :return:
        """
        n_sample = X.shape[0]
        if sample_weight is None:
            sample_weight = np.asarray([1.0] * n_sample)
        # check sample_weight
        if len(sample_weight) != n_sample:
            raise Exception('sample_weight size error:', len(sample_weight))
        y = copy.deepcopy(y2)
        y[y == 0] = -1
        # 初始化参数
        self.init_params(X, y)
        for _ in range(0, self.epochs):
            if_all_match_kkt = True
            for i in range(0, len(self.alpha)):
                x_i = X[i, :]
                y_i = y[i]
                alpha_i_old = self.alpha[i]
                E_i_old = self.E[i]
                # 外层循环：选择违反KKT条件的点i
                if not self._meet_kkt(x_i, y_i, alpha_i_old, sample_weight[i]):
                    if_all_match_kkt = False
                    # 内层循环，选择使|Ei-Ej|最大的点j
                    best_j = self._select_j(i)

                    alpha_j_old = self.alpha[best_j]
                    x_j = X[best_j, :]
                    y_j = y[best_j]
                    E_j_old = self.E[best_j]

                    # 进行更新
                    # 1.首先获取无裁剪的最优alpha_2
                    eta = self.kernel_function(x_i, x_i) + self.kernel_function(x_j, x_j) - 2.0 * self.kernel_function(
                        x_i, x_j)
                    # 如果x_i和x_j很接近，则跳过
                    if eta < 1e-3:
                        continue
                    alpha_j_unc = alpha_j_old + y_j * (E_i_old - E_j_old) / eta
                    # 2.裁剪并得到new alpha_2
                    if y_i == y_j:
                        L = max(0., alpha_i_old + alpha_j_old - self.C * sample_weight[best_j])
                        H = min(self.C * sample_weight[best_j], alpha_i_old + alpha_j_old)
                    else:
                        L = max(0, alpha_j_old - alpha_i_old)
                        H = min(self.C * sample_weight[best_j],
                                self.C * sample_weight[best_j] + alpha_j_old - alpha_i_old)

                    if alpha_j_unc < L:
                        alpha_j_new = L
                    elif alpha_j_unc > H:
                        alpha_j_new = H
                    else:
                        alpha_j_new = alpha_j_unc

                    # 如果变化不够大则跳过
                    if np.abs(alpha_j_new - alpha_j_old) < 1e-5:
                        continue
                    # 3.得到alpha_1_new
                    alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new)
                    # 5.更新alpha_1,alpha_2
                    self.alpha[i] = alpha_i_new
                    self.alpha[best_j] = alpha_j_new
                    # 6.更新b
                    b_i_new = y_i - self.f(x_i) + self.b
                    b_j_new = y_j - self.f(x_j) + self.b
                    if self.C * sample_weight[i] > alpha_i_new > 0:
                        self.b = b_i_new
                    elif self.C * sample_weight[best_j] > alpha_j_new > 0:
                        self.b = b_j_new
                    else:
                        self.b = (b_i_new + b_j_new) / 2.0
                    # 7.更新E
                    for k in range(0, len(self.E)):
                        self.E[k] = self.f(X[k, :]) - y[k]

                    # 8.更新支持向量相关的信息
                    self.support_vectors = np.where(self.alpha > 1e-3)[0]
                    self.support_vector_x = [X[i, :] for i in self.support_vectors]
                    self.support_vector_y = [y[i] for i in self.support_vectors]
                    self.support_vector_alpha = [self.alpha[i] for i in self.support_vectors]

                    # 显示训练过程
                    if show_train_process is True:
                        utils.plot_decision_function(X, y2, self, [i, best_j])
                        utils.plt.pause(0.1)
                        utils.plt.clf()

            # 如果所有的点都满足KKT条件，则中止
            if if_all_match_kkt is True:
                break

        # 显示最终结果
        if show_train_process is True:
            utils.plot_decision_function(X, y2, self, self.support_vectors)
            utils.plt.show()

    def predict_proba(self, x):
        """
        :param x:ndarray格式数据: m x n
        :return: m x 2
        """
        return np.c_[1.0 - utils.sigmoid(self.f(x)), utils.sigmoid(self.f(x))]

    def predict(self, x):
        """
        :param x:ndarray格式数据: m x n
        :return: m x 1
        """
        return np.argmax(self.predict_proba(x), axis=1)
