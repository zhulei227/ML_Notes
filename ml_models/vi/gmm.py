"""
高斯混合模型的变分推断实现
"""
from ml_models import utils
import numpy as np


class GMMCluster(object):
    def __init__(self, n_components=1, tol=1e-5, n_iter=100, prior_dirichlet_alpha=None, prior_gaussian_mean=None,
                 prior_gaussian_beta=None, prior_wishart_w=None, prior_wishart_v=None,
                 verbose=False):
        """
        GMM的VI实现
        :param n_components: 高斯混合模型数量
        :param tol: -log likehold增益<tol时，停止训练
        :param n_iter: 最多迭代次数
        :prior_dirichlet_alpha:先验狄利克雷分布的超参数
        :prior_gaussian_mean:先验高斯分布的均值
        :prior_gaussian_beta:先验高斯分布的beta值，即精度前的系数
        :prior_wishart_w:先验wishart分布的w
        :prior_wishart_v:先验wishart分布的v
        :param verbose: 是否可视化训练过程
        """
        self.n_components = n_components
        self.tol = tol
        self.n_iter = n_iter
        self.verbose = verbose
        self.prior_dirichlet_alpha = prior_dirichlet_alpha
        self.prior_gaussian_mean = prior_gaussian_mean
        self.prior_gaussian_beta = prior_gaussian_beta
        self.prior_wishart_w = prior_wishart_w
        self.prior_wishart_v = prior_wishart_v
        # 填充默认值
        if self.prior_dirichlet_alpha is None:
            self.prior_dirichlet_alpha_0 = np.asarray([1] * n_components)
        self.prior_dirichlet_alpha = self.prior_dirichlet_alpha_0 + np.random.random(size=n_components) * 0.1
        if self.prior_gaussian_beta is None:
            self.prior_gaussian_beta_0 = np.asarray([1] * n_components)
        self.prior_gaussian_beta = self.prior_gaussian_beta_0 + np.random.random(size=n_components) * 0.1

        # 高斯模型参数
        self.params = []
        # 记录数据维度
        self.D = None

    def _init_params(self):
        """
        初始化另一部分参数
        :return:
        """
        if self.prior_gaussian_mean is None:
            self.prior_gaussian_mean_0 = np.zeros(shape=(self.n_components, self.D))
        self.prior_gaussian_mean = self.prior_gaussian_mean_0 + np.random.random(
            size=(self.n_components, self.D)) * 0.1
        if self.prior_wishart_w is None:
            self.prior_wishart_w_0 = [np.identity(self.D)] * self.n_components  # 单位矩阵
        self.prior_wishart_w = []
        for w in self.prior_wishart_w_0:
            self.prior_wishart_w.append(w + np.random.random(size=(self.D, self.D)) * 0.1)

        if self.prior_wishart_v is None:
            self.prior_wishart_v_0 = np.asarray([self.D] * self.n_components)
        self.prior_wishart_v = self.prior_wishart_v_0 + np.random.random(size=self.n_components) * 0.1

    def _update_single_step(self, X):
        # 首先计算3个期望
        E_ln_Lambda = []
        for k in range(0, self.n_components):
            value = self.D * np.log(2) + np.log(np.linalg.det(self.prior_wishart_w[k]))
            for i in range(1, self.D + 1):
                value += utils.special.digamma((self.prior_wishart_v[k] + 1 - i) / 2)
            E_ln_Lambda.append(value)
        E_ln_pi = []
        hat_alpha = np.sum(self.prior_dirichlet_alpha)
        for k in range(0, self.n_components):
            E_ln_pi.append(utils.special.digamma(self.prior_dirichlet_alpha[k]) - utils.special.digamma(hat_alpha))
        E_mu_Lambda = []
        for k in range(0, self.n_components):
            value = self.D * (1.0 / self.prior_gaussian_beta[k]) + np.sum(self.prior_wishart_v[k] * (
                X - self.prior_gaussian_mean[k]) @ self.prior_wishart_w[k] * (X - self.prior_gaussian_mean[k]), axis=1)
            E_mu_Lambda.append(value)
        # 然后计算 r_nk
        rho_n_k = []
        for k in range(0, self.n_components):
            value = np.exp(E_ln_pi[k] + 0.5 * E_ln_Lambda[k] - self.D / 2.0 * np.log(2 * np.pi) - 0.5 * E_mu_Lambda[k])
            rho_n_k.append(value)
        rho_n_k = np.asarray(rho_n_k).T
        r_n_k = rho_n_k / np.sum(np.asarray(rho_n_k), axis=1, keepdims=True)

        # 然后计算N_k,\bar{x}_k,S_k
        N_k = np.sum(r_n_k, axis=0)
        x_k = []
        for k in range(0, self.n_components):
            x_k.append(np.sum(r_n_k[:, [k]] * X, axis=0) / N_k[k])
        S_k = []
        for k in range(0, self.n_components):
            S_k.append(np.transpose(r_n_k[:, [k]] * (X - x_k[k])) @ (r_n_k[:, [k]] * (X - x_k[k])) / N_k[k])

        # 最后更新变分分布中的各个参数
        for k in range(0, self.n_components):
            self.prior_dirichlet_alpha[k] = self.prior_dirichlet_alpha_0[k] + N_k[k]

            self.prior_gaussian_beta[k] = self.prior_gaussian_beta_0[k] + N_k[k]

            self.prior_gaussian_mean[k] = 1.0 / self.prior_gaussian_beta[k] * (
                self.prior_gaussian_beta_0[k] * self.prior_gaussian_mean_0[k] + N_k[k] * x_k[k])

            W_k_inv = np.linalg.inv(self.prior_wishart_w_0[k]) + N_k[k] * S_k[k] + (self.prior_gaussian_beta_0[k] * N_k[
                k]) / (self.prior_gaussian_beta_0[k] + N_k[k]) * np.transpose(
                x_k[k] - self.prior_gaussian_mean_0[k]) @ (x_k[k] - self.prior_gaussian_mean_0[k])

            self.prior_wishart_w[k] = np.linalg.inv(W_k_inv)

            self.prior_wishart_v[k] = self.prior_wishart_v_0[k] + N_k[k]

    def fit(self, X):
        n_sample, n_feature = X.shape
        self.D = n_feature
        self._init_params()
        last_rst = np.zeros(n_sample)
        # 迭代训练
        for _ in range(0, self.n_iter):
            self._update_single_step(X)
            if self.verbose:
                utils.plot_contourf(X, lambda x: self.predict_sample_generate_proba(x), lines=5)
                utils.plt.pause(0.1)
                utils.plt.clf()
            current_rst = self.predict_sample_generate_proba(X)
            if np.mean(np.abs(current_rst - last_rst)) < self.tol:
                break
            last_rst = current_rst

        if self.verbose:
            utils.plot_contourf(X, lambda x: self.predict_sample_generate_proba(x), lines=5)
            utils.plt.show()

    def predict_proba(self, X):
        # 预测样本在几个St上的概率分布
        _, D = X.shape
        hat_alpha = np.sum(self.prior_dirichlet_alpha)
        W = np.asarray([utils.St(X, self.prior_gaussian_mean[k],
                                 (self.prior_wishart_v[k] + 1 - D) / (1 + self.prior_gaussian_beta[k]) *
                                 self.prior_wishart_w[k] * self.prior_gaussian_beta[k],
                                 self.prior_wishart_v[k] + 1 - D) * self.prior_dirichlet_alpha[
                            k] / hat_alpha for k in range(0, self.n_components)]).T
        W = W / np.sum(W, axis=1, keepdims=True)
        return W

    def predict(self, X):
        # 预测样本最有可能产生于那一个高斯模型
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_sample_generate_proba(self, X):
        # 返回样本的生成概率
        _, D = X.shape
        hat_alpha = np.sum(self.prior_dirichlet_alpha)
        W = np.asarray([utils.St(X, self.prior_gaussian_mean[k],
                                 (self.prior_wishart_v[k] + 1 - D) / (1 + self.prior_gaussian_beta[k]) *
                                 self.prior_wishart_w[k] * self.prior_gaussian_beta[k],
                                 self.prior_wishart_v[k] + 1 - D) * self.prior_dirichlet_alpha[
                            k] / hat_alpha for k in range(0, self.n_components)]).T
        return np.sum(W, axis=1)
