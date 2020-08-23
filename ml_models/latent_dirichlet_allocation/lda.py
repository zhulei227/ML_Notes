"""
隐狄利克雷分布的代码实现，包括Gibbs采样和变分EM算法
"""
import numpy as np
from scipy.special import digamma


class LDA(object):
    def __init__(self, alpha=None, beta=None, K=10, tol=1e-3, epochs=100, method="gibbs", lr=1e-5):
        """
        :param alpha: 主题分布的共轭狄利克雷分布的超参数
        :param beta: 单词分布的共轭狄利克雷分布的超参数
        :param K: 主题数量
        :param tol:容忍度，允许tol的隐变量差异
        :param epochs:最大迭代次数
        :param method:优化方法，默认gibbs,另外还有变分EM,vi_em
        :param lr:学习率,对vi_em生效
        """
        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.tol = tol
        self.epochs = epochs
        self.method = method
        self.lr = lr
        self.phi = None  # 主题-单词矩阵

    def _init_params(self, W):
        """
        初始化参数
        :param W:
        :return:
        """
        M = len(W)  # 文本数
        V = 0  # 词典大小
        I = 0  # 单词总数
        for w in W:
            V = max(V, max(w))
            I += len(w)
        V += 1  # 包括0
        # 文本话题计数
        N_M_K = np.zeros(shape=(M, self.K))
        N_M = np.zeros(M)
        # 话题单词计数
        N_K_V = np.zeros(shape=(self.K, V))
        N_K = np.zeros(self.K)
        # 初始化隐状态,计数矩阵
        Z = []  # 隐状态，与W一一对应
        p = [1 / self.K] * self.K
        hidden_status = list(range(self.K))
        for m, w in enumerate(W):
            z = np.random.choice(hidden_status, len(w), replace=True, p=p).tolist()
            Z.append(z)
            for n, k in enumerate(z):
                v = w[n]
                N_M_K[m][k] += 1
                N_M[m] += 1
                N_K_V[k][v] += 1
                N_K[k] += 1
        # 初始化alpha和beta
        if self.alpha is None:
            self.alpha = np.ones(self.K)
        if self.beta is None:
            self.beta = np.ones(V)
        return Z, N_M_K, N_M, N_K_V, N_K, M, V, I, hidden_status

    def _fit_gibbs(self, W):
        """
        :param W: 文本集合[[...],[...]]
        :return:
        """
        Z, N_M_K, N_M, N_K_V, N_K, M, V, I, hidden_status = self._init_params(W)
        for _ in range(self.epochs):
            error_num = 0
            for m, w in enumerate(W):
                z = Z[m]
                for n, topic in enumerate(z):
                    word = w[n]
                    N_M_K[m][topic] -= 1
                    N_M[m] -= 1
                    N_K_V[topic][word] -= 1
                    N_K[topic] -= 1
                    # 采样一个新k
                    p = []  # 更新多项分布
                    for k_ in range(self.K):
                        p_ = (N_K_V[k_][word] + self.beta[word]) * (N_M_K[m][k_] + self.alpha[topic]) / (
                            (N_K[k_] + np.sum(self.beta)) * (N_M[m] + np.sum(self.alpha)))
                        p.append(p_)
                    ps = np.sum(p)
                    p = [p_ / ps for p_ in p]
                    topic_new = np.random.choice(hidden_status, 1, p=p)[0]
                    if topic_new != topic:
                        error_num += 1
                    Z[m][n] = topic_new
                    N_M_K[m][topic_new] += 1
                    N_M[m] += 1
                    N_K_V[topic_new][word] += 1
                    N_K[topic_new] += 1
            if error_num / I < self.tol:
                break

        # 计算参数phi
        self.phi = N_K_V / np.sum(N_K_V, axis=1, keepdims=True)

    def _fit_vi_em(self, W):
        """
        分为两部分，迭代计算：
        （1）给定lda参数，更新变分参数
        （2）给定变分参数，更新lda参数
        :param W:
        :return:
        """
        V = 0  # 词典大小
        for w in W:
            V = max(V, max(w))
        V += 1
        M = len(W)

        # 给定lda参数，更新变分参数
        def update_vi_params(alpha, phi):
            eta = []
            gamma = []
            for w in W:
                N = len(w)
                eta_old = np.ones(shape=(N, self.K)) * (1 / self.K)
                gamma_old = alpha + N / self.K
                eta_new = np.zeros_like(eta_old)
                for _ in range(self.epochs):
                    for n in range(0, N):
                        for k in range(0, self.K):
                            eta_new[n, k] = phi[k, w[n]] * np.exp(digamma(gamma_old[k]) - digamma(np.sum(gamma_old)))
                    eta_new = eta_new / np.sum(eta_new, axis=1, keepdims=True)
                    gamma_new = alpha + np.sum(eta_new, axis=0)
                    if (np.sum(np.abs(gamma_new - gamma_old)) + np.sum(np.abs((eta_new - eta_old)))) / (
                                (N + 1) * self.K) < self.tol:
                        break
                    else:
                        eta_old = eta_new.copy()
                        gamma_old = gamma_new.copy()
                eta.append(eta_new)
                gamma.append(gamma_new)
            return eta, gamma

        # 给定变分参数，更新lda参数
        def update_lda_params(eta, gamma, alpha_old):
            # 更新phi
            phi = np.zeros(shape=(self.K, V))
            for m, w in enumerate(W):
                for n, word in enumerate(w):
                    for k in range(0, self.K):
                        for v in range(0, V):
                            phi[k, v] += eta[m][n, k] * (word == v)
            # 更新alpha
            d_alpha = []
            for k, alpha_ in enumerate(alpha_old):
                tmp = M * (digamma(np.sum(alpha_old)) - digamma(alpha_))
                for m in range(M):
                    tmp -= (digamma(gamma[m][k]) - digamma(np.sum(gamma[m])))
                d_alpha.append(tmp)
            alpha_new = alpha_old - self.lr * np.asarray(d_alpha)
            alpha_new = np.where(alpha_new < 0.0, 0.0, alpha_new)
            alpha_new = alpha_new / (1e-9 + np.sum(alpha_new)) * self.K
            phi = phi / (np.sum(phi, axis=1, keepdims=True) + 1e-9)
            return alpha_new, phi

        # 初始化alpha和phi
        alpha_old = np.random.random(self.K)
        phi_old = np.random.random(size=(self.K, V))
        phi_old = phi_old / np.sum(phi_old, axis=1, keepdims=True)
        for _ in range(self.epochs):
            eta, gamma = update_vi_params(alpha_old, phi_old)
            alpha_new, phi_new = update_lda_params(eta, gamma, alpha_old)
            if (np.sum(np.abs(alpha_new - alpha_old)) + np.sum(np.abs((phi_new - phi_old)))) / (
                        (V + 1) * self.K) < self.tol:
                break
            else:
                alpha_old = alpha_new.copy()
                phi_old = phi_new.copy()
        self.phi = phi_new

    def fit(self, W):
        if self.method == "gibbs":
            self._fit_gibbs(W)
        else:
            self._fit_vi_em(W)

    def transform(self, W):
        rst = []
        for w in W:
            tmp = np.zeros(shape=self.K)
            for v in w:
                try:
                    v_ = self.phi[:, v]
                except:
                    v_ = np.zeros(shape=self.K)
                tmp += v_
            if np.sum(tmp) > 0:
                tmp = tmp / np.sum(tmp)
            rst.append(tmp)
        return np.asarray(rst)
