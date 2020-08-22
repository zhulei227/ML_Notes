"""
隐狄利克雷分布的代码实现，包括Gibbs采样和变分EM算法
"""
import numpy as np


class LDA(object):
    def __init__(self, alpha=None, beta=None, K=10, tol=1e-3, epochs=100):
        """
        :param alpha: 主题分布的共轭狄利克雷分布的超参数
        :param beta: 单词分布的共轭狄利克雷分布的超参数
        :param K: 主题数量
        :param tol:容忍度，允许tol的隐变量差异
        :param epochs:最大迭代次数
        """
        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.tol = tol
        self.epochs = epochs
        self.theta = None  # 文本-主题矩阵
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

    def fit(self, W):
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

        # 计算参数theta和phi
        self.theta = N_M_K / np.sum(N_M_K, axis=1, keepdims=True)
        self.phi = N_K_V / np.sum(N_K_V, axis=1, keepdims=True)

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