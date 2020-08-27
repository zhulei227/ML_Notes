"""
DBSCAN密度聚类的代码实现
"""
import numpy as np
from queue import Queue


class DBSCAN(object):
    def __init__(self, eps=0.5, min_sample=3, dist_method=None):
        """
        :param eps:epsilon领域半径
        :param min_sample: 核心对象的epsilon领域半径内的最少样本量
        :param dist_method:样本距离度量，默认欧氏距离
        """
        self.eps = eps
        self.min_sample = min_sample
        self.dist_method = dist_method
        if self.dist_method is None:
            self.dist_method = lambda x, y: np.sqrt(np.sum(np.power(x - y, 2)))
        self.label_ = None  # 记录样本标签，-1表示异常点

    def fit(self, X):
        rows = X.shape[0]
        self.label_ = np.ones(rows) * -1
        M = np.zeros(shape=(rows, rows))
        # 计算样本间的距离
        for i in range(rows - 1):
            for j in range(i, rows):
                M[i, j] = self.dist_method(X[i], X[j])
                M[j, i] = M[i, j]
        # 记录核心矩阵
        H = set()
        for i in range(0, rows):
            if np.sum(M[i] <= self.eps) >= self.min_sample:
                H.add(i)
        # 初始化聚类簇数
        k = 0
        # 初始化未访问样本集合
        W = set(range(rows))
        while len(H) > 0:
            # 记录当前未访问样本集合
            W_old = W.copy()
            # 随机选择一个核心对象
            o = np.random.choice(list(H))
            # 初始化队列
            Q = Queue()
            Q.put(o)
            # 未访问队列中去掉核心对象o
            W = W - set([o])
            while not Q.empty():
                # 取出首个样本
                q = Q.get()
                # 判断是否为核心对象
                if q in H:
                    # 获取领域内样本与未访问样本的交集
                    delta = set(np.argwhere(M[q] <= self.eps).reshape(-1).tolist()) & W
                    # 将其放入队列
                    for d in delta:
                        Q.put(d)
                    # 从未访问集合中去掉
                    W = W - delta
            # 获取聚类簇idx
            C_k = W_old - W
            k_idx = list(C_k)
            self.label_[k_idx] = k
            k += 1
            # 去掉在当前簇中的核心对象
            H = H - C_k

    def fit_predict(self, X):
        self.fit(X)
        return self.label_
