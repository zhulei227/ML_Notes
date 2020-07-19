"""
线性回归的bayes估计
"""
import numpy as np
import matplotlib.pyplot as plt

"""
对应15_05的实现
"""


class LinearRegression1(object):
    def __init__(self, basis_func=None, alpha=1, beta=1):
        """
        :param basis_func: list,基函数列表，包括rbf,sigmoid,poly_{num},linear，默认None为linear，其中poly_{num}中的{num}表示多项式的最高阶数
        :param alpha: alpha/beta表示理解为L2正则化项的大小，默认为1
        :param beta: 噪声，默认为1
        """
        if basis_func is None:
            self.basis_func = ['linear']
        else:
            self.basis_func = basis_func
        self.alpha = alpha
        self.beta = beta
        # 特征均值、标准差
        self.feature_mean = None
        self.feature_std = None
        # 训练参数
        self.w = None

    def _map_basis(self, X):
        """
        将X进行基函数映射
        :param X:
        :return:
        """
        x_list = []
        for basis_func in self.basis_func:
            if basis_func == "linear":
                x_list.append(X)
            elif basis_func == "rbf":
                x_list.append(np.exp(-0.5 * X * X))
            elif basis_func == "sigmoid":
                x_list.append(1 / (1 + np.exp(-1 * X)))
            elif basis_func.startswith("poly"):
                p = int(basis_func.split("_")[1])
                for pow in range(1, p + 1):
                    x_list.append(np.power(X, pow))
        return np.concatenate(x_list, axis=1)

    def fit(self, X, y):
        self.feature_mean = np.mean(X, axis=0)
        self.feature_std = np.std(X, axis=0) + 1e-8
        X_ = (X - self.feature_mean) / self.feature_std
        X_ = self._map_basis(X_)
        X_ = np.c_[np.ones(X_.shape[0]), X_]
        self.w = self.beta * (
            np.linalg.inv(self.alpha * np.eye(X_.shape[1]) + self.beta * X_.T @ X_)) @ X_.T @ y.reshape((-1, 1))

    def predict(self, X):
        X_ = (X - self.feature_mean) / self.feature_std
        X_ = self._map_basis(X_)
        X_ = np.c_[np.ones(X_.shape[0]), X_]
        return (self.w.T @ X_.T).reshape(-1)

    def plot_fit_boundary(self, x, y):
        """
        绘制拟合结果
        :param x:
        :param y:
        :return:
        """
        plt.scatter(x[:, 0], y)
        plt.plot(x[:, 0], self.predict(x), 'r')


"""
对应15_06的实现
"""


class LinearRegression(object):
    def __init__(self, basis_func=None, tol=1e-7, epochs=100, normalized=True):
        """
        :param basis_func: list,基函数列表，包括rbf,sigmoid,poly_{num},linear，fm,默认None为linear，其中poly_{num}中的{num}表示多项式的最高阶数,fm表示构建交叉因子
        :param tol:  两次迭代参数平均绝对值变化小于tol则停止
        :param epochs: 默认迭代次数
        :param normalized:是否归一化
        """
        if basis_func is None:
            self.basis_func = ['linear']
        else:
            self.basis_func = basis_func
        self.tol = tol
        self.epochs = epochs
        self.normalized = normalized
        # 特征均值、标准差
        self.feature_mean = None
        self.feature_std = None
        # 训练参数
        self.w = None

    def _map_basis(self, X):
        """
        将X进行基函数映射
        :param X:
        :return:
        """
        n_sample, n_feature = X.shape
        x_list = []
        for basis_func in self.basis_func:
            if basis_func == "linear":
                x_list.append(X)
            elif basis_func == "rbf":
                x_list.append(np.exp(-0.5 * X * X))
            elif basis_func == "sigmoid":
                x_list.append(1 / (1 + np.exp(-1 * X)))
            elif basis_func.startswith("poly"):
                p = int(basis_func.split("_")[1])
                for pow in range(1, p + 1):
                    x_list.append(np.power(X, pow))
            elif basis_func == 'fm':
                X_fm = np.zeros(shape=(n_sample, int(n_feature * (n_feature - 1) / 2)))
                c = 0
                for i in range(0, n_feature - 1):
                    for j in range(i + 1, n_feature):
                        X_fm[:, c] = X[:, i] * X[:, j]
                        c += 1
                x_list.append(X_fm)
        return np.concatenate(x_list, axis=1)

    def fit(self, X, y):
        if self.normalized:
            self.feature_mean = np.mean(X, axis=0)
            self.feature_std = np.std(X, axis=0) + 1e-8
            X_ = (X - self.feature_mean) / self.feature_std
        else:
            X_ = X
        X_ = self._map_basis(X_)
        X_ = np.c_[np.ones(X_.shape[0]), X_]
        n_sample, n_feature = X_.shape
        alpha = 1
        beta = 1
        current_w = None
        for _ in range(0, self.epochs):
            A = alpha * np.eye(n_feature) + beta * X_.T @ X_
            self.w = beta * np.linalg.inv(A) @ X_.T @ y.reshape((-1, 1))  # 即m_N
            if current_w is not None and np.mean(np.abs(current_w - self.w)) < self.tol:
                break
            current_w = self.w
            # 更新alpha,beta
            if n_sample // n_feature >= 100:
                # 利用prml中的公式3.98与3.99进行简化计算，避免求特征值的开销
                alpha = n_feature / np.dot(self.w.reshape(-1), self.w.reshape(-1))
                beta = n_sample / np.sum(np.power(y.reshape(-1) - self.predict(X).reshape(-1), 2))
            else:
                gamma = 0.0
                for lamb in np.linalg.eig(beta * X_.T @ X_)[0]:
                    gamma += lamb / (lamb + alpha)
                alpha = gamma.real / np.dot(self.w.reshape(-1), self.w.reshape(-1))
                beta = 1.0 / (
                    1.0 / (n_sample - gamma) * np.sum(np.power(y.reshape(-1) - self.predict(X).reshape(-1), 2)))

    def predict(self, X):
        if self.normalized:
            X_ = (X - self.feature_mean) / self.feature_std
        else:
            X_ = X
        X_ = self._map_basis(X_)
        X_ = np.c_[np.ones(X_.shape[0]), X_]
        return (self.w.T @ X_.T).reshape(-1)

    def plot_fit_boundary(self, x, y):
        """
        绘制拟合结果
        :param x:
        :param y:
        :return:
        """
        plt.scatter(x[:, 0], y)
        plt.plot(x[:, 0], self.predict(x), 'r')