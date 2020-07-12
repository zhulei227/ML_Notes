"""
线性回归的vi实现
"""
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression(object):
    def __init__(self, basis_func=None, beta=1e-12, tol=1e-5, epochs=100):
        """
        :param basis_func: list,基函数列表，包括rbf,sigmoid,poly_{num},linear，默认None为linear，其中poly_{num}中的{num}表示多项式的最高阶数
        :param beta: 生成t标签的高斯噪声，这里可以设置低一点
        :param tol:  两次迭代参数平均绝对值变化小于tol则停止
        :param epochs: 默认迭代次数
        """
        if basis_func is None:
            self.basis_func = ['linear']
        else:
            self.basis_func = basis_func
        self.beta = beta
        self.tol = tol
        self.epochs = epochs
        # 特征均值、标准差
        self.feature_mean = None
        self.feature_std = None
        # 训练参数，也就是m_N
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
        n_sample, n_feature = X_.shape

        E_alpha = self.beta  # 初始就和beta设置一样，让它自动去调节（这里设置任意大于0的值都是可以的）
        current_w = None
        for _ in range(0, self.epochs):
            S_N = np.linalg.inv(E_alpha * np.eye(n_feature) + self.beta * X_.T @ X_)
            self.w = self.beta * S_N @ X_.T @ y.reshape((-1, 1))  # 即m_N
            if current_w is not None and np.mean(np.abs(current_w - self.w)) < self.tol:
                break
            current_w = self.w
            E_w = (self.w.T @ self.w)[0][0] + np.trace(S_N)
            E_alpha = (n_feature - 1) / E_w  # 这里假设a_0,b_0都为0

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
