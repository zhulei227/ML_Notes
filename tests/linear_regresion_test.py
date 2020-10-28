import numpy as np

# 造伪样本
X = np.linspace(0, 100, 100)
X = np.c_[X, np.ones(100)]
w = np.asarray([3, 2])
Y = X.dot(w)
X = X.astype('float')
Y = Y.astype('float')
X[:, 0] += np.random.normal(size=(X[:, 0].shape)) * 3  # 添加噪声

Y = Y.reshape(100, 1)

from ml_models.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 测试
lr = LinearRegression(solver='sgd')
lr.fit(X[:, :-1], Y)
predict = lr.predict(X[:, :-1])
# 查看w
print('w', lr.get_params())
# 查看标准差
print(np.std(Y - predict))

lr.plot_fit_boundary(X[:, :-1], Y)
plt.show()
