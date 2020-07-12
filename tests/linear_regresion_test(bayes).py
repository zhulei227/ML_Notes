import numpy as np
import matplotlib.pyplot as plt

# 造伪样本
# 造伪样本
X = np.linspace(0, 100, 100)
X = np.c_[X, np.ones(100)]
w = np.asarray([3, 2])
Y = X.dot(w)
X = X.astype('float')
Y = Y.astype('float')
X[:, 0] += np.random.normal(size=(X[:, 0].shape)) * 3  # 添加噪声
Y = Y.reshape(100, 1)
X = np.concatenate([X, np.asanyarray([[100, 1], [101, 1], [102, 1], [103, 1], [104, 1]])])
Y = np.concatenate([Y, np.asanyarray([[3000], [3300], [3600], [3800], [3900]])])

from ml_models.bayes import LinearRegression

# 测试
lr = LinearRegression(beta=1e-8,alpha=1e-8)
lr.fit(X[:, :-1], Y)
predict = lr.predict(X[:, :-1])
# 查看标准差
print(np.std(Y - predict))
print(lr.w)

lr.plot_fit_boundary(X[:, :-1], Y)
plt.show()
