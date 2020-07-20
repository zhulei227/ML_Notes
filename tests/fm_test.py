import numpy as np
import os

os.chdir('../')
import matplotlib.pyplot as plt
from ml_models.fm import FFM, FM
from ml_models.vi.linear_regression import *

data1 = np.linspace(0, 1, num=1000)
data2 = np.linspace(0, 1, num=1000) + np.random.random(size=1000)
data3 = np.linspace(1, 0, num=1000)
target = data1 * 2 + data3 * 0.1 + data2 * 1 + 100 * data1 * data2 + np.random.random(size=1000)
data = np.c_[data1, data2, data3]

# X = np.linspace(0, 100, 100)
# X = np.c_[X, np.ones(100)]
# w = np.asarray([3, 2])
# Y = X.dot(w)
# X = X.astype('float')
# Y = Y.astype('float')
# X[:, 0] += np.random.normal(size=(X[:, 0].shape)) * 3  # 添加噪声
# Y = Y.reshape(100, 1)
# # 加噪声
# X = np.concatenate([X, np.asanyarray([[100, 1], [101, 1], [102, 1], [103, 1], [104, 1]])])
# Y = np.concatenate([Y, np.asanyarray([[3000], [3300], [3600], [3800], [3900]])])
#
# target = Y.reshape(105, 1)
# data = X[:, 0].reshape((-1, 1))

# data = np.random.random((50000, 25))
# data = np.c_[np.linspace(0, 1, 50000), data]
# target = data[:, 0] * 1 + data[:, 1] * 2 + 2 * data[:, 8] * data[:, 9]
#
# from ml_models.wrapper_models import DataBinWrapper
#
# binwrapper = DataBinWrapper()
# binwrapper.fit(data)
# new_data = binwrapper.transform(data)
#
# from sklearn.preprocessing import OneHotEncoder
#
# one_hot_encoder = OneHotEncoder()
# new_data = one_hot_encoder.fit_transform(new_data).toarray()
# print(new_data.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.4, random_state=0)

# model = FM(batch_size=10, epochs=5, objective='poisson')
# model = FM(batch_size=10, epochs=5)
# train_losses, eval_losses = model.fit(X_train, y_train, eval_set=(X_test, y_test))
model = LinearRegression(normalized=False)
model.fit(X_train, y_train)

plt.scatter(data[:, 0], target)
plt.plot(data[:, 0], model.predict(data), color='r')
plt.show()
# plt.plot(range(0, len(train_losses)), train_losses, label='train loss')
# plt.plot(range(0, len(eval_losses)), eval_losses, label='eval loss')
# plt.legend()
# plt.show()
# print(model.V)
print(model.w)
