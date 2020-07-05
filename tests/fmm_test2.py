import numpy as np
import os

os.chdir('../')
import matplotlib.pyplot as plt
from ml_models.fm import FFM

# data1 = np.linspace(1, 10, num=100)
# data2 = np.linspace(1, 10, num=100) + np.random.random(size=100)
# data3 = np.linspace(10, 1, num=100)
# target = data1 * 2 + data3 * 0.1 + data2 * 1 + 10 * data1 * data3 + np.random.random(size=100)
# data = np.c_[data1, data2, data3]

# data = np.random.random((50000, 25))
# data = np.c_[np.linspace(0, 1, 50000), data]
# target = data[:, 0] * 1 + data[:, 1] * 2 + 2 * data[:, 8] * data[:, 9]

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

# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.4, random_state=0)
#
# model = FFM(batch_size=1, epochs=20, solver='adam',objective='tweedie')
# train_losses, eval_losses = model.fit(X_train, y_train, eval_set=(X_test, y_test), show_log=True)
#
# plt.scatter(data[:, 0], target)
# plt.plot(data[:, 0], model.predict(data), color='r')
# plt.show()
# plt.plot(range(0, len(train_losses)), train_losses, label='train loss')
# plt.plot(range(0, len(eval_losses)), eval_losses, label='eval loss')
# plt.legend()
# plt.show()
# print(model.V)
# print(model.w)

"""
二分类
"""
from ml_models import utils

from sklearn.datasets import make_classification

data, target = make_classification(n_samples=200, n_features=2, n_classes=2, n_informative=1, n_redundant=0,
                                   n_repeated=0, n_clusters_per_class=1)

# ffm=FFM(batch_size=1, epochs=20, solver='adam',objective='logistic')
# ffm.fit(data,target,show_log=True)
# utils.plot_decision_function(data,target,ffm)
# utils.plt.show()


"""
多分类
"""
from sklearn.datasets.samples_generator import make_blobs

X, y = make_blobs(n_samples=400, centers=4, cluster_std=0.85, random_state=0)
X = X[:, ::-1]
from ml_models.wrapper_models import *
#
ffm = FFM(epochs=10, solver='adam', objective='logistic')
ovo = MultiClassWrapper(ffm, mode='ovo')
ovo.fit(X, y)
utils.plot_decision_function(X, y, ovo)
utils.plt.show()