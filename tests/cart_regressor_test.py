import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir('../')

from sklearn.datasets import make_regression

# data, target = make_regression(n_samples=100, n_features=1, random_state=44, bias=0.5, noise=2)
data = np.linspace(1, 10, num=100)
target = np.sin(data) + np.random.random(size=100)
data = data.reshape((-1, 1))

# indices = np.argsort(target)
#
# data = data[indices]
# target = target[indices]

from ml_models.tree import CARTRegressor

tree = CARTRegressor(max_bins=50)
tree.fit(data, target)
tree.prune(10000)

plt.scatter(data, target)
plt.plot(data, tree.predict(data), color='r')
plt.show()
