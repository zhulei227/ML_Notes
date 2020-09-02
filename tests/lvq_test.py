from sklearn.datasets.samples_generator import make_blobs
import numpy as np

X, y = make_blobs(n_samples=400, centers=4, cluster_std=0.85, random_state=0)
X = X[:, ::-1]

# 将0,2类归为0类
y = np.where(y == 2, 0, y)
# 将1,3类归为1类
y = np.where(y == 3, 1, y)

from ml_models.cluster import LVQ

kmeans = LVQ(class_label=[0, 0, 1, 1])
kmeans.fit(X, y)

from ml_models import utils

utils.plot_decision_function(X, y, kmeans)
utils.plt.show()
