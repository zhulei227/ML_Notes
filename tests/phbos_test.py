import os

os.chdir('../')
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

X, _ = make_classification(n_samples=5000, n_features=2,
                           n_informative=2, n_redundant=0,
                           n_repeated=0, n_classes=3,
                           n_clusters_per_class=1, weights=[0.01, 0.05, 0.94],
                           class_sep=0.8, random_state=0)

from ml_models.outlier_detect import pHbos

hbos = pHbos()
y = hbos.fit_transform(X)
plt.scatter(x=X[:, 0], y=X[:, 1], c=y)
plt.show()