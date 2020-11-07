from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt

X, y = make_blobs(n_samples=400, centers=4, cluster_std=0.85, random_state=0)
X = X[:, ::-1]

from ml_models.cluster import Spectral

spectral = Spectral(n_clusters=4)

plt.scatter(X[:, 0], X[:, 1], c=spectral.fit_predict(X))
plt.show()