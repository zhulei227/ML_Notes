from sklearn.datasets.samples_generator import make_blobs
from ml_models import utils

X, y = make_blobs(n_samples=400, centers=4, cluster_std=0.85, random_state=0)
X = X[:, ::-1]

from ml_models.pgm import NaiveBayesCluster

nb = NaiveBayesCluster(n_iter=500, tol=1e-5, n_components=4, max_bins=20, verbose=False)
nb.fit(X)
print(nb.predict(X))
utils.plot_decision_function(X, y, nb)
utils.plt.show()
