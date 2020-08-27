from sklearn.datasets.samples_generator import make_blobs

X, y = make_blobs(n_samples=400, centers=4, cluster_std=0.85, random_state=0)
X = X[:, ::-1]

from ml_models.cluster import AGNES

agnes = AGNES(k=4)
agnes.fit(X)

from ml_models import utils
utils.plot_decision_function(X, y, agnes)
utils.plt.show()