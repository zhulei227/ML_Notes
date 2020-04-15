from sklearn.datasets.samples_generator import make_blobs
from ml_models import utils

X, y = make_blobs(n_samples=400, centers=4, cluster_std=0.85, random_state=0)
X = X[:, ::-1]

from ml_models.em import GMMClassifier

gmm = GMMClassifier(n_iter=100)
gmm.fit(X, y)
print(gmm.predict(X))
utils.plot_decision_function(X, y, gmm)
utils.plt.show()
