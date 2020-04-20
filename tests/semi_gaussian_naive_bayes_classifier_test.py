from sklearn.datasets.samples_generator import make_blobs
from ml_models import utils

X, y = make_blobs(n_samples=400, centers=4, cluster_std=0.85, random_state=0)
X = X[:, ::-1]

from ml_models.pgm import SemiGaussianNBClassifier

nb = SemiGaussianNBClassifier(link_rulers=[(0, 1)])
nb.fit(X, y)
print(nb.predict_proba(X).shape)
utils.plot_decision_function(X, y, nb)
utils.plt.show()
