from sklearn.datasets import make_classification, make_moons
import numpy as np
from ml_models import utils
from ml_models.svm import SVC

X, y = make_classification(n_samples=500, n_features=2,
                           n_informative=2, n_redundant=0,
                           n_repeated=0, n_classes=2,
                           n_clusters_per_class=1, weights=[0.05, 0.95],
                           class_sep=3, flip_y=0.05, random_state=0)
# X, y = make_moons(noise=0.01)

weights = np.where(y == 0, 50, 1)
svc_with_sample_weight = SVC(kernel='rbf', gamma=2.0)
svc_with_sample_weight.fit(X, y, sample_weight=weights, show_train_process=True)
utils.plot_decision_function(X=X, y=y, clf=svc_with_sample_weight)
