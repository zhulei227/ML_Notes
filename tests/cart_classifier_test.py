import numpy as np
import os
from sklearn import model_selection

os.chdir('../')

from sklearn.datasets import make_classification
from ml_models import utils

data, target = make_classification(n_samples=100, n_features=2, n_classes=2, n_informative=1, n_redundant=0,
                                   n_repeated=0, n_clusters_per_class=1, class_sep=.3, random_state=44)
X_train, X_test, y_train, y_test = model_selection.train_test_split(data, target, test_size=0.1, random_state=0)

from ml_models.tree import CARTClassifier

tree = CARTClassifier()
tree.fit(X_train, y_train)
tree.prune(5)

# # 计算F1
from sklearn.metrics import f1_score

print(f1_score(y_test, tree.predict(X_test)))
print(np.sum(np.abs(y_test - tree.predict(X_test))))
#
utils.plot_decision_function(X_train, y_train, tree)
utils.plt.show()
