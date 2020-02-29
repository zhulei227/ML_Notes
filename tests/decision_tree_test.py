import numpy as np
import os

os.chdir('../')

from sklearn.datasets import make_classification
from ml_models import utils

data, target = make_classification(n_samples=100, n_features=2, n_classes=2, n_informative=1, n_redundant=0,
                                   n_repeated=0, n_clusters_per_class=1, class_sep=.3)

from ml_models.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_bins=20)
tree.fit(data, target)
tree.prune(alpha=1)

# # 计算F1
from sklearn.metrics import f1_score

print(f1_score(target, tree.predict(data)))
print(np.sum(np.abs(target - tree.predict(data))))
#
utils.plot_decision_function(data, target, tree)
utils.plt.show()
