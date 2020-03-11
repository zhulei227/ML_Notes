import numpy as np
import os

os.chdir('../')

from sklearn.datasets import make_classification
from ml_models import utils

# data, target = make_classification(n_samples=100, n_features=2, n_classes=2, n_informative=1, n_redundant=0,
#                                    n_repeated=0, n_clusters_per_class=1, class_sep=3.0)
from sklearn import datasets

data, target = datasets.make_moons(noise=0.01)

from ml_models.svm import SVC

svm = SVC(C=3.0, kernel='rbf',gamma=0.1, epochs=10, tol=0.2)
# svm = SVC(tol=0.01)
svm.fit(data, target, show_train_process=True)

# 计算F1
from sklearn.metrics import f1_score

print(f1_score(target, svm.predict(data)))
print(np.sum(np.abs(target - svm.predict(data))))

utils.plt.close()
utils.plot_decision_function(data, target, svm, svm.support_vectors)
utils.plt.show()
print('support vector', svm.support_vectors)
