import numpy as np
import os
os.chdir('../')

from sklearn import model_selection

from sklearn.datasets import make_classification
from ml_models import utils
from ml_models.linear_model import LogisticRegression
from ml_models.tree import CARTClassifier
from ml_models.svm import SVC


data, target = make_classification(n_samples=100, n_features=2, n_classes=2, n_informative=1, n_redundant=0,
                                   n_repeated=0, n_clusters_per_class=1, class_sep=0.5)
X_train, X_test, y_train, y_test = model_selection.train_test_split(data, target, test_size=0.1)

from ml_models.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(base_estimator=CARTClassifier(),n_estimators=10)
classifier.fit(X_train, y_train)
# # 计算F1
from sklearn.metrics import f1_score
print(f1_score(y_test, classifier.predict(X_test)))
print(np.sum(np.abs(y_test - classifier.predict(X_test))))
#
utils.plot_decision_function(X_train, y_train, classifier)
utils.plt.show()
