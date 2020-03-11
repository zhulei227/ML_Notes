import numpy as np
import os

os.chdir('../')
from ml_models import utils
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

data, target = make_classification(n_samples=200, n_features=2, n_classes=2, n_informative=1, n_redundant=0,
                                   n_repeated=0, n_clusters_per_class=1)

from ml_models.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(data, target)

# 计算F1
from sklearn.metrics import f1_score

print(f1_score(target, lr.predict(data)))
print(len(data))
print(np.sum(np.abs(target - lr.predict(data))))
lr.plot_decision_boundary(data, target)
lr.plot_losses()
