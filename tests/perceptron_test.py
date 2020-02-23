import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

data, target = make_classification(n_samples=200, n_features=2, n_classes=2, n_informative=1, n_redundant=0,
                                   n_repeated=0, n_clusters_per_class=1)
from ml_models.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(data, target)
perceptron.plot_decision_boundary(data, target)
