from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_moons(noise=0.01)

from ml_models.cluster import DBSCAN

dbscan = DBSCAN(eps=0.2, min_sample=3)
lable = dbscan.fit_predict(X)
print(lable)
plt.scatter(X[:, 0], X[:, 1], c=lable)
plt.show()