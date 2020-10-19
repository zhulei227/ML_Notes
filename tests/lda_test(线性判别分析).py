import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

data, target = make_classification(n_samples=50, n_features=2,
                                   n_informative=2, n_redundant=0,
                                   n_repeated=0, n_classes=2,
                                   n_clusters_per_class=1,
                                   class_sep=3, random_state=32)
plt.scatter(data[:, 0], data[:, 1], c=target, s=50)
plt.show()

# 开始转换
from ml_models.decomposition import LDA

lda = LDA()
lda.fit(data, target)
new_data = lda.transform(data)
plt.scatter(new_data[:, 0], new_data[:, 1], c=target, s=50)
plt.show()
