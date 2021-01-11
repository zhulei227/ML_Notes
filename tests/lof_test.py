import os

os.chdir('../')
import matplotlib.pyplot as plt
import numpy as np

X = np.c_[np.random.random(size=(100, 2)).T, np.random.random(size=(200, 2)).T * 5].T

from ml_models.outlier_detect import LOF

lof = LOF(n_neighbors=10)
score = lof.fit_transform(X)
import numpy as np

thresh = np.percentile(score, 95)
plt.scatter(x=X[:, 0], y=X[:, 1], c=score > thresh)
plt.show()
