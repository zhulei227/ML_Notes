from matplotlib import pyplot as plt
import numpy as np

n = 200
r = np.linspace(0, 1, n)
l = np.linspace(0, 1, n)

t = (3 * np.pi) / 2 * (1 + 2 * r)
x = t * np.cos(t)
y = 10 * l
z = t * np.sin(t)

data = np.c_[x, y, z]

from ml_models.decomposition import Isomap

isomap = Isomap(n_components=2, epsilon=15)
new_data = isomap.fit_transform(data)
plt.scatter(new_data[:, 0], new_data[:, 1])
plt.show()