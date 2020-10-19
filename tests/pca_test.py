import numpy as np
import matplotlib.pyplot as plt

# 造伪样本
X = np.linspace(0, 10, 100)
Y = 3 * X + 2
X += np.random.normal(size=(X.shape)) * 0.3  # 添加噪声
data = np.c_[X, Y]

from ml_models.decomposition import PCA

pca = PCA()
pca.fit(data)
new_data = pca.transform(data)

plt.scatter(new_data[:, 0], new_data[:, 1])
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.show()