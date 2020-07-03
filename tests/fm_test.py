import numpy as np
import os

os.chdir('../')
import matplotlib.pyplot as plt
from ml_models.fm import FM

# data1 = np.linspace(1, 10, num=10000)
# data2 = np.linspace(1, 10, num=10000) + np.random.random(size=10000)
# data3 = np.linspace(10, 1, num=10000)
# target = data1 * 2 + data3 * 0.1 + data2 * 1 + 10 * data1 * data2 + np.random.random(size=10000)
# data = np.c_[data1, data2, data3]

data = np.random.random((10000, 500))
data = np.c_[np.linspace(0, 1, 10000), data]
target = data[:, 0] * 1 + data[:, 1] * 2 + 2 * data[:, 8] * data[:, 9]

model = FM(batch_size=128, lr=1e-3, epochs=10,hidden_dim=4,solver='adam')
losses = model.fit(data, target)

plt.scatter(data[:, 0], target)
plt.plot(data[:, 0], model.predict(data), color='r')
plt.show()
plt.plot(range(0, len(losses)), losses)
plt.show()
print(model.V)
print(model.w)
