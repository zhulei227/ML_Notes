import numpy as np
import os

os.chdir('../')
import matplotlib.pyplot as plt

data = np.linspace(1, 10, num=100)
target = np.sin(data) + np.random.random(size=100) + 1  # 添加噪声
data = data.reshape((-1, 1))

from ml_models.ensemble import XGBoostRegressor

model = XGBoostRegressor(loss='tweedie', p=1.5)
model.fit(data, target)

plt.scatter(data, target)
plt.plot(data, model.predict(data), color='r')
plt.show()
