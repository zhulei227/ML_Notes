import numpy as np
import os

os.chdir('../')
import matplotlib.pyplot as plt
from ml_models.tree import CARTRegressor
from ml_models.linear_model import LinearRegression

data = np.linspace(1, 10, num=100)
target = np.sin(data) + np.random.random(size=100)  # 添加噪声
data = data.reshape((-1, 1))

from ml_models.ensemble import RandomForestRegressor

model = RandomForestRegressor(base_estimator=[LinearRegression(), CARTRegressor()], n_estimators=10)
model.fit(data, target)

plt.scatter(data, target)
plt.plot(data, model.predict(data), color='r')
plt.show()
