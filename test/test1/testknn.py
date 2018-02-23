# X = [[0], [1], [2], [3]]
# y = [0, 2, 6, 10]
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

X = np.array(range(1000))
noise = np.random.uniform(-0.2, 0.2, 1000)
y = np.sin(np.pi * X / 100) + X / 200. + noise

T = (np.array(range(700, 1000))).reshape(-1, 1)
neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X.reshape(-1, 1), y)
y_ = neigh.predict(T)
plt.plot(X, y, 'r', label='data')
plt.plot(T, y_, c='g', label='prediction')
plt.axis('tight')
plt.legend()
plt.show()
