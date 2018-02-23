import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.cross_validation import cross_val_predict
# from sklearn.kernel_ridge import KernelRidge
from sklearn import neighbors
import matplotlib.pyplot as plt
""" generat data : y = sin(pi/100 *  x) + x / 200 noise """


x = np.array(range(1000))
noise = np.random.uniform(-0.2, 0.2, 1000)
y = np.sin(np.pi * x / 100) + x / 200. + noise
trainPST = int(0.7*len(x))
X_train = x[:1000]
X_test = x[700:]
y_train = y[:1000]
y_test = y[700:]

# print(X_train,)
plt.plot(X_train, y_train)
plt.plot(X_test, y_test)

# lr = LinearRegression()
# lr.fit(X_train.reshape(-1, 1), y_train)
# # print(knn.predict(X_test))
# y_predict = lr.predict(X_test.reshape(-1, 1))

# kr = KernelRidge()
# kr.fit(X_train.reshape(-1, 1), y_train)
# y_predict = kr.predict(X_test.reshape(-1, 1))
n_neighbors = 1
weights = 'distance'  # 'uniform', 'distance'
knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
knn.fit(X_train.reshape(-1, 1), y_train)
y_predict = knn.predict(X_test.reshape(-1, 1))
print(y_predict)
plt.plot(X_test, y_predict)

plt.legend(('train', 'test', 'predict'))
plt.title('KNeighborsRegressor')
plt.savefig('./test/test1/KNeighborsRegressor')
plt.show()
