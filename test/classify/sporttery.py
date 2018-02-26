# X = [[0], [1], [2], [3]]
# y = [0, 2, 6, 10]
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv('./data/data2.csv')
y = np.array(data[['result']])
x = np.array(data[['h', 'd', 'a', 'fixedodds']])

x = preprocessing.scale(x)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
# y_predict = knn.predict(X_test)
total = 0
right = 0
for index, value in enumerate(X_test):
    total += 1
    y_ = knn.predict([value])
    if y_ == y_test[index]:
        right += 1
    print(X_test[index], y_, y_test[index])

print("测试%d场,猜对率:%.2f" % (total,  right/total*100))
