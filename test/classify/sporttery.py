# X = [[0], [1], [2], [3]]
# y = [0, 2, 6, 10]
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

data = pd.read_csv('./data/data2.csv')
y = np.array(data[['result']]).flatten()
x = np.array(
    data[['h', 'd', 'a', 'fixedodds', 'win_per', 'draw_per', 'lose_per']])

x = preprocessing.scale(x)
# X_train, X_test, y_train, y_test = train_test_split(
# x, y, test_size=0.1)
train_size = 0.9
sliceLength = int(train_size*len(x))
X_train = x[:sliceLength]
y_train = y[:sliceLength]

X_test = x[sliceLength:]
y_test = y[sliceLength:]

"""knn:44.22"""
# from sklearn.neighbors import KNeighborsClassifier

# clf = KNeighborsClassifier()
# clf.fit(X_train, y_train)


"""svm:50.44"""
from sklearn.svm import SVC, LinearSVC
clf = LinearSVC()
# clf = SVC()
clf.fit(X_train, y_train)

"""naive_bayes"""
# from sklearn.naive_bayes import GaussianNB, BernoulliNB
# clf = BernoulliNB()
# # clf = GaussianNB()
# clf.fit(X_train, y_train)

total = 0
right = 0
gainrate = 0
for index, value in enumerate(X_test):
    total += 1
    gainrate -= 1
    y_ = clf.predict([value])
    if y_[0] == y_test[index]:
        right += 1
        tp = data[['h', 'd', 'a'][y_[0]]][index+sliceLength]
        # print(tp)
        gainrate += tp
    print(index, right/total*100, y_[0], y[index+sliceLength])

print("测试%d场,猜对率:%.2f,盈利情况:%.2f" % (total,  right/total*100, gainrate))
