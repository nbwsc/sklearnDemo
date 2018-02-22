# sklearnDemo

机器学习 scikit-learn

## 分类/回归 regression

* [plot_cv_predict](./regression/plot_cv_predict.py)
  use cross_val_predict to visualize prediction errors.
* [Isotonic regression](./regression/plot_isotonic_regression.py)
  保序回归(Isotonic Regression)对生成的数据进行保序回归的一个实例.保序回归能在训练数据上发现一个非递减逼近函数的同时最小化均方误差。这样的模型的好处是，它不用假设任何形式的目标函数，(如线性)。为了比较，这里用一个线性回归作为参照。
* [plot_multioutput_face_completion](./regression/plot_multioutput_face_completion.py)

## 广义线性模型

### 普通最小二乘法

* [LinearRegression](./regression/LinearRegression.py)
  这种方法通过对矩阵 X 奇异值分解（SVD）的方式来计算最小二乘的解。如果 X 是一个(n, p)大小的矩阵,那么代价为 O(np^2),假设 n \geq p.

###  岭回归

    \underset{w}{min\,} {{|| X w - y||\_2}^2 + \alpha {||w||\_2}^2}

* [RidgeRegression](./regression/RidgeRegression.py)

###
