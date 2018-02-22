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

### 岭回归

* [RidgeRegression](./regression/RidgeRegression.py)
  RidgeCV 实现了带缺省 alpha 参数的交叉验证的岭回归模型.这个对象和 GridSearchCV 除了它默认为广义交叉验证(GCV),其他工作方式一样。
* [Ridge2](./regression/Ridge2.py)

### Lasso

* []()
  Lasso 是一种估计稀疏线性模型的方法.由于它倾向具有少量参数值的情况，对于给定解决方案是相关情况下，有效的减少了变量数量。 因此，Lasso 及其变种是压缩感知(压缩采样)的基础。Lasso 类实现使用了坐标下降法(一种非梯度优化算法) 来拟合系数
