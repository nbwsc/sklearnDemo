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

* [Lasso and Elastic Net](./regression/Lasso.py)

    Lasso 是一种估计稀疏线性模型的方法.由于它倾向具有少量参数值的情况，对于给定解决方案是相关情况下，有效的减少了变量数量。 因此，Lasso 及其变种是压缩感知(压缩采样)的基础。Lasso 类实现使用了坐标下降法(一种非梯度优化算法) 来拟合系数

### 弹性网络 ElasticNet

* [Elastic Net](./regression/ElasticNet.py)

    ElasticNet 是一种使用 L1 和 L2 先验作为正则化矩阵的线性回归模型.这种组合用于只有很少的权重非零的稀疏模型，比如:class:Lasso, 但是又能保持:class:Ridge 的正则化属性.我们可以使用 l1_ratio 参数来调节 L1 和 L2 的凸组合(一类特殊的线性组合)。当多个特征和另一个特征相关的时候弹性网络非常有用。Lasso 倾向于随机选择其中一个，而弹性网络更倾向于选择两个.
    在实践中，Lasso 和 Ridge 之间权衡的一个优势是它允许在循环过程（Under rotate）中继承 Ridge 的稳定性.

### 最小角回归 Least Angle Regression

    最小角回归是针对高维数据的回归算法，由Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani开发。

* LARS 的优势如下:

    当 p >> n 时计算是非常高效的。（比如当维数远大于点数）它和前向选择计算速度差不多一样块，并且和普通最小二乘复杂度一样。它生成一个完整的分段线性的解的路径，这对于交叉验证或者类似的尝试来调整模型是有效的。如果两个变量的相应总是相同，那么它们的系数应该有近似相同的增长速率。因此这算法和直觉判断一样，并且增长总是稳定的。这个算法对于其他评估模型来说很容易被修改来产生解，和 Lasso 差不多。（待修改）
    It is easily modified to produce solutions for other estimators, like the Lasso.

* LARS 方法的缺点包括：

    因为 LARS 是基于剩余误差多次迭代拟合,所以对噪声的影响比较敏感。这个问题在 Efron et al. (2004) Annals of Statistics article 这篇文章中讨论部分详细谈论了。
    LARS 模型可以使用 estimator Lars ，或者底层实现 lars_path 。

### LARS Lasso

* [LARS Lasso](./regression/LARS_Lasso.py)

    LARS 的优势如下:
    当 p >> n 时计算是非常高效的。（比如当维数远大于点数）它和前向选择计算速度差不多一样块，并且和普通最小二乘复杂度一样。它生成一个完整的分段线性的解的路径，这对于交叉验证或者类似的尝试来调整模型是有效的。如果两个变量的相应总是相同，那么它们的系数应该有近似相同的增长速率。因此这算法和直觉判断一样，并且增长总是稳定的。这个算法对于其他评估模型来说很容易被修改来产生解，和 Lasso 差不多。（待修改）
    It is easily modified to produce solutions for other estimators, like the Lasso.
    LARS 方法的缺点包括：

    因为 LARS 是基于剩余误差多次迭代拟合,所以对噪声的影响比较敏感。这个问题在 Efron et al. (2004) Annals of Statistics article 这篇文章中讨论部分详细谈论了。

### 正交匹配跟踪 Orthogonal Matching Pursuit (OMP)

* [OMP](./regression/omp.py)Using orthogonal matching pursuit for recovering a sparse signal from a noisy measurement encoded with a dictionary

    OrthogonalMatchingPursuit and orthogonal_mp 实现了一个用来逼近在非零系数的个数上加约束的线性模型的拟合的 OMP 算法(比如 L 0 pseudo-norm)

    和 Least Angle Regression 最小角回归 一样，作为一个前向特征选择方法，OMP 可以用一个固定非零的数来逼近最优的解向量:

### 贝叶斯回归 Bayesian Regression

* 贝叶斯回归的优势：

    * 根据数据调节参数
    * 在估计过程中包含正则化参数

* 贝叶斯回归劣势:

    模型的推理比较耗时

- [贝叶斯岭回归 BayesianRidge](./regression/bayesian_ridge.py)

    由于贝叶斯框架，权重的发现同 普通最小二乘法 略有不同。然而 Bayesian Ridge Regression 对于病态问题更具有鲁棒性。

- [Automatic Relevance Determination - ARD](./regression/ard.py)`ARDRegression` 和 `Bayesian Ridge Regression`非常相似，但是主要针对稀疏权重 w 。 ARDRegression 提出一个不同于 w 的先验，通过弱化高斯分布为球形的假设。
