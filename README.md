# sklearnDemo

机器学习 scikit-learn

## 分类/回归 regression

* [plot_cv_predict](./regression/plot_cv_predict.py)

    use cross_val_predict to visualize prediction errors.

* [Isotonic regression](./regression/plot_isotonic_regression.py)

    保序回归(Isotonic Regression)对生成的数据进行保序回归的一个实例.保序回归能在训练数据上发现一个非递减逼近函数的同时最小化均方误差。这样的模型的好处是，它不用假设任何形式的目标函数，(如线性)。为了比较，这里用一个线性回归作为参照。

* [plot_multioutput_face_completion](./regression/plot_multioutput_face_completion.py)

## 广义线性模型

## 监督学习 Supervised learning

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

### 逻辑回归 LogisticRegression

* [plot logistic path](./regression/)

### 感知机

    Perceptron 是另一种简单的适合大规模学习的算法。默认情况下:

    它不需要学习率不需要正则化(罚项)
    只会在判错情况下更新模型。

### Passive Aggressive Algorithms¶

Passive Aggressive Algorithms 是一些列大规模学习的算法。这些算法和感知机非常相似，并不需要学习率。但是和感知机不同的是，这些算法都包含有一个正则化参数 C

### Robust regression(稳健回归)

主要思路是对异常值十分敏感的经典最小二乘回归目标函数的修改。 它主要用来拟合含异常数据(要么是异常数据,要么是模型错误) 的回归模型。

## 线性与二次判别分析

## Kernel ridge regression(KRR)

## 支持向量机 Support vector machines (SVMs)

SVMs are a set of supervised learning methods used for classification, regression and outliers detection.

* The advantages of support vector machines are:

    * Effective in high dimensional spaces.
    * Still effective in cases where number of dimensions is greater than the number of samples.
    * Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
    * Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

* The disadvantages of support vector machines include:

    * If the number of features is much greater than the number of samples, the method is likely to give poor performances.
    * SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

## 随机梯度下降 Stochastic Gradient Descent (SGD)

Stochastic Gradient Descent (SGD) 是一种简单但又非常高效的方式判别式学习方法，比如凸损失函数的线性分类器如 Support Vector Machines 和 Logistic Regression. 虽然 SGD 已经在机器学习社区出现很长时间，但是在近期在大规模机器学习上受到了相当大数量的关注。

SGD 已经被成功应用到大规模和稀疏机器学习问题上，通常为文本分类和自然语言处理。如果给定数据是稀疏的，那么该模块中的分类器 很容易把问题规模缩放到超过 10^5 训练样本和超过 10^5 的特征数量。

* SGD 的优势如下：

    * 高效性.
    * 容易实现 (lots of opportunities for code tuning 大量代码调整的机会).

* SGD 缺点如下：

    * SGD 需要许多超参数,比如正则化参数、迭代次数
    * SGD 对特征规模比较敏感(应该是特征维数)

### 最邻近法

主要是一种非监督或基于临近的监督学习方法. 非监督最邻近法是许多其他学习算法的基础，特别是流行学习方法及谱聚类方法. 基于临近的监督分类主要在一下两方面具有优势: 具有离散标签数据的`分类`和 连续标签数据的`回归`..

### 高斯过程(Gaussian Processes)

针对机器学习的高斯过程(Gaussian Processes for Machine Learning,即 GPML) 是一个通用的监督学习方法，主要被设计用来解决 回归 问题。 它也可以扩展为 概率分类(probabilistic classification)，但是在当前的实现中，这只是 回归 练习的一个后续处理。

* GPML 的优势如下:

    * 预测是对观察值的插值（至少在普通相关模型上是的）.
    * 预测是带有概率的(Gaussian)。所以可以用来计算经验置信区间和超越概率 以便对感兴趣的区域重新拟合（在线拟合，自适应拟合）预测。
    * 多样性: 可以指定不同的线性回归模型 linear regression models 和相关模型 correlation models 。 它提供了普通模型，但也能指定其它静态的自定义模型

* GPML 的缺点如下:

    * 不是稀疏的，它使用全部的样本/特征信息来做预测。多维空间下会变得低效 – 即当特征的数量超过几十个,它可能确实会表现很差，而且计算效率下降。分类只是一个后处理过程, 意味着要建模， 首先需要提供试验的完整浮点精度标量输出 y 来解决回归问题。要感谢高斯的预测的属性，已经有了广泛应用，比如：最优化和概率分类

### Cross decomposition

### 朴素贝叶斯(Naive Bayes)

### 决策树(Decision Trees)

## 非监督学习 Unsupervised learning

### clustering methods

* MiniBatchMeans

* AffinityPropagation

* MeanShift

* SpectralClustering

* Ward

* AgglomerativeClustering

* DBSCAN

* Birch

### K-means(Lloyd’s algorithm)

* Mini Batch K-Means

    K-Means (分批处理) 算法是常用的聚类算法，但其算法本身存在一定的问题，例如在大数据量下的计算时间过长就是一个重要问题。为此，Mini Batch K-Means，这个基于 K-Means 的变种聚类算法应运而生。
    实际上，这种思路不仅应用于 K-Means 聚类，还广泛应用于梯度下降、深度网络等机器学习和深度学习算法。

### Affinity Propagation

    AP 算法的基本思想是将全部样本看作网络的节点，然后通过网络中各条边的消息传递计算出各样本的聚类中心。聚类过程中，共有两种消息在各节点间传递，分别是吸引度( responsibility)和归属度(availability) 。AP 算法通过迭代过程不断更新每一个点的吸引度和归属度值，直到产生 m 个高质量的 Exemplar（类似于质心），同时将其余的数据点分配到相应的聚类中。

    1. 无需指定聚类“数量”参数。AP 聚类不需要指定 K（经典的 K-Means）或者是其他描述聚类个数（SOM 中的网络结构和规模）的参数，这使得先验经验成为应用的非必需条件，人群应用范围增加。
    2. 明确的质心（聚类中心点）。样本中的所有数据点都可能成为 AP 算法中的质心，叫做 Examplar，而不是由多个数据点求平均而得到的聚类中心（如 K-Means）。
    3. 对距离矩阵的对称性没要求。AP 通过输入相似度矩阵来启动算法，因此允许数据呈非对称，数据适用范围非常大。
    4. 初始值不敏感。多次执行 AP 聚类算法，得到的结果是完全一样的，即不需要进行随机选取初值步骤（还是对比 K-Means 的随机初始值）。
    5. 算法复杂度较高，为 O(N*N*logN)，而 K-Means 只是 O(N\*K)的复杂度。因此当 N 比较大时(N>3000)，AP 聚类算法往往需要算很久。
    6. 若以误差平方和来衡量算法间的优劣，AP 聚类比其他方法的误差平方和都要低。（无论 k-center clustering 重复多少次，都达不到 AP 那么低的误差平方和）

    缺点

    1. AP 聚类应用中需要手动指定 Preference 和 Damping factor，这其实是原有的聚类“数量”控制的变体。
    2. 算法较慢。由于 AP 算法复杂度较高，运行时间相对 K-Means 长，这会使得尤其在海量数据下运行时耗费的时间很多。

### Mean Shift
