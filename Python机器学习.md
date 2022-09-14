---
title: Python机器学习
---

# 第一章	sklearn 数据集

**scikit-learn 数据集 API 介绍**

- `sklearn.datasets`
  - 加载获取流行数据集
  - `datasets.load_*()`
    - 获取小规模数据集，数据包含在 datasets 里
  - `datasets.fetch_*(data_home=None)`
    - 获取大规模数据集，需要从网络上下载，函数的第一个参数是`data_home`，表示数据集下载的目录，默认是`~/scikit_learn_data/`

**sklearn 小数据集**

- `sklearn.datasets.load_iris()`
  - 加载并返回鸢尾花数据集

| 名称         | 数量 |
| ------------ | ---- |
| 类别         | 3    |
| 特征         | 4    |
| 样本数量     | 150  |
| 每个类别数量 | 50   |





# 第二章 特征工程

## 2.1 特征提取

**特征提取 API**

```python
sklearn.feature_extraction
```



### 2.1.1 字典特征提取

作用：对字典数据进行特征值化

```python
sklearn.feature_extraction.DictVectorizer(sparse=True, ...)
```

- `DictVectorizer.fit_transform(X)` X：字典或者包含字典的迭代器	返回值：返回 sparse 矩阵
- `DictVectorizer.inverse_transform(X)` X：array 数组或者 sparse 矩阵    返回值：转换之前数据格式
- `DictVectorizer.get_feature_names()` 返回类别名称

示例：

```python
"""
    @File       : 02_字典特征提取.py
    @Description: TODO 
    @Software   : PyCharm
    @Author     : Crisp077
    @CreateTime : 2022/9/12 14:46
"""


def dict_extract(dict):
    # 导入sklearn中的特征提取库
    from sklearn.feature_extraction import DictVectorizer
    # 实例化一个转换器类 取消sparse稀疏矩阵表示
    DictVectorizer = DictVectorizer(sparse=False)
    # 传入字典 返回独热编码 one hot
    extract = DictVectorizer.fit_transform(dict)
    # 转换为原来的数据格式
    origin_data = DictVectorizer.inverse_transform(extract)
    # 返回类别名称
    feature_names = DictVectorizer.get_feature_names()
    return extract, origin_data, feature_names


if __name__ == '__main__':
    dict = [
        {'city': '北京', 'temperature': 28},
        {'city': '上海', 'temperature': 36},
        {'city': '武汉', 'temperature': 38}
    ]
    extract, origin_data, feature_names = dict_extract(dict)
    print(extract)
    print(origin_data)
    print(feature_names)

```



### 2.1.2 文本特征提取

作用：对文本数据进行特征值化

```python
sklearn.feature_extraction.text.CountVectorizer(stop_words=[])
```

- 返回词频矩阵
- `CountVectorizer.fit_transform(X)`    X：文本或者包含文本字符串的可迭代对象    返回值：sparse 矩阵
- `CountVectorizer.inverse_tranform(X)`    X：array 数组或者 sparse 矩阵    返回值：转换之前数据格式
- `CountVectorizer.get_feature_names()`    返回值：单词列表
- `stop_words`    停用词，可以将我们认为的对于文本没有影响的词填入其中

示例：

```python
"""
    @File       : 03_文本特征提取.py
    @Description: TODO 
    @Software   : PyCharm
    @Author     : Crisp077
    @CreateTime : 2022/9/12 15:14
"""


def txt_extract(txt):
    # 导入文本特征提取库
    from sklearn.feature_extraction.text import CountVectorizer
    # 实例化转换器
    transfer = CountVectorizer()
    # 使用fit_trasform
    extract = transfer.fit_transform(txt)
    # 转换为原来格式
    origin_data = transfer.inverse_transform(extract)
    # 返回文本特征类型
    feature_names = transfer.get_feature_names()

    return extract, origin_data, feature_names


if __name__ == '__main__':
    txt = ["life is short, I like Python",
           "life is not long, I like java"]
    extract, origin_data, feature_names = txt_extract(txt)
    print(origin_data)
    print(feature_names)
    print(extract.toarray())

```

> 注意：
>
> - CountVectorizer 方法对中文分词特征提取不太友好，因为中文之间没有分割符
> - 若要准确的对于中文词语进行特征提取，需要先进行分词操作



### 2.1.3 中文文本特征提取

使用`jieba`进行分词处理

```python
jieba.cut(txt)    # 返回值为生成器
```

示例：

```python
"""
    @File       : 04_中文文本特征提取.py
    @Description: TODO 
    @Software   : PyCharm
    @Author     : Crisp077
    @CreateTime : 2022/9/12 15:49
"""


def cut_chiesewords(txt):
    # 导入jieba库
    import jieba
    new_txt = " ".join(list((jieba.cut(txt))))

    return new_txt


def txt_extract(txt):
    # 导入文本特征提取库
    from sklearn.feature_extraction.text import CountVectorizer
    # 实例化转换器
    transfer = CountVectorizer()
    # 使用fit_trasform
    extract = transfer.fit_transform(txt)
    # 转换为原来格式
    origin_data = transfer.inverse_transform(extract)
    # 返回文本特征类型
    feature_names = transfer.get_feature_names()

    return extract, origin_data, feature_names


if __name__ == '__main__':
    texts = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
             "我们看到的从很远圣系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
             "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    # 将new_txt初始化为列表方便添加
    new_txt = []
    for txt in texts:
        # 对中文进行分词
        new_txt.append(cut_chiesewords(txt))
        # 文本特征提取
        extract, origin_txt, feature_names = txt_extract(new_txt)
        print(feature_names)
        print(extract.toarray())

```



### 2.1.4 Tf-idf 文本特征提取

- TF-IDF 的主要思想是：如果**某个词或短语在一篇文章中出现的概率高，并且在其他文章中很少出现**，则认为此词或者短语具有很好的类别区分能力，适合用来分类
- **TF-IDF 作用：用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度**

**公式**

- 词频（term frequency，tf）：指的是某一个给定的词语在该文件中出现的频率
- 逆向文档频率（inverse document frequency，idf）：是一个词语普遍重要性的度量。某一特定词语的 idf，可以**由总文件数目除以包含该词语之文件的数目，再将得到的商取以 10 为底得对数得到**
- $ tfidf_{i,j} = tf_{i,j} * idf_{i} $
- 最终得出结果可以理解为重要程度

**API**

```python
sklearn.feature_extraction.text.TfidfVectorizer(stop_word=None, ...)
```

- 返回词的权重矩阵
- `TfidfVectorizer.fit_tranform(X)`    X：文本或者包含文本字符串的可迭代对象
- `TfidfVectorizer.inverse_tranform(X)`    X：array 数组或者 sparse 矩阵    返回值：转换之前的数据格式
- `TfidfVectorizer.get_feature_names()`    返回值：单词列表

示例：

```python
"""
    @File       : 05_Tfidf文本特征提取.py
    @Description: 尝试提取关键字
    @Software   : PyCharm
    @Author     : Crisp077
    @CreateTime : 2022/9/12 16:51
"""


def Tfidf_extract(txt):
    # 导入相关包
    from sklearn.feature_extraction.text import TfidfVectorizer
    # 创建实例化对象
    transfer = TfidfVectorizer()
    # 传入文本
    extract = transfer.fit_transform(txt)
    # 特征值
    feature_names = transfer.get_feature_names()

    return extract, feature_names


if __name__ == '__main__':
    txt = ["life is short, I like Python",
           "life is not long, I like java"]
    extract, feature_names = Tfidf_extract(txt)
    print(feature_names)
    print(extract.toarray())

```





## 2.2 特征预处理

**作用**

- 通过一些转换函数将特征数据转换成更加适合算法模型的特征数据过程

**数值型数据的无量纲化**

- 归一化
- 标准化

**特征预处理 API**

```python
sklearn.preprocessing
```



### 2.2.1 归一化

**公式**

![归一化公式](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\归一化公式.png)

- 作用与每一列，max 为一列的最大值，min 为一列的最小值
- mx，mi 分别为指定区间值默认为（0，1）

**API**

```python
sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1)...)
```

- `MinMaxScaler.fit_transform(X)`
- X：numpy array 格式的数据`[n_samples, n_features]`
- 返回值：转换后形状相同的 array

**示例**

```python
"""
    @File       : 06_归一化.py
    @Description: TODO 
    @Software   : PyCharm
    @Author     : Crisp077
    @CreateTime : 2022/9/12 17:43
"""


def maxmin_demo(f_data):
    # 导入归一化相关库
    from sklearn.preprocessing import MinMaxScaler
    # 实例化
    scaler = MinMaxScaler()
    # 传入数据
    transfer = scaler.fit_transform(f_data)

    return transfer


if __name__ == '__main__':
    # 导入pandas库读取数据
    import pandas as pd

    # 读取数据
    data = pd.read_csv("dating.txt")
    # 不需要目标值，只取前三行数据
    data = data.iloc[:, :3]
    # 归一化
    new_data = maxmin_demo(data)
    print(new_data)

```

> 注意：
>
> - 最大值最小值是变化的
> - 最大值与最小值非常容易受异常点影响
> - 这种方法的鲁棒性较差，只适合传统精确小数据场景



### 2.2.2 标准化

**定义**

​	通过对原始数据进行变换把数据变换到均值为 0，标准差为 1 的范围内

**公式**

![标准化公式](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\标准化公式.png)

- 作用于每一列，mean 为平均值，sigma 为标准差

**归一化于标准化**

- 对于归一化来说：如果出现了异常点，影响了最大值和最小值，那么结果显然会发生改变
- 对于标准化来说：如果出现异常点，由于具有一定数据量，少量的异常点对于平均值的影响并不大，从而方差改变较小

**API**

```python
sklearn.preprocessing.StandardScaler()
```

- 处理之后，对每列来说，所有的数据都集中在均值 0 附近，标准差为 1
- `StandardScaler.fit_tranform(X)`
  - X：numpy array 格式的数据`[n_samples, n_features]`
  - 返回值：转换后的形状相同的 array

**示例**

```python
"""
    @File       : 07_标准化.py
    @Description: TODO 
    @Software   : PyCharm
    @Author     : Crisp077
    @CreateTime : 2022/9/12 18:12
"""


def standard_demo(data):
    # 导入相关库
    from sklearn.preprocessing import StandardScaler
    # 实例化
    scaler = StandardScaler()
    # 传入数据
    transfer = scaler.fit_transform(data)

    return transfer


if __name__ == '__main__':
    import pandas as pd

    # 导入数据
    data = pd.read_csv("dating.txt")
    # 取出我们要用的数据
    data = data.iloc[:, :3]
    # 标准化
    new_data = standard_demo(data)
    print(new_data)

```

> 总结：
>
> ​	在已有样本足够多的情况下比较稳定，适合现代嘈杂大数据场景





## 2.3 降维

​	**降维**是指在某些限定条件下，**降低随机变量（特征）个数**，得到一组**不相关主变量**的过程

**降维的两种方式**

- 特征选择
- 主成分分析



### 2.3.1 特征选择

**定义**

​	数据中包含**冗余或相关变量（或称特征、属性、指标等）**，旨在从**原有特征中找出主要特征**

**方法**

- Filtter（过滤式）：主要探究特征本身特点、特征于特征和目标之间关联
  - 方差选择法：低方差特征过滤
  - 相关系数
- Embedded（嵌入式）：算法自动选择特征（特征于目标值之间的关联）
  - 决策树：信息熵、信息增益
  - 正则化：L1、L2
  - 深度学习：卷积等

**模块**

```python
sklearn.feature_selection
```

#### 2.3.1.1 低方差特征过滤

​	删除低方差的一些特征

**API**

```python
sklearn.feature_selection.VarianceThreshold(threshold=0.0)
```

- 删除所有低方差特征
- `Variance.fit_transform(X)`
  - X：numpy array 个数的数据
  - 返回值：训练集差异低于 threshold 的特征值将被删除。默认值是保留所有非零方差特征，即删除所有样本中具有相同值的特征

**示例**

```python
"""
    @File       : 08_低方差特征过滤.py
    @Description: TODO 
    @Software   : PyCharm
    @Author     : Crisp077
    @CreateTime : 2022/9/12 18:41
"""


def variancethreshold_demo(data):
    # 导入相关包
    from sklearn.feature_selection import VarianceThreshold
    # 实例化对象
    threshold = VarianceThreshold(threshold=5)
    # 放入数据
    transfer = threshold.fit_transform(data)

    return transfer


if __name__ == '__main__':
    import pandas as pd

    # 读取数据
    data = pd.read_csv("factor_returns.csv")
    # 选取需要的列
    data = data.iloc[:, 1:-2]
    # 低方差特征过滤
    new_data = variancethreshold_demo(data)

    print(new_data)

```

#### 2.3.1.2 相关系数

- 皮尔逊相关系数（Pearson Correlation Coefficient）
  - 反应变量之间相关关系紧密程度的统计指标

**公式**

![皮尔逊相关系数公式](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\皮尔逊相关系数公式.png)

**特点**

相关系数的值介于 -1 与 +1 之间。性质如下：

- 当 r > 0 时，表示两变量正相关，r < 0 时，两变量为负相关
- 当 | r | = 1 时，表示两变量完全相关，当 r = 0 时，表示两变量间无相关关系
- 一般可按三级划分：
  - | r | < 0.4 为低度相关
  - 0.4 <= | r | < 0.7 为显著性相关
  - 0.7 <= | r | < 1 为高度线性相关

**API**

```py
from scipy.stats import pearsonr
```

**示例**

```python
"""
    @File       : 09_相关系数.py
    @Description: 皮尔逊相关系数
    @Software   : PyCharm
    @Author     : Crisp077
    @CreateTime : 2022/9/12 20:27
"""


def pearson_demo(x, y):
    # 导入相关包
    from scipy.stats import pearsonr
    factor = pearsonr(x, y)

    return factor


if __name__ == '__main__':
    import pandas as pd

    # 导入数据
    data = pd.read_csv("factor_returns.csv")
    factor = pearson_demo(data["revenue"], data["total_expense"])
    print(factor)

```

> 特征与特征之间相关性很高：
>
> 1. 选取其中一个作为代表
> 2. 加权求和



### 2.3.2 主成分分析（PCA）

- 定义：高维数据转化为低维数据的过程，在此过程中可能会舍弃原有数据、创建新的变量
- 作用：使数据维数压缩，尽可能降低维数（复杂度），损失少量信息
- 应用：回归分析或者聚类分析当中

**API**

```python
sklearn.decomposition.PCA(n_component=None)
```

- 将数据分解为较低维数空间
- `n_components`
  - 小数：表示保留百分之多少的信息
  - 整数：减少到多少特征
- `PCA.fit_transform(X)`    X：numpy array 格式的数据
- 返回值：转换后指定维度的 array

**示例**

```python
"""
    @File       : 11_instacart案例演示.py
    @Description: TODO 
    @Software   : PyCharm
    @Author     : Crisp077
    @CreateTime : 2022/9/12 20:54
"""


def PCA_demo(f_data, f_component):
    # 导入库
    from sklearn.decomposition import PCA
    # 实例化对象
    PCA_instant = PCA(n_components=f_component)
    # 传入数据
    transfer = PCA_instant.fit_transform(f_data)

    return transfer


if __name__ == '__main__':
    import pandas as pd

    # 1、获取数据
    order_products = pd.read_csv("instacart/order_products__prior.csv")
    products = pd.read_csv("instacart/products.csv")
    orders = pd.read_csv("instacart/orders.csv")
    aisles = pd.read_csv("instacart/aisles.csv")

    # 2、合并表
    tab1 = pd.merge(aisles, products, on=["aisle_id", "aisle_id"])
    tab2 = pd.merge(tab1, order_products, on=["product_id", "product_id"])
    tab3 = pd.merge(tab2, orders, on=["order_id", "order_id"])

    # 3、找到user_id和aisle之间的关系
    table = pd.crosstab(tab3["user_id"], tab3["aisle"])

    # 4、PCA降维
    new_data = PCA_demo(table, 0.95)

    print(new_data)

```





# 第三章 分类算法

## 3.1 sklearn 转换器和估计器

**转换器**

之前做特征工程的步骤

1. 实例化（实例化的是一个转换器类（Transformer））
2. 调用 fit_transform（对于文档建立分类词频矩阵，不能同时调用）

我们把特征工程的接口称之为转换器，其中转换器的调用有以下几种形式

- fit_transform
- fit
- tranform

**估计器（sklearn 机器学习算法的实现）**

在 sklearn 中，估计器（estimator）是一个重要的角色，是一类实现了算法的 API

- 用于分类的估计器：
  - `sklearn.neighbors`    k-近邻算法
  - `sklearn.navie_bayes`    贝叶斯
  - `sklearn.linear_model.LogisticRegression`    逻辑回归
  - `sklearn.tree`    决策树与随机森林
- 用于回归的估计器：
  - `sklearn.linear_model.LinearRegression`    线性回归
  - `sklearn.linear_model.Ridge`    岭回归
- 用于无监督学习的估计器
  - `sklearn.cluster.KMeans`    聚类

**估计器（estimator）**

1. 实例化一个 `estimator`

2. `estimator.fit(x_train, y_train)`    计算

   - 调用完毕，模型生成

3. 模型评估

   1. 直接对比真实值和预测值

      `y_predict = estimator.predict(x_test)`

      `y_test == y_predict`

   2. 计算准确率

      `accuracy = estimator.score(x_test, y_test)`



## 3.2 K-近邻算法

**算法原理**

​	K Nearest Neighbor 算法又叫 KNN 算法，如果一个样本在特征空间中的 **k 个最相似（即特征空间中最邻近）的样本中的大多数属于某一个类别**，则该样本也属于这个类别

**距离公式**

欧氏距离

![欧式距离公式](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\欧式距离公式.png)

**k 的取值的影响**

- k 取值过小，容易受到异常点的影响
- k 取值过大，易受样本不均衡的影响

**KNN API**

```python
sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='auto')
```

- `n_neighbors`：int，可选（默认5），k_neighbors 查询默认使用的邻居数
- `algorithm`：`{'auto', 'ball_tree', 'kd_tree', 'brute'}`，可选用于计算最近邻居的算法

**示例**

```python
"""
    @File       : 12_KNN_iris.py
    @Description: 利用KNN算法分类鸢尾花
    @Software   : PyCharm
    @Author     : Crisp077
    @CreateTime : 2022/9/12 22:02
"""

if __name__ == '__main__':
    # 1、获取数据集
    from sklearn.datasets import load_iris

    iris = load_iris()

    # 2、划分数据集
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=6)

    # 3、特征工程：标准化
    from sklearn.preprocessing import StandardScaler

    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4、KNN算法预估器
    from sklearn.neighbors import KNeighborsClassifier

    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)

    # 5、模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值：\n", y_test == y_predict)

    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

```

**总结**

- 优点
  - 简单，易于理解，易于实现，无需训练
- 缺点
  - 懒惰算法，对测试样本分类时的计算量大，内存开销大
  - 必须指定 k 值，k 值选择不当则分类精度不能保证
- 使用场景：小数据场景



## 3.3 模型选择与调优

**交叉验证（cross validation）**

​	将拿到的训练数据，分为训练和验证集。例如将数据分成 4 份，其中一份作为验证集。然后经过 4 次的测试，每次都更换不同的验证集。即得到 4 组模型的结果，取平均值作为最终结果。又称 4 折交叉验证

![交叉验证](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\交叉验证.png)

**超参数搜索-网格搜索（Grid Search）**

​	通常情况下，有很多参数是需要手动指定的（如 K-近邻算法中的 K 值），这种叫超参数。但是手动过程繁杂，所以需要对模型预设计中超参数组合。每种超参数都采用交叉验证来进行评估。最后选出最优参数组合建立模型

![交叉验证](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\网格搜索.png)

**模型选择与调优 API**

```python
sklearn.model_selection.GridSearchCV(estimator, param_grid=None, cv=None)
```

- 对估计器的指定参数进行详尽搜索
- `estimator`：估计器对象
  - `param_grid`：估计器参(dict)`{"n_neighbors": [1, 3, 5]}`
  - `cv`：指定几折交叉验证
  - `fit()`：输入训练数据
  - `score()`：准确率
  - 结果分析
    - 最佳参数：`best_params_`
    - 最佳结果：`best_score_`
    - 最佳估计器：`best_estimator_`
    - 交叉验证法：`cv_results_`

示例：

```python
"""
    @File       : 13_网格搜索与交叉验证.py
    @Description: TODO 
    @Software   : PyCharm
    @Author     : Crisp077
    @CreateTime : 2022/9/13 9:28
"""

if __name__ == '__main__':
    # 1、获取数据集
    from sklearn.datasets import load_iris

    iris = load_iris()

    # 2、划分数据集
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=6)

    # 3、特征工程：标准化
    from sklearn.preprocessing import StandardScaler

    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4、KNN算法预估器
    from sklearn.neighbors import KNeighborsClassifier

    estimator = KNeighborsClassifier()

    # 加入网格搜索与交叉验证
    from sklearn.model_selection import GridSearchCV

    # 参数准备
    param_dict = {"n_neighbors": [1, 3, 5, 9, 11]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)
    estimator.fit(x_train, y_train)

    # 5、模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值：\n", y_test == y_predict)

    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    # 打印网格搜索和交叉验证内容
    print("最佳参数：\n", estimator.best_params_)
    print("最佳结果：\n", estimator.best_score_)
    print("最佳估计器：\n", estimator.best_estimator_)
    print("交叉验证结果：\n", estimator.cv_results_)

```



## 3.4 朴素贝叶斯算法

**联合概率、条件概率与相互独立**

- 联合概率：包含多个条件，且所有条件同时成立的概率
  - 记作：P(A, B)
- 条件概率：就是事件 A 在另一个事件 B 已经发生条件下的发生概率
  - 记作：P(A | B)
- 相互独立：如果 P(A, B) = P(A)P(B)，则称事件 A 与事件 B 相互独立

**贝叶斯公式**

![贝叶斯公式](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\贝叶斯公式.png)

注：W 为给定文档的特征值（频数统计，预测文档提供），C 为文档类别

**拉普拉斯平滑系数**

目的：防止计算出的分类概率为 0

![拉普拉斯平滑系数](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\拉普拉斯平滑系数.png)

**API**

```python
sklearn.naive_bayes.MultinomialNB(alpha=1.0)
```

- 朴素贝叶斯分类
- alpha：设置拉普拉斯平滑系数

**示例**

```python
"""
    @File       : 14_朴素贝叶斯算法.py
    @Description: TODO 
    @Software   : PyCharm
    @Author     : Crisp077
    @CreateTime : 2022/9/13 12:23
"""


# 流程分析：
# 1、获取数据
# 2、划分数据集
# 3、特征工程：Tfidf文本特征抽取
# 4、朴素贝叶斯算法预估器流程
# 5、模型评估


def tfidf_demo(f_x_train, f_x_test):
    # 导入相关包
    from sklearn.feature_extraction.text import TfidfVectorizer
    # 创建实例化对象
    transfer = TfidfVectorizer()
    # 传入文本
    new_x_train = transfer.fit_transform(f_x_train)
    new_x_test = transfer.transform(f_x_test)

    return new_x_train, new_x_test


def navie_bayes_demo(f_x_train, f_y_train):
    # 导包
    from sklearn.naive_bayes import MultinomialNB
    # 创建实例化对象
    estimator = MultinomialNB()
    # 传入对象
    estimator.fit(f_x_train, f_y_train)

    return estimator


if __name__ == '__main__':
    # 1、获取数据
    from sklearn.datasets import fetch_20newsgroups

    news = fetch_20newsgroups(subset="all")

    # 2、划分数据集
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)

    # 3、特征工程：Tfidf文本特征抽取
    x_train, x_test = tfidf_demo(x_train, x_test)

    # 4、朴素贝叶斯算法预估器流程
    estimator = navie_bayes_demo(x_train, y_train)

    # 5、模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值：\n", y_test == y_predict)

    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

```

**总结**

- 优点：
  - 朴素贝叶斯模型起源于古典数学理论，有稳定的分类效率
  - 对缺失值不太敏感，算法也比较简单，常用于文本分类
  - 分类准确度高，速度快
- 缺点：
  - 由于使用了样本属性独立性的假设，所以如果特征属性有关联时其效果不好



## 3.5 决策树

​	信息：消除随机不定性的东西

**信息熵的定义**

![信息熵](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\信息熵.png)

**决策树的划分依据之一——信息增益**

​	特征 A 对训练集 D 的信息增益 g(D, A)，定义为集合 D 的信息熵 H(D) 与特征 A 给定条件下 D 的信息条件熵 H(D | A) 之差，即公式为：

![信息增益](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\信息增益.png)

公式的详细解释：

![信息熵与条件熵的计算](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\信息熵与条件熵的计算.png)

注：$ C_k $表示属于某个类别的样本数

> 注：
>
> ​	信息增益表示得知特征 X 的信息以后的不确定性减少的程度使得类 Y 的信息熵减少的程度

**决策树 API**

```python
sklearn.tree.DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=None)
```

- 决策树分类器
- `criterion`：默认是 'gini' 系数，也可以选择信息增益的熵 'entropy'
- `max_depth`：树的深度大小
- `random_state`：随机数种子

**决策树可视化 API**

```python
sklearn.tree.export_graphviz()	# 该函数能够导出DOT格式
tree.export_graphviz(estimator, out_file='tree.dot', feature_names=[...])
```

**案例**

```python
"""
    @File       : 15_决策树.py
    @Description: 用决策树对鸢尾花进行分类
    @Software   : PyCharm
    @Author     : Crisp077
    @CreateTime : 2022/9/13 14:47
"""


def decisiontree_demo(f_x_train, f_y_train):
    # 导入包
    from sklearn.tree import DecisionTreeClassifier
    # 实例化
    estimator = DecisionTreeClassifier(criterion='entropy')
    # 传入参数
    estimator.fit(f_x_train, f_y_train)

    return estimator


if __name__ == '__main__':
    # 1、获取数据集
    from sklearn.datasets import load_iris

    iris = load_iris()

    # 2、划分数据集
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    # 3、决策树预估器
    estimator = decisiontree_demo(x_train, y_train)

    # 4、模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值：\n", y_test == y_predict)
    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    # 5、决策树可视化
    from sklearn.tree import export_graphviz

    export_graphviz(estimator, out_file='iris_tree.dot', feature_names=iris.feature_names)

```

**决策树总结**

- 优点：
  - 简单的理解和解释，可视化
- 缺点：
  - 决策树学习者可以创建不能很好地推广数据地过于复杂的树，这被称为过拟合
- 改进：
  - 减枝 cart 算法（决策树 API 当中已经实现，随机森林参数调优有相关介绍）
  - 随机森林

注：企业重要决策，由于决策树很好地分析能力，在决策过程中应用较多，可以选择特征



## 3.6 集成学习方法之随机森林

**集成学习方法**

​	集成学习通过建立几个模型组合地来解决单一预测问题。它的工作原理是生成多个分类器 / 模型，各自独立地学习和做出预测。这些预测最后结果成组合预测，因此优于任何一个单分类地做出预测

**随机森林**

​	在机器学习中，随机森林是一个包含多个决策树地分类器，并且输出的类别是由个别树输出的类别的众数而定

**随机森林原理过程**

学习算法根据下列算法而建造每棵树：

- 用 N 来表示训练用例（样本的个数），M 表示特征数目
  1. 一次随机选出一个样本，重复 N 次（有可能出现重复的样本）
  2. 随机去选出 m 个特征，m << M，建立随机数
- 采取 bootStrap 抽样

**为什么采取 BootStrap 抽样**

- 为什么要随机抽样训练集？
  - 如果不进行随机抽样，每棵树的训练集都一样，那么最终训练出的树分类结果也是完全一样的
- 为什么要有放回地抽样？
  - 如果不是有放回的抽样，那么每棵树的训练样本都是不同的，都是没有交集的，
    这样每棵树都是“有偏的”，都是绝对“片面的”（当然这样说可能不对），也就是
    说每棵树训练出来都是有很大的差异的；而随机森林最后分类取决于多棵树（弱
    分类器)的投票表决

**随机森林 API **

```python
sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, bootstrap=True, random_state=None, min_sample_split=2)
```

- 随机森林分类器
- `n_estimators`：integer，optional（default = 10）森林里的树木数量
- `criteria`：string，可选（default = 'gini'）分割特征的测量方法
- `max_depth`：integer 或 None，可选（默认=无）树的最大深度
- `max_features='auto'`：每个决策树的最大特征数量
  - `auto`：`max_features=sqrt(n_features)`
  - `sqrt`：`max_features=sqrt(n_features)`同 auto
  - `log2`：`max_features=log2(n_features)`
  - `None`：`max_features=n_features`
- `bootstrap`：boolean，optional（default = True）是否在构建时使用放回抽样
- `min_samples_split`：节点划分最少样本数
- `min_samples_leaf`：叶子节点的最小样本数

**总结**

- 在当前所有算法中，具有较好的准确率
- 能够有效地运行在大数据集上，处理具有高维特征地输入样本，而且不需要降维
- 能够评估各个特征在分类问题上地重要性





# 第四章 回归与聚类算法

## 4.1 线性回归

**应用场景**

- 房价预测
- 销售额度预测
- 金融：贷款额度预测、利用线性回归以及系数分析因子

**定义与公式**

​	线性回归（Linear regression）是利用回归方程（函数）对一个或多个自变量（特征值）和因变量（目标值）之间关系进行建模的一种分析方式

- 特点：只有一个自变量的情况称为单变量回归，多于一个自变量情况的叫做多元回归

![线性回归公式](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\线性回归公式.png)

**损失函数**

总损失定义为

![损失函数](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\损失函数.png)

- $ y_i $为第 i 个训练样本的真实值
- $ h(x_i) $为第 i 个训练样本特征值组合预测函数
- 又称最小二乘法

**优化算法**

线性回归经常使用两种优化算法

- 正规方程

![正规方程](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\正规方程.png)

> 理解：X 为特征值矩阵，y 为目标值矩阵。直接求到最好的结果
>
> 缺点：当特征值过多过复杂时，求解速度太慢并且得不到结果

- 梯度下降（Gradient Descent）

![梯度下降](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\梯度下降.png)

> 理解：
>
> ​	alpha 为学习速率，需要手动指定（超参数），alpha 旁边的整体表示方向沿着这个函数下降的方向找，最后就能找到山谷的最低点，然后更新 W 值
>
> 使用：面对训练数据规模十分庞大的任务，能够找到较好的结果

**线性回归 API**

```python
sklearn.linear_model.LinearRegression(fit_intercept=True)
```

- 通过正规方程优化
- `fit_intercept`：是否计算偏置
- `LinearRegression.coef_`：回归系数
- `LinearRegression.intercept_`：偏置

```python
sklearn.linear_model.SGDRRegressor(loss="squared_loss", fit_intercept=True, learning_rate="invscaling", eya0=0.01)
```

- SGDRegressor 类实现了随机梯度下降学习，它支持不同的 loss 函数和正则惩罚项来拟合线性回归模型
- `loss`：损失类型
  - `loss="squared_loss"`：普通最小二乘法
- `fit_intercept`：是否计算偏置
- `learning_rate`：String，optional
  - 学习率填充
  - `constant`：eta = eta0
  - `optimal`：eta = 1.0 / ( alpha * ( t + t0 )[default]
  - `invscaling`：eta = eta0 / pow(t, power_t)
    - `power_t=0.25`：存在父类当中
  - 对于一个异常值的学习率来说，可以使用 `learning_rate='constant'`，并使用 eta0 来指定学习率
  - `LinearRegression.coef_`：回归系数
  - `LinearRegression.intercept_`：偏置

**回归性能评估**

均方误差（Mean Squared Error，MSE）评级机制：

![均方误差](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\均方误差.png)

> 注：$ y^i $是预测值，$ y^- $ 是真实值

```python
sklearn.metrics.mean_squared_error(y_true, y_pred)
```

- 均方误差回归损失
- `y_true`：真实值
- `y_pred`：预测值
- `return`：浮点数结果

**示例**

```python
"""
    @File       : 16_线性回归.py
    @Description: 预测波士顿房价案例
    @Software   : PyCharm
    @Author     : Crisp077
    @CreateTime : 2022/9/13 17:12
"""


def linear1_demo():
    """
    正规方程的优化方法对波士顿房价进行预测
    :return:
    """
    # 1、获取数据
    from sklearn.datasets import load_boston
    boston = load_boston()

    # 2、划分数据集
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 3、标准化
    from sklearn.preprocessing import StandardScaler
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4、预估器
    from sklearn.linear_model import LinearRegression
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    # 5、得出模型
    print("正规方程权重系数为:", estimator.coef_)
    print("正规方程偏置为:", estimator.intercept_)

    # 6、评估模型
    from sklearn.metrics import mean_squared_error
    y_predict = estimator.predict(x_test)
    print("预测房价:", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("正规方程-均方误差:", error)

    return None


def linear2_demo():
    """
        梯度下降的优化方法对波士顿房价进行预测
        :return:
    """
    # 1、获取数据
    from sklearn.datasets import load_boston
    boston = load_boston()

    # 2、划分数据集
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 3、标准化
    from sklearn.preprocessing import StandardScaler
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4、预估器
    from sklearn.linear_model import SGDRegressor
    estimator = SGDRegressor(learning_rate="constant", eta0=0.01, max_iter=10000)
    estimator.fit(x_train, y_train)

    # 5、得出模型
    print("梯度优化权重系数为:", estimator.coef_)
    print("梯度优化偏置为:", estimator.intercept_)

    # 6、评估模型
    from sklearn.metrics import mean_squared_error
    y_predict = estimator.predict(x_test)
    print("预测房价:", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("梯度下降-均方误差:", error)

    return None


if __name__ == '__main__':
    linear1_demo()
    linear2_demo()

```

**正规方程与梯度下降对比**

|        梯度下降        |          正规方程          |
| :--------------------: | :------------------------: |
|     需要选择学习率     |           不需要           |
|      需要迭代求解      |        一次运算得出        |
| 特征数量较大时可以使用 | 需要计算方程，时间复杂度高 |

- 选择
  - 小规模数据：
    - LinearRegression（不能解决拟合问题）
    - 岭回归
  - 大规模数据：SGDRegressor



## 4.2 欠拟合与过拟合

**原因以及解决办法**

- 欠拟合原因以及解决办法
  - 原因：学习到数据的特征过少
  - 解决办法：增加数据的特征数量
- 过拟合原因以及解决办法
  - 原因：原始特征过多，存在一些嘈杂特征，模型过于复杂是因为模型尝试去兼顾各个测试数据点
  - 解决办法：正则化

**正则化类别**

- L2 正则化
  - 作用：可以使得其中一些 W 的值很小，都接近于 0，消除某个特征值的影响
  - 优点：越小的参数说明模型越简单，越简单的模型则越不容易产生过拟合现象
  - Ridge 回归
  - 加入 L2 正则化后的损失函数：

![L2正则化](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\L2正则化.png)

> 注：m 为样本数，n 为特征数

- L1 正则化
  - 作用：可以使得其中一些 W 的值直接为 0，删除这个特征的影响
  - LASSO 回归



## 4.3 线性回归的改进—岭回归

​	岭回归，其实也是一种线性回归。只不过在算法建立回归方程时候，加上正则化的限制，从而达到解决过拟合的效果

**API**

```python
sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True, solver="auto", normalize=False)
```

- 具有 L2 正则化的线性回归
- `alpha`：正则化力度，也叫`lambda`
  - 取值：`0~1`，`1~10`
- `solver`：会根据数据自动化选择优化方法
  - `sag`：如果数据集、特征值都比较大，选择该随机梯度下降优化
- `normalize`：数据是否进行标准化
  - `normalize=False`：可以在 fit 之前调用 `preprocessing.StandardScaler`标准化数据
- `Ridge.coef_`：回归权重
- `Ridge.intercept_`：回归偏置

​	**Ridge 方法相当于 SGDRegressor(penalty='l2', loss='squared_loss')，只不过 SGDRegressor 实现了一个普通的随机梯度下降学习，推荐使用 Ridge（实现了 SAG）**

```python
sklearn.linear_model.RidgeCV(_BaseRidgeCV, RegressorMixin)
```

- 具有 L2 正则化的线性回归，可以进行交叉验证
- `coef_`：回归系数

**示例**

```python
"""
    @File       : 17_岭回归.py
    @Description: TODO 
    @Software   : PyCharm
    @Author     : Crisp077
    @CreateTime : 2022/9/13 22:39
"""


def linear3_demo():
    # 1、获取数据
    from sklearn.datasets import load_boston
    boston = load_boston()

    # 2、划分数据集
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 3、标准化
    from sklearn.preprocessing import StandardScaler
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4、预估器
    from sklearn.linear_model import Ridge
    estimator = Ridge(alpha=0.5, max_iter=10000)
    estimator.fit(x_train, y_train)

    # 5、得出模型
    print("岭回归权重系数为:", estimator.coef_)
    print("岭回归偏置为:", estimator.intercept_)

    # 6、评估模型
    from sklearn.metrics import mean_squared_error
    y_predict = estimator.predict(x_test)
    print("预测房价:", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("岭回归-均方误差:", error)

    return None


if __name__ == '__main__':
    linear3_demo()

```



## 4.4 分类算法—逻辑回归与二分类

​	逻辑回归（Logistic Regression）是机器学习中的一种分类模型，逻辑回归是一种分类算法，虽然名字中带有回归，但是它与回归之间有一定的联系。由于算法的简单和高效，在实际应用非常广泛

**逻辑回归原理**

- 输入

![逻辑回归输入](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\逻辑回归输入.png)

​		逻辑回归的输入就是一个线性回归的结果

- 激活函数
  - sigmoid 函数

![逻辑回归激活函数](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\逻辑回归激活函数.png)

​		回归的结果输入到 sigmoid 函数中

​		输出结果：[ 0, 1 ] 区间中的一个概率值，默认 0.5 为阈值

![sigmoid函数](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\sigmoid函数.png)

**损失以及优化**

- 损失
  - 逻辑回归的损失，称之为对数似然损失

![综合完整损失函数](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\综合完整损失函数.png)

- 优化
  - 同样使用梯度下降优化算法，去减少损失函数的值。这样去更新逻辑回归前面对应算法的权重参数，提升原本属于 1 类别的概率，降低原本是 0 类别的概率

**逻辑回归 API**

```python
sklearn.linear_model.LogisticRegression(solver='liblinear', penalty='l2', C=1.0)
```

- `solver`：优化求解方式（默认开源的 bilinear 库实现，内部使用了坐标轴下降法来迭代优化损失函数）
  - `sag`：根据数据集自动选择，随机平均梯度下降
- `penalty`：正则化的种类
- `C`：正则化力度

**分类的评估方法**

- 混淆矩阵
  - 在分类任务下，预测结果（Predicted Condition）与正确标记（True Condition）之间存在四种不同的组合，构成混淆矩阵（适用于多分类）

![混淆矩阵](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\混淆矩阵.png)

- 精确率（Precision）与召回率（Recall）
  - 精确率：预测结果为正例样本中真实为正例的比例
  - 召回率：真实为正例样本中预测结果为正例的比例

**分类评估报告 API**

```python
sklearn.metrics.classification_reply(y_true, y_pres, labels=[], target_names=None)
```

- `y_true`：真实目标值
- `y_pred`：估计器预测目标值
- `labels`：指定类别对应的数字
- `target_names`：目标类别名称
- `return`：每个类别精确率与召回率

**ROC 曲线与 AUC 指标**

**TPR 与 FPR**

- TPR = TP / (TP + FN)
  - 所有真实类别为 1 的样本中，预测类别为 1 的概率
- FPR = FP / (FP + TN)
  - 所有真实类别为 0 的样本中，预测类别为 1 的比例

**ROC 曲线**

- ROC 曲线的横轴就是 FPRate，纵轴就是 TPRate，当二者相等时，表示的意义则是：对于不论真实类别是 1 还是 0 的样本，分类器预测为 1 的概率是相等的，此时 AUC 为 0.5

![ROC曲线](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\ROC曲线.png)

**AUC 指标**

- AUC 的概率意义是随机取一对正负样本，正样本得分大于负样本的概率
- AUC 的最小值为 0.5，最大值为 1，取值越高越好
- AUC = 1，完美分类器，采用这个预测模型时，不管设定什么阀值都能得出完美预测。绝大多数预测的场合，不存在完美分类器
- 0.5 < AUC < 1，优于随机猜测。这个分类器（模型）妥善设定闻值的话，能有预测价值

**AUC 计算 API**

```python
sklearn.metric.roc_auc_score(y_true, y_score)
```

- 计算 ROC 曲线面积，即 AUC 值
- `y_true`：每个样本的真实类别，必须为 0 （反例），1 （正例）标记
- `y_score`：预测得分，可以是正类的估计概率、置信率或者分类器方法的返回值



## 4.5 模型保存和加载

**API**

```python
import pickle
```

- 保存：`pickle.dump(rf, 'test.pkl')`
-  加载：`estimator = pickle.load('test.pkl')`

**案例**

```python
# 保存模型
import pickle
with open('linear3_model.pkl', 'wb') as f:
  pickle.dump(estimator_orgin, f)
# 加载模型
with open('linear3_model.pkl', 'rb') as f:
    estimator = pickle.load(f)
```





## 4.6 无监督学习—K-means 算法

**K-means 聚类步骤**

![k-means聚类](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\k-means聚类.png)

1. 随机设置 K 个特征空间内的点作为初识的聚类中心
2. 对于其他每个点计算到 K 和中心的距离，未知的点选择最近的一个聚类中心点作为标记类别
3. 接着对着标记的聚类中心之后，中心计算出每个聚类的新中心点（平均值）
4. 如果计算得出的新中心点与原中心点一样，那么结束，否则重新进行第二步过程

**API**

```python
sklearn.cluster.Kmeans(n_cluster=8, init='k-means++')
```

- K-means 聚类
- `n_cluster`：开始的聚类中心数量
- `init`：初识方法，默认为 'k-means++'
- `labels_`：默认标记的类型，可以和真实值比较

**K-means 性能评估指标**

- 轮廓系数

![轮廓系数](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\轮廓系数.png)

> 注：对于每个点 i 为已聚类数据中的样本，$ b_i $ 为 i 到其它族群的所有样本的距离最小值，$ a_i $ 为ⅰ到本身簇的距离平均值。最终计算出所有的样本点的轮廓系数平均值

![轮廓系数图](C:\Users\15640\Desktop\学习笔记\Python机器学习_images\轮廓系数图.png)

结论：

- 如果 $ b_i >> a_i $：趋近于 1 效果越好
- 如果 $ b_i << a_i $：趋近于 -1 效果不好

**API**

```python
sklearn.metrics.silhouette_score(X, labels)
```

- 计算所有样本的平均轮廓系数
- `X`：特征值
- `labels`：被聚类标记的目标值

**K-means 总结**

- 特点分析：采用迭代算法，直观易懂并且非常实用
- 缺点：容易收敛到局部最优解（多次聚类）
