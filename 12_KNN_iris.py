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
