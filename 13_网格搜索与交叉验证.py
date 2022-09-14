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
