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
