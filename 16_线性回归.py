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
