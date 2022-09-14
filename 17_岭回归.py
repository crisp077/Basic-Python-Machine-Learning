"""
    @File       : 17_岭回归.py
    @Description: TODO 
    @Software   : PyCharm
    @Author     : Crisp077
    @CreateTime : 2022/9/13 22:39
"""
import pickle


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
    estimator_orgin = Ridge(alpha=0.5, max_iter=10000)
    estimator_orgin.fit(x_train, y_train)

    # 保存模型
    import pickle
    # with open('linear3_model.pkl', 'wb') as f:
    #     pickle.dump(estimator_orgin, f)

    # 加载模型
    with open('linear3_model.pkl', 'rb') as f:
        estimator = pickle.load(f)

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
