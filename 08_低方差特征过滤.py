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
