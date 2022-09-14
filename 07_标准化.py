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
