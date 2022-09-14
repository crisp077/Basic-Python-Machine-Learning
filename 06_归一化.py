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
