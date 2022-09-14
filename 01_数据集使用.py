"""
    @File       : 01_数据集使用.py
    @Description: TODO 
    @Software   : PyCharm
    @Author     : Crisp077
    @CreateTime : 2022/9/11 16:43
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def datasets_demo():
    """
    sklearn数据集使用
    :return:
    """
    # 获取数据集
    iris = load_iris()
    print("鸢尾花数据集：\n", iris)
    print("查看数据集描述：\n", iris["DESCR"])

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)

    return None


if __name__ == '__main__':
    # 代码1：sklearn数据集使用
    datasets_demo()
