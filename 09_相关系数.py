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
