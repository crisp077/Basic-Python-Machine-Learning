"""
    @File       : 10_主成分分析.py
    @Description: TODO 
    @Software   : PyCharm
    @Author     : Crisp077
    @CreateTime : 2022/9/12 20:42
"""


def PCA_demo(f_data, f_component):
    # 导入库
    from sklearn.decomposition import PCA
    # 实例化对象
    PCA_instant = PCA(n_components=f_component)
    # 传入数据
    transfer = PCA_instant.fit_transform(f_data)

    return transfer


if __name__ == '__main__':
    data = [
        [2, 8, 4, 5],
        [6, 3, 0, 8],
        [5, 4, 9, 1]
    ]
    new_data = PCA_demo(data, 2)

    print(new_data)
