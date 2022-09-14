"""
    @File       : 02_字典特征提取.py
    @Description: TODO 
    @Software   : PyCharm
    @Author     : Crisp077
    @CreateTime : 2022/9/12 14:46
"""


def dict_extract(dict):
    # 导入sklearn中的特征提取库
    from sklearn.feature_extraction import DictVectorizer
    # 实例化一个转换器类 取消sparse稀疏矩阵表示
    DictVectorizer = DictVectorizer(sparse=False)
    # 传入字典 返回独热编码 one hot
    extract = DictVectorizer.fit_transform(dict)
    # 转换为原来的数据格式
    origin_data = DictVectorizer.inverse_transform(extract)
    # 返回类别名称
    feature_names = DictVectorizer.get_feature_names()
    return extract, origin_data, feature_names


if __name__ == '__main__':
    dict = [
        {'city': '北京', 'temperature': 28},
        {'city': '上海', 'temperature': 36},
        {'city': '武汉', 'temperature': 38}
    ]
    extract, origin_data, feature_names = dict_extract(dict)
    print(extract)
    print(origin_data)
    print(feature_names)
