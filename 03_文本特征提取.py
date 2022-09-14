"""
    @File       : 03_文本特征提取.py
    @Description: TODO 
    @Software   : PyCharm
    @Author     : Crisp077
    @CreateTime : 2022/9/12 15:14
"""


def txt_extract(txt):
    # 导入文本特征提取库
    from sklearn.feature_extraction.text import CountVectorizer
    # 实例化转换器
    transfer = CountVectorizer()
    # 使用fit_transform
    extract = transfer.fit_transform(txt)
    # 转换为原来格式
    origin_data = transfer.inverse_transform(extract)
    # 返回文本特征类型
    feature_names = transfer.get_feature_names()

    return extract, origin_data, feature_names


if __name__ == '__main__':
    txt = ["life is short, I like Python",
           "life is not long, I like java"]
    extract, origin_data, feature_names = txt_extract(txt)
    print(origin_data)
    print(feature_names)
    print(extract.toarray())
