"""
    @File       : 05_Tfidf文本特征提取.py
    @Description: 尝试提取关键字
    @Software   : PyCharm
    @Author     : Crisp077
    @CreateTime : 2022/9/12 16:51
"""


def Tfidf_extract(txt):
    # 导入相关包
    from sklearn.feature_extraction.text import TfidfVectorizer
    # 创建实例化对象
    transfer = TfidfVectorizer()
    # 传入文本
    extract = transfer.fit_transform(txt)
    # 特征值
    feature_names = transfer.get_feature_names()

    return extract, feature_names


if __name__ == '__main__':
    txt = ["life is short, I like Python",
           "life is not long, I like java"]
    extract, feature_names = Tfidf_extract(txt)
    print(feature_names)
    print(extract.toarray())
