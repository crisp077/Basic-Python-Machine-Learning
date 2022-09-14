"""
    @File       : 04_中文文本特征提取.py
    @Description: TODO 
    @Software   : PyCharm
    @Author     : Crisp077
    @CreateTime : 2022/9/12 15:49
"""


def cut_chiesewords(txt):
    # 导入jieba库
    import jieba
    new_txt = " ".join(list((jieba.cut(txt))))

    return new_txt


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
    texts = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
             "我们看到的从很远圣系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
             "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    # 将new_txt初始化为列表方便添加
    new_txt = []
    for txt in texts:
        # 对中文进行分词
        new_txt.append(cut_chiesewords(txt))
        # 文本特征提取
        extract, origin_txt, feature_names = txt_extract(new_txt)
        print(feature_names)
        print(extract.toarray())
