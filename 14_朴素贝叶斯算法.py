"""
    @File       : 14_朴素贝叶斯算法.py
    @Description: TODO 
    @Software   : PyCharm
    @Author     : Crisp077
    @CreateTime : 2022/9/13 12:23
"""


# 流程分析：
# 1、获取数据
# 2、划分数据集
# 3、特征工程：Tfidf文本特征抽取
# 4、朴素贝叶斯算法预估器流程
# 5、模型评估


def tfidf_demo(f_x_train, f_x_test):
    # 导入相关包
    from sklearn.feature_extraction.text import TfidfVectorizer
    # 创建实例化对象
    transfer = TfidfVectorizer()
    # 传入文本
    new_x_train = transfer.fit_transform(f_x_train)
    new_x_test = transfer.transform(f_x_test)

    return new_x_train, new_x_test


def navie_bayes_demo(f_x_train, f_y_train):
    # 导包
    from sklearn.naive_bayes import MultinomialNB
    # 创建实例化对象
    estimator = MultinomialNB()
    # 传入对象
    estimator.fit(f_x_train, f_y_train)

    return estimator


if __name__ == '__main__':
    # 1、获取数据
    from sklearn.datasets import fetch_20newsgroups

    news = fetch_20newsgroups(subset="all")

    # 2、划分数据集
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)

    # 3、特征工程：Tfidf文本特征抽取
    x_train, x_test = tfidf_demo(x_train, x_test)

    # 4、朴素贝叶斯算法预估器流程
    estimator = navie_bayes_demo(x_train, y_train)

    # 5、模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值：\n", y_test == y_predict)

    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)
