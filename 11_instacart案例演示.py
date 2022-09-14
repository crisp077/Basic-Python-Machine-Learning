"""
    @File       : 11_instacart案例演示.py
    @Description: TODO 
    @Software   : PyCharm
    @Author     : Crisp077
    @CreateTime : 2022/9/12 20:54
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
    import pandas as pd

    # 1、获取数据
    order_products = pd.read_csv("instacart/order_products__prior.csv")
    products = pd.read_csv("instacart/products.csv")
    orders = pd.read_csv("instacart/orders.csv")
    aisles = pd.read_csv("instacart/aisles.csv")

    # 2、合并表
    tab1 = pd.merge(aisles, products, on=["aisle_id", "aisle_id"])
    tab2 = pd.merge(tab1, order_products, on=["product_id", "product_id"])
    tab3 = pd.merge(tab2, orders, on=["order_id", "order_id"])

    # 3、找到user_id和aisle之间的关系
    table = pd.crosstab(tab3["user_id"], tab3["aisle"])

    # 4、PCA降维
    new_data = PCA_demo(table, 0.95)

    print(new_data)
