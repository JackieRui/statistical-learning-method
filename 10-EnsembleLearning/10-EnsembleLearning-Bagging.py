#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
Ensembel Learning 集成学习
Bagging-Bootstrap Aggregating,引导聚集算法
具体算法流程 数据集D
1.从数据集D中随机选取m个样本数据 组成训练数据Di
2.对训练数据Di 应用弱分类器进行分类 测试数据T得到测试结果Ri
3.重复1,2步 t次 得到测试结果数据集R
4.对于测试结果数据集R 采用多数表决法来作为最终的结果
"""

import numpy as np
import pandas as pd
from kNN import kNN

class Bagging(object):

    # 重复t次 每次随机取m个样本
    def __init__(self, data, t, m):
        self.data = data
        self.t = t
        self.m = m

    def run(self):
        # 总测试数据
        for test_data in self.data[900:, :]:
            xt = test_data[:-1]
            yt = test_data[-1]
            # 存储t次的分类结果
            bags = {}
            for i in range(self.t):
                # 随机选取m个训练数据
                random_index = np.random.choice(range(900), self.m)
                train_data = self.data[random_index]
                x = train_data[:, :-1]
                y = train_data[:, -1]
                knn = kNN(x=x, y=y, xt=xt, yt=yt, k=3, mode=0)
                label = knn.run()
                if str(label) not in bags:
                    bags[str(label)] = 0
                bags[str(label)] += 1
            print(bags)
            sort_bags = sorted(bags.items(), key=lambda x:x[1], reverse=True)
            print("result:{} yt:{}".format(sort_bags[0][0], yt))

def load_data():
    data = np.loadtxt("data.txt", delimiter=",")
    m, n = data.shape
    data[:, :-1] -= np.tile(data[:, :-1].min(0), (m, 1))
    data[:, :-1] /= np.tile(data[:, :-1].max(0), (m, 1))
    return data

def main():
    data = load_data()
    bag = Bagging(data=data, t=10, m=300)
    bag.run()

if __name__ == "__main__":
    main()

"""
Bagging:集成学习的一种算法 从训练数据集中放回式随机采样m次 应用弱分类器对测试数据进行分类
统计m次的结果 选择种类最多的标签作为最终的测试结果
随机森林-Random Forest: Bagging with Decision Tree
对采样数据应用决策树算法进行分类
特点: 随机采样的数据 以及本次采样的测试结果与下一轮过程无关
"""





