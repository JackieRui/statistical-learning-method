#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
K近邻 kNN
Author:Jackie
Date:2020-06-17
Email:jackie_programmer@126.com
"""

import numpy as np
import operator

np.set_printoptions(suppress=True)

class kNN(object):

    def __init__(self, data, k=3):
        """
        :param data: 样本数据
        :param k: K个候选值
        """
        self.x = data[:, 1:]
        self.y = data[:, 0]
        self.k = k

    def run(self, tx):
        m, n = self.x.shape
        # print("x m:{} n:{}".format(m, n))
        """
        np.tile(tx, (m, 1))
        生成矩阵 m行1列 与self.x的同行同列的矩阵
        矩阵之间减法后平方 相当于(tx1-x1)**2+(tx2-x2)**2 计算欧式距离
        d.sum(axis=0) 按行求和
        """
        d = (np.tile(tx, (m, 1)) - self.x) ** 2
        dist = d.sum(axis=1)
        sorted_dist = dist.argsort()
        class_dist = {}
        """
        numpy中的argsort()对矩阵中的数据进行排序 提取排序完成后的索引值
        注意：sorted_dist对应的是排序完成后原数列的索引值
        class_dist对应的是某类别的距离总和
        这样的好处：可以避免因为单纯计数造成不同类别但是数值相同的情况
        """
        # print(sorted_dist)
        for i in range(self.k):
            label = self.y[sorted_dist[i]]
            class_dist[str(label)] = class_dist.get(str(label), 0) + dist[i]
        # sort_class对应的是元组(cls, dist)
        sort_class = sorted(class_dist.items(), key=operator.itemgetter(1), reverse=True)
        """
        整个计算过程采用了numpy矩阵运算 相对来说比逐行计算要快
        """
        # print(sort_class[0][0])
        return sort_class[0][0]
    
    def test(self, data):
        train_x = data[:, 1:]
        train_y = data[:, 0]
        accuracy = 0
        for i, tx in enumerate(train_x):
            ry = self.run(tx=tx)
            # print('ry:{} train_y:{}'.format(type(ry), type(train_y[i])))
            if float(ry) == train_y[i]:
                accuracy += 1
                print("accuracy:{} total:{}".format(accuracy, len(train_x)))
        return accuracy / len(train_x)

def load_data(path):
    data = np.loadtxt(path, delimiter=",")
    return data

def main():
    train_data_path = "../00-data-set/train-data.txt"
    test_data_path = "../00-data-set/test-data.txt"
    print("--------------------")
    print("load train data:")
    train_data = load_data(path=train_data_path)
    knn = kNN(data=train_data, k=3)
    print("load train data done")
    print("--------------------")
    print("load test data:")
    test_data = load_data(path=test_data_path)
    print("load test data done")
    print("--------------------")
    print("test kNN......")
    result = knn.test(data=test_data)
    print("result:{}".format(result))

if __name__ == "__main__":
    main()

"""
数据集是从该博主的博客https://www.pkudodo.com/2018/11/19/1-2/中获得
博主采用的k值为25 运行时间是308s 正确率是97%
我采用k值是3 正确率是97.04% 时间远远超过308s
"""