#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
AdaBoost算法实现思想
有训练数据集D 初始化训练数据集每条数据的权重值w为1/N 执行m轮学习处理 每一轮执行的操作如下:
1.在权重值为1/N的训练数据集上得到分类器Gm
2.计算分类器Gm的误差率为em 其中em=∑wI(Gm!=ym)
根据误差率来计算分类器Gm的权重 alph = 1/2*log((1-em)/em)
alph代表Gm在最终的组合分类器中的权重
3.根据alph来更新训练数据集D的权重
wm+1 = wm/z* exp(-alph * yi * Gi)
其中z=∑wm * exp(-alph * yi * Gi)

程序中假设两个分类器 knn1 knn2 分类器有100条数据作为模型内部参数
另外单独有100条数据来作为测试数据 作为AdaBoost算法的训练数据
"""

import operator
import numpy as np
from math import log

np.set_printoptions(suppress=True)

class kNN(object):

    """
    x, y为样本数据 样本标签值
    xt, yt为测试数据 测试标签
    k:近邻参数
    mode: 0-分类 1-回归
    """
    def __init__(self, x, y, k=3):
        self.x = x
        self.y = y
        self.k = k

    def knn_classification(self, xt):
        m, n = self.x.shape
        d = (np.tile(xt, (m, 1)) - self.x) ** 2
        dist = d.sum(axis=0)
        sorted_dist = dist.argsort()
        class_dist = {}
        for i in range(self.k):
            label = self.y[sorted_dist[i]]
            class_dist[str(label)] = class_dist.get(str(label), 0) + dist[i]
        sort_class = sorted(class_dist.items(), key=operator.itemgetter(1), reverse=True)
        return int(float(sort_class[0][0]))

class Adaboost(object):

    # 测试数据data ks分类器集合
    def __init__(self, data):
        self.data = data
        k1 = kNN(x=self.data[:100, :-1], y=self.data[:100, -1])
        k2 = kNN(x=self.data[100:200, :-1], y=self.data[100:200, -1])
        self.ks = [k1, k2]
        # 分类器权重
        self.k_weight = [1, 1]
        # 训练数据权重
        self.train_data = self.data[200:300, :]
        self.d_weight = np.array([0.01] * len(self.train_data))

    def run(self):
        # 第i轮训练
        for i, ki in enumerate(self.ks):
            print("train number:{}".format(i + 1))
            # 计算ki的训练结果
            g = np.array([ki.knn_classification(xt=td[:-1]) for td in self.train_data])
            verify = np.array(self.train_data[:, -1] == g)
            # 取反操作
            e = sum(self.d_weight[~verify])
            alph = 1/2*log((1-e)/e)
            # 计算ki的权重
            self.k_weight[i] = alph
            # 更新训练参数的权重
            z = sum(self.d_weight * np.exp(np.array([-1 * alph] * len(g)) * self.train_data[:, -1] * g))
            self.d_weight = self.d_weight * np.exp(np.array([-1 * alph] * len(g)) * self.train_data[:, -1] * g) / z
        print(self.k_weight)

def load_data():
    data = np.loadtxt("data.txt", delimiter=",")
    m, n = data.shape
    data[:, :-1] -= np.tile(data[:, :-1].min(0), (m, 1))
    data[:, :-1] /= np.tile(data[:, :-1].max(0), (m, 1))
    return data

def main():
    data = load_data()
    boost = Adaboost(data=data)
    boost.run()


if __name__ == "__main__":
    main()

"""
测试结果：[-0.1003353477310761, 0.0768086424831723]
代码中只是实现Adaboost的逻辑 数据以及结果没有说服力
代码构造了两个由kNN实现的分类器k1, k2 只不过其固有的数据不同 前100个 随后的100个
然后取200-300的数据作为Adaboost的训练数据 迭代执行下述步骤:
1.计算分类器i的误差
2.计算分类器系数alphi
3.更新训练数据的权重
最终得到的结果叠加 就是分类器的结果
f(x) = alph1 * k1 + alph2 * k2
"""












