#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import operator

np.set_printoptions(suppress=True)

class kNN(object):

    """
    x, y为样本数据 样本标签值
    xt, yt为测试数据 测试标签
    k:近邻参数
    mode: 0-分类 1-回归
    """
    def __init__(self, x, y, xt, yt, k=3, mode=0):
        self.x = x
        self.y = y
        self.xt = xt
        self.yt = yt
        self.k = k
        self.mode = mode

    def run(self):
        # 分类
        if self.mode == 0:
            return self.knn_classification()
        # 回归
        elif self.mode == 1:
            return self.knn_regression()

    def knn_classification(self):
        m, n = self.x.shape
        d = (np.tile(self.xt, (m, 1)) - self.x) ** 2
        dist = d.sum(axis=0)
        sorted_dist = dist.argsort()
        class_dist = {}
        for i in range(self.k):
            label = self.y[sorted_dist[i]]
            class_dist[str(label)] = class_dist.get(str(label), 0) + dist[i]
        sort_class = sorted(class_dist.items(), key=operator.itemgetter(1), reverse=True)
        cal_y = sort_class[0][0]
        return cal_y

    def knn_regression(self):
        m, n = self.x.shape
        d = (np.tile(self.xt, (m, 1)) - self.x) ** 2
        dist = d.sum(axis=0)
        sorted_dist = dist.argsort()
        total_labels = 0
        for i in range(self.k):
            total_labels += int(self.y[sorted_dist[i]])
        cal_y = total_labels / self.k
        print("cal_y:{} yt:{}".format(cal_y, self.yt))

def load_data():

    data = np.loadtxt("data.txt", delimiter=",")
    m, n = data.shape
    data[:, :-1] -= np.tile(data[:, :-1].min(0), (m, 1))
    data[:, :-1] /= np.tile(data[:, :-1].max(0), (m, 1))
    data_train = data[:900]
    data_test = data[900:]
    return data_train, data_test

def main():
    train, test = load_data()
    for t in test:
        knn = kNN(x=train[:, :-1],
                  y=train[:, -1],
                  xt=t[:-1],
                  yt=t[-1],
                  k=3,
                  mode=1)
        knn.run()


if __name__ == "__main__":
    main()

"""
代码只能说明kNN算法的逻辑实现 并不能通过data.txt中的数据有效验证算法的正确率
因为数据是随机生成的 样本各维数据与样本标签之间并没有必然的联系
因此算法的结果 并没有很强的说服力
"""