#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

class KMeans(object):

    def __init__(self, data, K, max_iter):
        self.K = K
        self.data = data
        self.center = []
        self.max_iter = max_iter

    def calculate_distince(self, center, data):
        return sum([(center[i] - data[i]) ** 2 for i in range(len(center))])

    def run(self):
        # print(self.data)
        # 随机选取K个中心
        index = np.random.choice(len(self.data), self.K)
        # print(index)
        self.center = self.data[index]
        flag = True
        for i in range(self.max_iter):
            kinds = {}
            for line in self.data:
                # 计算到中心的距离
                distences = [self.calculate_distince(center=cnt, data=line) for cnt in self.center]
                # print(distences)
                sort_index = np.array(distences).argsort()
                if str(sort_index[0]) not in kinds:
                    kinds[str(sort_index[0])] = []
                # 分属到不同的类别
                kinds[str(sort_index[0])].append(line)
            # 重新计算K个中心
            new_center = [0] * self.K
            for k, v in kinds.items():
                new_center[int(k)] = sum(v)/len(v)

            # 判断中心是否变化
            for i in range(self.K):
                if self.calculate_distince(center=new_center[i], data=self.center[i]) > 0.001:
                    flag = False
                    break
            self.center = new_center
            if flag:
                break

        print(self.center)

def load_data():
    data = np.loadtxt("data.txt", delimiter=",")
    # print(data)
    return data

def main():
    data = load_data()
    k_means = KMeans(data=data, K=2, max_iter=100)
    k_means.run()

if __name__ == "__main__":
    main()

"""
K-means聚类算法:
1.首先随机从数据中选取K个中心点
2.计算数据集到K个中心点的距离
3.选出K个距离中最小的一个中心点 归结到一个类别中
4.重新计算集合类别中的中心点
5.判断新的中心点和之前的中心点是否在阈值范围内 如果在 停止迭代 否则重复2，3，4，5
"""
