#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
EM算法 统计学习中 假如观测到的值由x决定
并且x由z来决定 求解的问题是由含有隐含变量的概率

"""

class EM(object):

    def __init__(self, PI, p, q, data, alph, max_iter=100):
        self.PI = PI
        self.p = p
        self.q = q
        self.data = data
        self.alph = alph
        self.max_iter = max_iter

    def run(self):
        for k in range(self.max_iter):
            values = []
            for label in self.data:
                value_sum = self.PI * (self.p ** label) * ((1-self.p) ** (1-label)) + (1 - self.PI) * (self.q ** label) * ((1 - self.q) ** (1 - label))
                value = self.PI * (self.p ** label) * ((1-self.p) ** (1-label)) / value_sum
                values.append(value)

            sum1 = 0.0
            for i in values:
                sum1 += i
            PI = sum1 / len(values)

            sum1 = 0.0
            sum2 = 0.0
            for i in range(len(values)):
                sum1 += values[i] * self.data[i]
                sum2 += values[i]
            p = sum1 / sum2

            sum1 = 0.0
            sum2 = 0.0
            for i in range(len(values)):
                sum1 += (1 - values[i]) * values[i]
                sum2 += (1 - values[i])
            q = sum1 / sum2
            if abs(self.PI - PI) <= self.alph and abs(self.p - p) <= self.alph and abs(self.q - q) <= self.alph:
                print("break")
                break
            else:
                self.PI = PI
                self.p = p
                self.q = q
        print("PI:{} p:{} q:{}".format(self.PI, self.p, self.q))


def main():
    data = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
    em = EM(PI=0.5, p=0.5, q=0.5, data=data, alph=0.001, max_iter=100)
    em.run()

if __name__ == "__main__":
    main()

"""
EM算法大致的流程：
(E步)通过p,q以及y来估计Z的分布
(M步)然后再利用Z的分布来更新p,q
M步中更新p,q这步操作没搞懂(三个硬币的操作) 等学习完HMM后再来看看这步操作
"""