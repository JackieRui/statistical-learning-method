#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
SVM 被称为分类效果最好的算法
"""
import random
import numpy as np

np.set_printoptions(suppress=True)

class SVM(object):

    def __init__(self, x, y, C, max_iter):
        self.x = x
        self.y = y
        self.C = C
        self.alph = np.zeros(self.x.shape[0])
        self.max_iter = max_iter

    # w = ∑alph * y * x
    def weight(self):
        w = np.zeros(self.x.shape[1])
        alph_y = self.alph * self.y
        for i in range(self.x.shape[0]):
            w += alph_y[i] * self.x[i]
        return w

    # g(x) = wx + b
    def g(self, x, b):
        w = self.weight()
        return w.dot(x) + b

    def E(self, x, b, y):
        return self.g(x, b) - y

    # 选择alph1
    def choose_alph_1(self):
        # 优先选择不满足KKT条件的alph
        for i in range(len(self.alph)):
            r = self.y[i] * self.g(self.x[i], b=0)

            if 0 < self.alph[i] < self.C and r != 1:
                return i
            elif self.alph[i] == 0 and r < 1:
                return i
            elif self.alph[i] == self.C and r > 1:
                return i
            else:
                return -1

    # 随机选择alph1
    def choose_alph_2(self, alph1):
        while True:
            j = random.choice(range(len(self.alph)))
            if j != alph1:
                return j

    def adjust_alph_2(self, alph2, L, H):
        if alph2 <= L:
            return L
        elif alph2 >= H:
            return H
        else:
            return alph2

    def run(self):
        iter_cnt = 0
        while iter_cnt < self.max_iter:
            print("iter_cnt:{}".format(iter_cnt))
            i = self.choose_alph_1()
            j = self.choose_alph_2(alph1=i)
            alph1 = self.alph[i]
            alph2 = self.alph[j]
            x1 = self.x[i]
            x2 = self.x[j]
            y1 = self.y[i]
            y2 = self.y[j]
            # 确定范围
            if y1 == y2:
                L = max([alph1 + alph2 - self.C, 0])
                H = min([alph1 + alph2, self.C])
            else:
                L = max([0, alph2 - alph1])
                H = min([self.C, self.C - alph1 + alph2])
            alph2_new = alph2 + y2 * (self.E(x=x1, b=0, y=y1) - self.E(x=x2, b=0, y=y2)) / (x1.dot(x1) - 2*x1.dot(x2) + x2.dot(x2))
            # 调整参数
            alph2_new = self.adjust_alph_2(alph2=alph2_new, L=L, H=H)
            # alph1_new
            alph1_new = alph1 + y1 * y2 * (alph2 - alph2_new)
            # 更新alph
            self.alph[i] = alph1_new
            self.alph[j] = alph2_new

def load_data():
    data = np.loadtxt("data.txt")
    return data

def main():
    data = load_data()
    svm = SVM(x=data[:, :-1], y=data[:, -1], C=3, max_iter=300)
    svm.run()

if __name__ == "__main__":
    main()

"""
代码实现是伪码形式 不能直接运行啊
关键在于图片的推导 以及里面包含的求解思想
"""