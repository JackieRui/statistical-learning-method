#! /usr/bin/python

"""
逻辑斯特回归
二分类的情况下 通过sigmoid函数来计算属于哪个类别的概率
对似然函数取对数 求取似然函数的极大值 对似然函数求导
使用梯度上升法来求解函数极大值 迭代求解参数w
"""

import numpy as np

np.set_printoptions(suppress=True)

class LR(object):

    def __init__(self, data, alph, max_iter):
        self.data = data
        m, n = self.data.shape
        self.w = np.ones((n-1, ))
        print("w:{}".format(self.w))
        # 步长
        self.alph = alph
        self.max_iter = max_iter

    def sigmoid(self, x):
        return 1/(1+np.exp(-1 * x.dot(self.w.T)))

    def cal_gradient_descent(self):
        x = self.data[:, :-1]
        y = self.data[:, -1]
        # print(x.dot(self.w.T))
        error = y - self.sigmoid(x)
        gd = error.dot(x)
        return gd

    def run(self):
        for i in range(self.max_iter):
            self.w += self.alph * self.cal_gradient_descent()
        print("w:{}".format(self.w))


def load_data():
    data = np.loadtxt("data.txt")
    return data

def main():
    data = load_data()
    lr = LR(data=data, alph=0.3, max_iter=200)
    lr.run()

if __name__ == "__main__":
    main()

"""
w:[29.3493868  22.94629116]
代码流程: 本代码所采用的是批量梯度上升来计算似然函数的最大值
核心计算是在计算函数梯度上 在利用numpy进行向量计算时 先少量数据来保证计算流程
然后再批量数据测试

运行时异常:
08-Logistic Regression.py:30: RuntimeWarning: overflow encountered in exp
解决方案:
def sigmoid(inx):
    if inx>=0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0/(1+exp(-inx))
    else:
        return exp(inx)/(1+exp(inx))
"""










