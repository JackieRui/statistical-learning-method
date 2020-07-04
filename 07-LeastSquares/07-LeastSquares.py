#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

np.set_printoptions(suppress=True)

"""
LeastSquares 最小二乘法
利用矩阵的方式 来直接求取参数
原来就是最小化均方差 来求取极值
alph 最终等于(X的转制*X)的逆矩阵 * Y的转制 * X

"""

class LS(object):

    def __init__(self, data):
        self.data = data

    def run(self):
        x = self.data[:, :-1]
        y = self.data[:, -1]
        x_1 = np.linalg.pinv(np.dot(x.T, x))
        alph = np.dot(x_1, y.T)
        result = np.dot(alph, x)
        print(result)

def load_data():
    data = np.loadtxt("data.txt", delimiter=",")
    m, n = data.shape
    data[:, :-1] -= np.tile(data[:, :-1].min(0), (m, 1))
    data[:, :-1] /= np.tile(data[:, :-1].max(0), (m, 1))
    return data

def main():
    data = load_data()
    least_squares = LS(data=data)
    least_squares.run()

if __name__ == "__main__":
    main()




