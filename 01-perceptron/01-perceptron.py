#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
感知机Perceptron
Author:Jackie
Date:2020-06-17
Email:jackie_programmer@126.com
"""

import numpy as np

np.set_printoptions(suppress=True)

class Perceptron(object):

    def __init__(self, data, alph, iter=50):
        """
        :param data: 训练数据
        :param alph: 更新步长
        :param iter: 迭代次数
        """
        self.x = data[:, 1:]
        self.y = data[:, 0]
        # 标签数据中 >5为1 <5为-1
        self.y[self.y > 5] = 1
        self.y[self.y < 5] = -1
        self.alph = alph
        self.iter = iter
        m, n = self.x.shape
        self.w = np.zeros((1, n))
        self.b = 0
        print("x:{}".format(self.x.shape))
        print("y:{}".format(len(self.y)))

    def run(self):
        print("perceptron training.....")
        # 训练迭代次数
        for k in range(self.iter):
            print("iter:{}/{}".format(k+1, self.iter))
            # 数据集中的每一行 判断x,y是否是误分类点
            for i, x in enumerate(self.x):
                # 误分类点判断条件 -1*y*(w*x+b)>=0
                if -1 * self.y[i] * (self.w.dot(x) + self.b) >= 0:
                    # 更新w, b
                    self.w = self.w + self.alph * self.y[i] * x
                    self.b = self.b + self.alph * self.y[i]
        print("perceptron train done")
    
    def test(self, data):
        print('perceptron testing.....')
        x = data[:, 1:]
        y = data[:, 0]
        y[y > 5] = 1
        y[y < 5] = -1
        error = 0
        for i, d in enumerate(x):
            if -1 * y[i] * (self.w.dot(d) + self.b) >= 0:
                error += 1
        print('total:{} error:{}'.format(len(x), error))
        print('perceptron test done')
        return (len(x) - error) / len(x)

def load_data(path):
    data = np.loadtxt(path, delimiter=",")
    return data

def main():
    train_data_path = "../00-data-set/train-data.txt"
    test_data_path = "../00-data-set/test-data.txt"
    train_data = load_data(path=train_data_path)
    print('----------------------')
    print('perceptron init...')
    perceptron = Perceptron(data=train_data, alph=0.0001, iter=30)
    print('perceptron init done')
    print('----------------------')
    perceptron.run()
    test_data = load_data(path=test_data_path)
    result = perceptron.test(data=test_data)
    print('result:{}'.format(result))

if __name__ == "__main__":
    main()

"""
数据集是从该博主的博客http://www.pkudodo.com/2018/11/18/1-4/中获得
博客中最终的准确率是81.72%
程序运行最终准确率是95.61%
"""
