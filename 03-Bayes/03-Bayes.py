#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
贝叶斯定理
根据已有的样本数据 计算目标数据归为某类的概率
根据已观测到的数据特征 推测新的样本数据的概率
p(A|B) = p(A)*p(B|A)/P(B)      A:类别 B:标签样本
p(A|B) 后验概率   p(B|A) 先验概率   p(A) 似然值   p(B) 全概率公式
"""

import numpy as np
import pandas as pd
from functools import reduce
from collections import Counter

np.set_printoptions(suppress=True)

class Bayes(object):

    def __init__(self, data):
        self.data = data
        # 标签值
        self.y = self.data[:, 0]
        # 向量数据 >128置为1 <128置为0
        self.x = self.data[:, 1:]
        self.x[self.x < 128] = 0
        self.x[self.x >= 128] = 1
    
    def run(self):
        """计算各初始概率值"""
        cls_number = 10
        # 计算标签概率
        self.Py = np.zeros((cls_number, 1))
        for i in range(cls_number):
            self.Py[i] = (np.sum(self.y[self.y == i]) + 1) / (len(self.y) + 10)
        # 求取对应的log形式 防止值过小溢出
        self.Py = np.log(self.Py)
        # 计算不同标签值下 x对应特征的概率
        m, n = self.x.shape
        # 标签数 x特征维数 特征取值数
        self.Px = np.zeros((cls_number, n, 2))
        # 标签数
        for i in range(m):
            label = int(self.y[i])
            # 每行的特征数
            for j in range(n):
                v = int(self.x[i][j])
                self.Px[label][j][v] += 1
        # 统计完成后 计算概率
        for i in range(cls_number):
            for j in range(n):
                v0 = self.Px[i][j][0]
                v1 = self.Px[i][j][1]
                self.Px[i][j][0] = (v0 + 1) / (v0 + v1 + 2)
                self.Px[i][j][1] = (v1 + 1) / (v0 + v1 + 2)
    
    def test(self, test_data):
        """训练集测试"""
        # 训练数据初始化
        ty = test_data[:, 0]
        tx = test_data[:, 1:]
        tx[tx < 128] = 0
        tx[tx >= 128] = 1
        
        # 记录争取的值
        correct = 0
        m, n = tx.shape
        cls_number = len(self.Py)
        # m行数据
        for i in range(m):
            # 可能属于cls_number的概率值
            P = [0] * cls_number
            # 不同的标签
            for j in range(cls_number):
                txp = [self.Px[j][k][int(tx[i][k])] for k in range(n)]
                P[j] = sum(txp) + self.Py[j]
            # 最大的标签概率值
            cal_p = P.index(max(P))
            if int(cal_p) == int(ty[i]):
                correct += 1
        
        return correct / m

def load_data(path):
    data = np.loadtxt(path, delimiter=",")
    return data

def main():
    train_data_path = "../00-data-set/train-data.txt"
    test_data_path = "../00-data-set/test-data.txt"
    print("--------------------")
    print("load train data:")
    train_data_01 = load_data(path=train_data_path)
    print("load train data done")
    bayes = Bayes(data=train_data_01)
    bayes.run()
    print("load test data:")
    test_data = load_data(path=test_data_path)
    print("load test data done")
    result = bayes.test(test_data=test_data)
    print("result:{}".format(result))

if __name__ == "__main__":
    main()

"""
数据集是从该博主的博客https://www.pkudodo.com/2018/11/21/1-3/中获得
博主运行最终正确率：84.3% 运行时长：103s
本程序运行结果58.08% 时间超过103s
"""





