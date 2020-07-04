#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
C4.5算法在选取最优特征时
将信息增益比作为依据
"""

import pandas as pd
from math import log

class DTC45(object):

    def __init__(self, data):
        self.data = data
        self.ptdt = {}

    def cal_entropy(self, data):
        columns = data.columns.tolist()
        label_prob = data[columns[-1]].value_counts(1)
        entropy = sum([-1 * p * log(p, 2) for p in label_prob])
        return entropy

    def choose_best_feature(self, data):
        # 计算初始熵
        init_entropy = self.cal_entropy(data)
        entropys = {}
        columns = data.columns.tolist()
        for column in columns[:-1]:
            feature_values = data[column].unique()
            entropy = 0.0
            punish = 0.0
            for feature_value in feature_values:
                feature_data = data[(data[column] == feature_value)]
                p = len(feature_data) / len(data)
                entropy += p * self.cal_entropy(data=feature_data)
                punish += -1 * p * log(p, 2)
            if punish != 0:
                entropys[column] = (entropy - init_entropy) / punish
        sorted_entropys = sorted(entropys.items(), key=lambda x: x[1], reverse=True)
        print(sorted_entropys)
        return sorted_entropys[0][0]

    def create_decision_tree(self, data):
        # 判断是否同属一类
        columns = data.columns.tolist()
        print("columns:{}".format(columns))
        print(data)
        labels = data[columns[-1]].unique()
        if len(labels) == 1:
            return labels[0]

        ptdt = {}
        best_feature = self.choose_best_feature(data)
        print("best_feature:{}".format(best_feature))
        feature_values = data[best_feature].unique()
        ptdt[best_feature] = {}
        for feature_value in feature_values:
            sub_data = data[(data[best_feature] == feature_value)]
            del sub_data[best_feature]
            print(sub_data)
            ptdt[best_feature][str(feature_value)] = self.create_decision_tree(data=sub_data)
        return ptdt

    def run(self):
        self.ptdt = self.create_decision_tree(data=self.data)
        print(self.ptdt)

def load_data():
    data = pd.read_csv("loan.txt")
    return data

def main():
    data = load_data()
    dt = DTC45(data)
    dt.run()

if __name__ == "__main__":
    main()

"""
{'年龄': {'青年': {'信贷': {'一般': {'工作': {'否': '否', '是': '是'}}, '好': {'工作': {'否': '否', '是': '是'}}}}, '中年': {'工作': {'否': {'信贷': {'一般': '否', '好': '否', '非常好': '是'}}, '是': '是'}}, '老年': {'工作': {'否': {'信贷': {'非常好': '是', '好': '是', '一般': '否'}}, '是': '是'}}}}
"""



