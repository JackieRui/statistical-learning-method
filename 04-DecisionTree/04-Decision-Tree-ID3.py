#! /usr/bin/python

"""
决策树
熵:随机变量不确定性的度量
信息增益:按照特征A分类后 前者和后者的熵之差
按照信息增益最大的特征来对集合进行分类 逐渐使集合趋于稳定
"""

import pandas as pd
from math import log

class DTID3(object):

    def __init__(self, data):
        self.init_data = data
        # 构造的决策树
        self.ptdt = {}
        self.iter = 0

    def cal_entropy(self, data):
        columns = data.columns.tolist()
        label_prob = data[columns[-1]].value_counts(1)
        entropy = sum([-1 * p * log(p, 2) for p in label_prob])
        return entropy

    def choose_best_feature(self, data):
        # 集合初始熵
        init_entropy = self.cal_entropy(data)
        entropys = {}
        columns = data.columns.tolist()
        for column in columns[:-1]:
            feature_values = data[column].unique()
            entropy = 0.0
            for feature_value in feature_values:
                feature_data = data[(data[column] == feature_value)]
                entropy += len(feature_value) / len(data) * self.cal_entropy(data=feature_data)
            entropys[column] = init_entropy - entropy
        sort_entropys = sorted(entropys.items(), key=lambda x: x[1], reverse=True)
        print(sort_entropys)
        return sort_entropys[0][0]

    def create_decision_tree_ID3(self, data):
        # 判断当前类别 如果同属一类 就返回
        columns = data.columns.tolist()
        print("columns:{}".format(columns))
        labels = data[columns[-1]].unique()
        if len(labels) == 1:
            print(data)
            return labels[0]

        ptdt = {}
        # 选择最优特征
        best_feature = self.choose_best_feature(data)
        print("best_feature:{}".format(best_feature))
        feature_values = data[best_feature].unique()
        ptdt[best_feature] = {}
        for feature_value in feature_values:
            # 子数据集构造决策树
            sub_data = data[(data[best_feature] == feature_value)]
            # 删除该列属性 该特征同属一类
            del sub_data[best_feature]
            ptdt[best_feature][str(feature_value)] = self.create_decision_tree_ID3(data=sub_data)
        return ptdt

    def run(self):
        self.ptdt = self.create_decision_tree_ID3(data=self.init_data)
        print(self.ptdt)

def load_data():
    data = pd.read_csv("loan.txt")
    return data

def main():

    data = load_data()
    dt = DTID3(data)
    dt.run()

if __name__ == "__main__":
    main()


