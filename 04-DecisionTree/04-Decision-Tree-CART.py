#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
CART算法 分类树采用基尼指数来作为划分的依据
Gini(D) = 1-∑(Ck/D)**2 Ck为样本类别数
Gini(D, A) = ∑(Ai/D)Gini(D, Ai)
如果特征A有多个属性值 则需要特征值组合划分
CART算法是二叉树的分类
"""

import pandas as pd

class DTCART(object):

    def __init__(self, data):
        self.data = data

    # Gini指数
    def cal_gini(self, data):
        columns = data.columns.tolist()
        label_pro = data[columns[-1]].value_counts(1)
        gini = 1 - sum([p ** 2 for p in label_pro])
        return gini

    # 集合中有多特征时 选择最优特征的某个特征值
    def choose_best_feature(self, data):
        columns = data.columns.tolist()
        ginis = {}
        for column in columns[:-1]:
            feature_values = data[column].unique()
            for feature_value in feature_values:
                # 按照特征A 特征值a划分后的数据集
                left = data[(data[column] == feature_value)]
                right = data[(data[column] != feature_value)]
                gini = len(left) / len(data) * self.cal_gini(data=left) + \
                       len(right) / len(data) * self.cal_gini(data=right)
                key = "{}@{}".format(column, feature_value)
                ginis[key] = gini
        sort_ginis = sorted(ginis.items(), key=lambda x: x[1], reverse=False)
        print(sort_ginis)
        return sort_ginis[0][0]

    def majority_labels(self, data):
        columns = data.columns.tolist()
        labels = self.data[columns[-1]].unique()
        label_cnt = self.data[columns[-1]].value_counts(1)
        prob = dict(zip(labels, [label_cnt[label] for label in labels]))
        sort_prob = sorted(prob.items(), key=lambda x:x[1], reverse=True)
        return sort_prob[0][0]

    def create_decision_tree(self, data):
        columns = data.columns.tolist()
        labels = data[columns[-1]].unique()
        # 同属一个类别
        if len(labels) == 1:
            return labels[0]

        # 单个特征值 多类别
        if len(columns) == 2 and len(data[columns[0]].unique()) == 1:
            return self.majority_labels(data=data)

        ptdt = {}
        # 特征@特征值
        best_choice = self.choose_best_feature(data=data)
        feature, value = best_choice.split("@")
        ptdt[feature] = {}
        left = data[(data[feature] == value)]
        del left[feature]
        ptdt[feature][value] = self.create_decision_tree(data=left)
        right = data[(data[feature] != value)]
        ptdt[feature]["!{}".format(value)] = self.create_decision_tree(data=right)
        return ptdt

    def run(self):
        ptdt = self.create_decision_tree(data=self.data)
        print(ptdt)

def load_data():
    data = pd.read_csv("loan.txt")
    return data

def main():
    data = load_data()
    dt = DTCART(data=data)
    dt.run()

if __name__ == "__main__":
    main()

"""
{'房子': {'否': {'工作': {'否': '否', '!否': '是'}}, '!否': '是'}}
TODO: 
在选择最佳特征A-特征值a之后 依据特征A的特征值a进行原始数据集划分
划分后的子数据集 还需判断特征A 是否还有多特征值
如果特征A的特征值唯一 则在子数据集Gini计算中 不再考虑特征A
否则 还需计算特征A的Gini系数
关于特征有多个特征值 还需完善
"""

