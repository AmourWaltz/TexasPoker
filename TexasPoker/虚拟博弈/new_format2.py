# -*- encoding:utf-8 -*-
"""
@Copyright: Copyright (c) 2020
@Author: JuQi
@E-mail: 964950472@qq.com 
@Title: new_format2
@Created on: 2020/2/24  12:37
@Abstract:
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import time

# 手牌种类
poker_num = 3


# 桶节点
class Node(object):
    def __init__(self, h):
        self.h = h
        tmp = np.random.rand(poker_num)
        self.dic = {'p': tmp, 'b': 1 - tmp}


# 信息集
information_list = ['_', '_p', '_b', '_pb']

# 桶队列
node_list = {}

# 生成桶队列
for info in information_list:
    node_list[info] = Node(info)

# 结束节点
z = ['_pp', '_pbp', '_pbb', '_bp', '_bb']

need_eye = (1 - np.eye(poker_num)) / (poker_num * (poker_num - 1))

# 判断胜负矩阵
judge = np.zeros((poker_num, poker_num))

for i in range(poker_num):
    for j in range(poker_num):
        if j == i:
            judge[i, j] = 0
        elif j > i:
            judge[i, j] = -1
        else:
            judge[i, j] = 1


# 游戏流程
def flow(h, loop_time, show=0):
    global node_list
    if h in z:
        # 如果游戏结束 返回结果矩阵 和误差（0）
        if len(h) % 2 == 0:
            p1 = np.reshape(node_list[h[0:-1]].dic[h[-1]], (poker_num, 1))
            p2 = node_list[h[0:-2]].dic[h[-2]]
        else:
            p1 = np.reshape(node_list[h[0:-2]].dic[h[-2]], (poker_num, 1))
            p2 = node_list[h[0:-1]].dic[h[-1]]

        probability = np.multiply(p1, p2) * need_eye

        if h == '_pp':
            return probability * judge * 2, 0
        elif h == '_pbp':
            return -probability * 2, 0
        elif h == '_pbb':
            return probability * judge * 3, 0

        elif h == '_bp':
            return probability * 2, 0
        elif h == '_bb':
            return probability * judge * 3, 0

    else:
        money_p, error_p = flow(h + 'p', loop_time, show)  # 左子树结果矩阵 和误差
        money_b, error_b = flow(h + 'b', loop_time, show)  # 右子树结果矩阵 和误差
        # 以下为找最优策略和更新策略
        nash_p = np.sum(money_p, axis=len(h) % 2) / node_list[h].dic['p']
        nash_b = np.sum(money_b, axis=len(h) % 2) / node_list[h].dic['b']
        error = 0
        if len(h) % 2 == 1:
            for i in range(poker_num):
                if nash_p[i] > nash_b[i]:
                    tmp_error = (nash_p[i] - nash_b[i]) * node_list[h].dic['b'][i]
                    node_list[h].dic['p'][i] = (node_list[h].dic['p'][i] * loop_time + 1) / (loop_time + 1)
                    if node_list[h].dic['p'][i] >= 1:
                        node_list[h].dic['p'][i] = 0.999999999
                    error += tmp_error
                else:
                    tmp_error = (nash_b[i] - nash_p[i]) * node_list[h].dic['p'][i]
                    node_list[h].dic['p'][i] = (node_list[h].dic['p'][i] * loop_time) / (loop_time + 1)
                    if node_list[h].dic['p'][i] <= 0:
                        node_list[h].dic['p'][i] = 0.000000001
                    error += tmp_error
        else:
            for i in range(poker_num):
                if nash_p[i] < nash_b[i]:
                    tmp_error = (nash_b[i] - nash_p[i]) * node_list[h].dic['b'][i]
                    node_list[h].dic['p'][i] = (node_list[h].dic['p'][i] * loop_time + 1) / (loop_time + 1)
                    if node_list[h].dic['p'][i] >= 1:
                        node_list[h].dic['p'][i] = 0.999999999
                    error += tmp_error
                else:
                    tmp_error = (nash_p[i] - nash_b[i]) * node_list[h].dic['p'][i]
                    node_list[h].dic['p'][i] = (node_list[h].dic['p'][i] * loop_time) / (loop_time + 1)
                    if node_list[h].dic['p'][i] <= 0:
                        node_list[h].dic['p'][i] = 0.000000001
                    error += tmp_error

        node_list[h].dic['b'] = 1 - node_list[h].dic['p']
        if show:
            if show == 1:
                if h == '_':
                    print(h)
                    print(nash_p)
                    print(nash_b)
            elif show == 2:
                print(h)
                print(nash_p)
                print(nash_b)

        # 结果矩阵乘以手牌概率并且返回
        if len(h) > 2:
            if len(h) % 2 == 1:
                return np.multiply(money_p + money_b,
                                   np.reshape(node_list[h[0:-2]].dic[h[-2]], (poker_num, 1))), error + error_p + error_b
            else:
                return np.multiply(money_p + money_b, node_list[h[0:-2]].dic[h[-2]]), error + error_p + error_b

        else:
            return money_p + money_b, error + error_p + error_b


if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True)
    iteration = 10000
    sum_x = np.zeros(iteration)

    # start = time.time()
    for loop_i in range(iteration):
        _, sum_x[loop_i] = flow('_', loop_i + 1, show=0)

    # end = time.time()

    # print("Execution Time: ", end - start)
    # print(x[-1])
    np.savetxt('New3_1.csv', sum_x)
    # print(sum_x / 500)
