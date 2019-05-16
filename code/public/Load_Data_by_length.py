#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:       Qiang Cui:  <cuiqiang1990[at]hotmail.com>
# Descripton:   load the dataset
# Create Date:  2016-12-05 10:00:00
# Modify Date:  2016-00-00 00:00:00
# Modify Disp:

from __future__ import print_function
import time
import numpy as np
import pandas as pd
import random
from collections import Counter
import math
from copy import deepcopy
import glob
import json
from numpy.random import binomial
__docformat__ = 'restructedtext en'


def exe_time(func):
    def new_func(*args, **args2):
        t0 = time.time()
        print("-- @%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__))
        back = func(*args, **args2)
        print("-- @%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__))
        print("-- @%.3fs taken for {%s}" % (time.time() - t0, func.__name__))
        return back
    return new_func


def load_data(dataset, mode, split):
    """
    加载购买记录文件，生成数据。
    """
    # 用户购买历史记录，原纪录. 嵌套列表, 元素为一个用户的购买记录(小列表)
    print('Original data ...')
    buys = pd.read_csv(dataset, sep=' ')
    all_user_buys = [[str(i) for i in ubuys.split(',')] for ubuys in buys['buys']]  # 用str类型。
    all_trans = [item for ubuys in all_user_buys for item in ubuys]
    tran_num, user_num, item_num = len(all_trans), len(all_user_buys), len(set(all_trans))
    print('\tusers, items, trans:    = {v1}, {v2}, {v3}'.format(v1=user_num, v2=item_num, v3=tran_num))
    print('\tavg. user buy:          = {val}'.format(val=1.0 * tran_num / user_num))
    print('\tavg. item bought:       = {val}'.format(val=1.0 * tran_num / item_num))

    # 选取训练集、验证集(测试集)，并对test去重。不管是valid还是test模式，统一用train，test表示。
    print('Split the training set, test set: mode = {val} ...'.format(val=mode))
    print('Remove duplicates in test set')
    tra_buys, tes_buys = [], []
    for ubuys in all_user_buys:
        # 按序列长度切分。
        le = len(ubuys)
        split0, split1, split2 = 0, int(le * split[0]), int(le * split[1])
        left, right = ubuys[split0: split1], ubuys[split1: split2]
        # test需要自身去重。不会减少用户数量。
        right_stay = list(set(right))       # 先去重
        right_stay.sort(key=right.index)    # test里剩余的按原顺序排
        # 保存
        tra_buys.append(left)
        tes_buys.append(right_stay)

    # 去重后的基本信息，
    all_trans = []
    for utra, utes in zip(tra_buys, tes_buys):
        all_trans.extend(utra)
        all_trans.extend(utes)
    tran_num, user_num, item_num = len(all_trans), len(tra_buys), len(set(all_trans))
    print('\tusers, items, trans:    = {v1}, {v2}, {v3}'.format(v1=user_num, v2=item_num, v3=tran_num))
    print('\tavg. user buy:          = {val}'.format(val=1.0 * tran_num / user_num))
    print('\tavg. item bought:       = {val}'.format(val=1.0 * tran_num / item_num))

    # 建立商品别名字典。更新购买记录，替换为0~len(se)-1的别名
    print('Use aliases to represent items ...')
    all_items = set(all_trans)
    aliases_dict = dict(zip(all_items, range(item_num)))
    tra_buys = [[aliases_dict[i] for i in utra] for utra in tra_buys]
    tes_buys = [[aliases_dict[i] for i in utes] for utes in tes_buys]

    return [(user_num, item_num), (tra_buys, tes_buys)]


def fun_data_buys_masks(all_user_buys, tail):
    # 将train/test中序列补全为最大长度，补的idx值=item_num. 为了能在theano里对其进行shared。
    # tail, 添加的。商品索引是0~item_num-1，所以该值[item_num]是没有对应商品实物的。
    us_lens = [len(ubuys) for ubuys in all_user_buys]
    len_max = max(us_lens)
    us_buys = [ubuys + tail * (len_max - le) for ubuys, le in zip(all_user_buys, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return np.asarray(us_buys), np.asarray(us_msks)


def fun_random_neg_masks_tra(item_num, tras_mask):
    """
    从num件商品里随机抽取与每个用户的购买序列等长且不在已购买商品里的标号。后边补全的负样本用虚拟商品[item_num]
    """
    us_negs = []
    for utra in tras_mask:     # 每条用户序列
        unegs = []
        for i, e in enumerate(utra):
            if item_num == e:                        # 表示该购买以及之后的，都是用虚拟商品[item_num]来补全的
                unegs += [item_num] * (len(utra) - i)   # 购买序列里对应补全商品的负样本也用补全商品表示
                break
            j = random.randint(0, item_num - 1)      # 负样本在商品矩阵里的标号
            while j in utra:                     # 抽到的不是用户训练集里的。
                j = random.randint(0, item_num - 1)
            unegs += [j]
        us_negs.append(unegs)
    return np.asarray(us_negs)


def fun_random_neg_masks_tes(item_num, tras_mask, tess_mask):
    """
    从num件商品里随机抽取与测试序列等长且不在训练序列、也不再测试序列里的标号
    """
    us_negs = []
    for utra, utes in zip(tras_mask, tess_mask):
        unegs = []
        for i, e in enumerate(utes):
            if item_num == e:                   # 尾部补全mask
                unegs += [item_num] * (len(utes) - i)
                break
            j = random.randint(0, item_num - 1)
            while j in utra or j in utes:         # 不在训练序列，也不在预测序列里。
                j = random.randint(0, item_num - 1)
            unegs += [j]
        us_negs.append(unegs)
    return np.asarray(us_negs)


def fun_tra_set(tras_mask, tail, set_len):
    """
    把一条序列做成set形式。多出来的item用tail补充。
    """
    assert set_len >= 2         # 每次最少处理两个items。
    tra_set = []
    for utra in tras_mask:
        u_set = []
        pre = set_len - 1        # 比如每次处理3个，则序列前补充2个tail
        utra = tail * pre + list(utra)
        for i in range(pre, len(utra)):
            u_set.append(utra[i-pre: i+1])
        tra_set.append(u_set)   # len(u_set) = len(utra) + 1, u_set第一个set全是tail组成的。

    # 举例
    # tra: 正样本 a = [a, b, c, d]  # 长度4, tail=[h], set_len = 2
    # tra: 负样本 a'= [a',b',c',d']
    # tra_set:   s = [[h, a], [a, b], [b, c], [c, d]]  # 长度4
    # 训练时，比如当前时刻t=0, 用于resnet建模的是[h, a]，注意正负样本是t=1的[b, b'],

    # b = list(range(4))      # [0, 1, 2, 3]
    # pre = 2 - 1
    # tail = [10]
    # b = tail * pre + b      # [10, 0, 1, 2, 3]
    # a = []                  # [[10, 0], [0, 1], [1, 2], [2, 3]]
    # for i in range(pre, len(b)):
    #     a.append(b[i-pre: i+1])

    return np.asarray(tra_set)


@exe_time  # 放到待调用函数的定义的上一行
def main():
    print('... load the dataset, and  no need to set shared.')


if '__main__' == __name__:
    main()
