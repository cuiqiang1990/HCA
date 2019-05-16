#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:       Qiang Cui:  <cuiqiang1990[at]hotmail.com>
# Descripton:   construct the class GRU
# Create Date:  2016-12-02 17:00:00
# Modify Date:  2016-00-00 00:00:00
# Modify Disp:

from __future__ import print_function
import time
import numpy as np
from numpy.random import uniform
import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh
from theano.tensor.extra_ops import Unique
from GRU import GruBasic
__docformat__ = 'restructedtext en'

theano.config.floatX = 'float32'


def exe_time(func):
    def new_func(*args, **args2):
        t0 = time.time()
        print("-- @%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__))
        back = func(*args, **args2)
        print("-- @%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__))
        print("-- @%.3fs taken for {%s}" % (time.time() - t0, func.__name__))
        return back
    return new_func


# 程序地址：https://github.com/cuiqiang1990/zoneout
# 用theano做，Sample masks那里错了，不需要用1-z_prob_states：https://github.com/teganmaharaj/zoneout/blob/master/zoneout_theano.py
# 内容更多：https://github.com/teganmaharaj/zoneout/blob/master/zoneout_blocksfuel.py
# ======================================================================================================================
class OboZoneOutGru(GruBasic):
    def __init__(self, train, test, zoneout, alpha_lambda, n_user, n_item, n_in, n_hidden):
        super(OboZoneOutGru, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        self.n_hidden = n_hidden
        # 保留上时刻值的比例。不需要做shared
        zoneout_cell, zoneout_hidd = zoneout
        self.zc = zoneout_cell
        self.zh = zoneout_hidd
        self.params = [self.ui, self.wh, self.bi]       # self.lt单独进行更新。
        self.l2_sqr = (
            T.sum(self.lt ** 2) +
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__(n_in, n_hidden)
        self.__theano_predict__(n_in, n_hidden)

    def zoneout_tra(self, shape, zone_prob):
        # 处理向量：randomly set some positions to zero. Vector.shape=(n, )
        # 例如一个向量20维，就有20个位置，也就是有20个神经元。最后各元素是1/0的比例分别是update_prob/zone_prob
        # train时做zoneout（保留上时刻值的比例），test时用定值。
        # if drop_prob < 0. or drop_prob >= 1.:
        #     raise Exception('Drop prob must be in interval [0, 1)')
        # 以zone_prob这个概率生成1，采用上时刻值的比例。比如zone_prob=0.2，那就是20%的位置用上次值，80%的用当前值。
        randoms = self.thea_rng.binomial(
            size=shape,
            n=1,                            # 每个神经元实验一次
            p=zone_prob,                    # 每个神经元*1的概率为zone_prob。
            dtype=theano.config.floatX)
        return randoms

    def zoneout_tes(self, shape, zone_prob):
        # 测试时，和训练不一样，示例程序里不用随机，用定值。经过实验，确实也是tes用定值的效果好。
        # randoms是保留上时刻值的比例。得用'T.ones'，不要用'np.ones'
        randoms = zone_prob * T.ones(shape, dtype=theano.config.floatX)
        return randoms

    def __theano_train__(self, n_in, n_hidden):
        """
        训练阶段跑一遍训练序列
        """
        ui, wh = self.ui, self.wh
        zh = self.zh

        tra_mask = T.ivector()
        seq_length = T.sum(tra_mask)                # 有效长度

        # 先生成mask，当成个sequence传给scan。如果在scan里逐step生成的话，会有问题。
        zones = self.zoneout_tra((seq_length, self.n_hidden), zh)

        h0 = self.h0
        bi = self.bi

        pidxs, qidxs = T.ivector(), T.ivector()
        xps, xqs = self.lt[pidxs], self.lt[qidxs]   # shape((seq_length, n_in))

        pqs = T.concatenate((pidxs, qidxs))         # 先拼接
        uiq_pqs = Unique(False, False, False)(pqs)  # 再去重
        uiq_x = self.lt[uiq_pqs]                    # 相应的items特征

        """
        输入t时刻正负样本、t-1时刻隐层，计算当前隐层、当前损失. 公式里省略了时刻t
        # 根据性质：T.dot((m, n), (n, ))得到shape=(m, )，且是矩阵每行与(n, )相乘
            # GRU
            z = sigmoid(ux_z * xp + wh_z * h_pre1)
            r = sigmoid(ux_r * xp + wh_r * h_pre1)
            c = tanh(ux_c * xp + wh_c * (r 点乘 h_pre1))
            h = z * h_pre1 + (1.0 - z) * c
        # 根据性质：T.dot((n, ), (n, ))得到scalar
            upq  = h_pre1 * (xp - xq)
            loss = log(1.0 + e^(-upq))
        """
        def recurrence(xp_t, xq_t, zone_hidd, h_t_pre1):
            z_r = sigmoid(T.dot(ui[:2], xp_t) +
                          T.dot(wh[:2], h_t_pre1) + bi[:2])
            z, r = z_r[0], z_r[1]
            c = tanh(T.dot(ui[2], xp_t) +
                     T.dot(wh[2], (r * h_t_pre1)) + bi[2])
            h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c
            upq_t = T.dot(h_t_pre1, xp_t - xq_t)
            loss_t = T.log(sigmoid(upq_t))

            # zoneout
            h_t = h_t_pre1 * zone_hidd + h_t * (1. - zone_hidd)

            return [h_t, loss_t]
        [h, loss], _ = theano.scan(
            fn=recurrence,
            sequences=[xps, xqs, zones],
            outputs_info=[h0, None],
            n_steps=seq_length,
            truncate_gradient=-1)

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        seq_l2_sq = T.sum([T.sum(par ** 2) for par in [xps, xqs, ui, wh, bi]])
        upq = T.sum(loss)
        seq_costs = (
            - upq +
            0.5 * l2 * seq_l2_sq)
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        update_x = T.set_subtensor(uiq_x, uiq_x - lr * T.grad(seq_costs, self.lt)[uiq_pqs])
        seq_updates.append((self.lt, update_x))     # 会直接更改到seq_updates里
        # ----------------------------------------------------------------------------

        # 输入正、负样本序列及其它参数后，更新变量，返回损失。
        uidx = T.iscalar()                              # T.iscalar()类型是 TensorType(int32, )
        self.seq_train = theano.function(
            inputs=[uidx],
            outputs=-upq,
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[uidx],       # 类型是 TensorType(int32, matrix)
                qidxs: self.tra_buys_neg_masks[uidx],
                tra_mask: self.tra_masks[uidx]})

    def __theano_predict__(self, n_in, n_hidden):
        """
        测试阶段再跑一遍训练序列得到各个隐层。用全部数据一次性得出所有用户的表达
        """
        ui, wh = self.ui, self.wh
        zh = self.zh

        tra_mask = T.imatrix()
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length

        zoneout_hidd = self.zoneout_tes((actual_batch_size, self.n_hidden), zh) # shape=(n, 20)

        h0 = T.alloc(self.h0, actual_batch_size, n_hidden)      # shape=(n, 20)
        bi = T.alloc(self.bi, actual_batch_size, 3, n_hidden)   # shape=(n, 3, 20), 原维度放在后边
        bi = bi.dimshuffle(1, 2, 0)                             # shape=(3, 20, n)

        # 隐层是1个GRU Unit：都可以用这个统一的格式。
        pidxs = T.imatrix()
        ps = self.trained_items[pidxs]      # shape((actual_batch_size, seq_length, n_hidden))
        ps = ps.dimshuffle(1, 0, 2)         # shape=(seq_length, batch_size, n_hidden)=(157, n, 20)

        def recurrence(p_t, h_t_pre1):
            # 特征、隐层都处理成shape=(batch_size, n_hidden)=(n, 20)
            z_r = sigmoid(T.dot(ui[:2], p_t.T) +
                          T.dot(wh[:2], h_t_pre1.T) + bi[:2])
            z, r = z_r[0].T, z_r[1].T                           # shape=(n, 20)
            c = tanh(T.dot(ui[2], p_t.T) +
                     T.dot(wh[2], (r * h_t_pre1).T) + bi[2])    # shape=(20, n)
            h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c.T     # shape=(n, 20)

            # zoneout
            h_t = h_t_pre1 * zoneout_hidd + h_t * (1. - zoneout_hidd)

            return h_t
        h, _ = theano.scan(         # h.shape=(157, n, 20)
            fn=recurrence,
            sequences=ps,
            outputs_info=h0,
            n_steps=seq_length)

        # 得到batch_hts.shape=(n, 20)，就是这个batch里每个用户的表达ht。
        # 必须要用T.sum()，不然无法建模到theano的graph里、报length not known的错
        hs = h.dimshuffle(1, 0, 2)                      # shape=(batch_size, seq_length, n_hidden)
        hts = hs[                                       # shape=(n, n_hidden)
            T.arange(actual_batch_size),                # 行. 花式索引a[[1,2,3],[2,5,6]]，需给定行列的表示
            T.sum(tra_mask, axis=1) - 1]                # 列。需要mask是'int32'型的

        # givens给数据
        start_end = T.ivector()
        self.seq_predict = theano.function(
            inputs=[start_end],
            outputs=hts,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                tra_mask: self.tra_masks[start_end]})

    def train(self, idx):
        # consider the whole user sequence as a mini-batch and perform one update per sequence
        return self.seq_train(idx)


# ======================================================================================================================
class OboZoneOutLstm(GruBasic):
    def __init__(self, train, test, zoneout, alpha_lambda, n_user, n_item, n_in, n_hidden):
        super(OboZoneOutLstm, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        self.n_hidden = n_hidden
        # 比例
        zoneout_cell, zoneout_hidd = zoneout
        self.zc = zoneout_cell
        self.zh = zoneout_hidd
        # 初始化，先定义局部变量，再self.修饰成实例变量
        rang = 0.5
        ui = uniform(-rang, rang, (4, n_hidden, n_hidden))
        wh = uniform(-rang, rang, (4, n_hidden, n_hidden))
        c0 = np.zeros((n_hidden, ), dtype=theano.config.floatX)
        bi = np.zeros((4, n_hidden), dtype=theano.config.floatX)
        # 建立参数。
        self.ui = theano.shared(borrow=True, value=ui.astype(theano.config.floatX))
        self.wh = theano.shared(borrow=True, value=wh.astype(theano.config.floatX))
        self.c0 = theano.shared(borrow=True, value=c0)
        self.bi = theano.shared(borrow=True, value=bi)
        self.params = [self.ui, self.wh, self.bi]       # self.lt单独进行更新。
        self.l2_sqr = (
            T.sum(self.lt ** 2) +
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__(n_in, n_hidden)
        self.__theano_predict__(n_in, n_hidden)

    def zoneout_tra(self, zone_prob):
        # 处理向量：randomly set some positions to zero. Vector.shape=(n, )
        # 例如一个向量20维，就有20个位置，也就是有20个神经元。最后各元素是1/0的比例分别是update_prob/zone_prob
        # train时做zoneout（保留上时刻值的比例），test时用定值。
        # if drop_prob < 0. or drop_prob >= 1.:
        #     raise Exception('Drop prob must be in interval [0, 1)')
        update_prob = 1. - zone_prob        # 以这个概率生成1，就是正常激活，保留当前值。zone_prob是采用上时刻值的比例。
        randoms = self.thea_rng.binomial(
            size=(self.n_hidden, ),
            n=1,                            # 每个神经元实验一次
            p=update_prob)                  # 每个神经元*1的概率为update_prob。*0的概率为zone_prob
        return randoms

    def zoneout_tes(self, batch_size, zone_prob):
        # 测试时，不用随机了，用定值。
        # randoms是保留当前值的比例。
        randoms = np.ones((batch_size, self.n_hidden), dtype=theano.config.floatX) * (1. - zone_prob)
        return randoms

    def __theano_train__(self, n_in, n_hidden):
        """
        训练阶段跑一遍训练序列
        """
        # self.alpha_lambda = ['alpha', 'lambda']
        ui, wh = self.ui, self.wh
        zc, zh = self.zc, self.zh

        tra_mask = T.ivector()
        seq_length = T.sum(tra_mask)                # 有效长度

        h0 = self.h0
        c0 = self.c0
        bi = self.bi

        pidxs, qidxs = T.ivector(), T.ivector()
        xps, xqs = self.lt[pidxs], self.lt[qidxs]   # shape((seq_length, n_in))

        pqs = T.concatenate((pidxs, qidxs))         # 先拼接
        uiq_pqs = Unique(False, False, False)(pqs)  # 再去重
        uiq_x = self.lt[uiq_pqs]                    # 相应的items特征

        """
        输入t时刻正负样本、t-1时刻隐层，计算当前隐层、当前损失. 公式里省略了时刻t
        # 根据性质：T.dot((m, n), (n, ))得到shape=(m, )，且是矩阵每行与(n, )相乘
            # GRU
            z = sigmoid(ux_z * xp + wh_z * h_pre1)
            r = sigmoid(ux_r * xp + wh_r * h_pre1)
            c = tanh(ux_c * xp + wh_c * (r 点乘 h_pre1))
            h = z * h_pre1 + (1.0 - z) * c
        # 根据性质：T.dot((n, ), (n, ))得到scalar
            upq  = h_pre1 * (xp - xq)
            loss = log(1.0 + e^(-upq))
        """
        def recurrence(xp_t, xq_t, c_t_pre1, h_t_pre1):
            gates = T.dot(ui, xp_t) + T.dot(wh, h_t_pre1) + bi
            i, f, g, o = sigmoid(gates[0]), sigmoid(gates[1]), tanh(gates[2]), sigmoid(gates[3])
            c_t = f * c_t_pre1 + i * g
            h_t = o * tanh(c_t)
            upq_t = T.dot(h_t_pre1, xp_t - xq_t)
            loss_t = T.log(sigmoid(upq_t))

            zoneout_cell = self.zoneout_tra(zc)
            zoneout_hidd = self.zoneout_tra(zh)
            c_t = c_t_pre1 * zoneout_cell + c_t * (1. - zoneout_cell)
            h_t = h_t_pre1 * zoneout_hidd + h_t * (1. - zoneout_hidd)

            return [c_t, h_t, loss_t]
        [c, h, loss], _ = theano.scan(
            fn=recurrence,
            sequences=[xps, xqs],
            outputs_info=[c0, h0, None],
            n_steps=seq_length,
            truncate_gradient=-1)

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        seq_l2_sq = T.sum([T.sum(par ** 2) for par in [xps, xqs, ui, wh, bi]])
        upq = T.sum(loss)
        seq_costs = (
            - upq +
            0.5 * l2 * seq_l2_sq)
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        update_x = T.set_subtensor(uiq_x, uiq_x - lr * T.grad(seq_costs, self.lt)[uiq_pqs])
        seq_updates.append((self.lt, update_x))     # 会直接更改到seq_updates里
        # ----------------------------------------------------------------------------

        # 输入正、负样本序列及其它参数后，更新变量，返回损失。
        uidx = T.iscalar()                              # T.iscalar()类型是 TensorType(int32, )
        self.seq_train = theano.function(
            inputs=[uidx],
            outputs=-upq,
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[uidx],       # 类型是 TensorType(int32, matrix)
                qidxs: self.tra_buys_neg_masks[uidx],
                tra_mask: self.tra_masks[uidx]})

    def train(self, idx):
        # consider the whole user sequence as a mini-batch and perform one update per sequence
        return self.seq_train(idx)

    def __theano_predict__(self, n_in, n_hidden):
        """
        测试阶段再跑一遍训练序列得到各个隐层。用全部数据一次性得出所有用户的表达
        """
        ui, wh = self.ui, self.wh
        zc, zh = self.zc, self.zh

        tra_mask = T.imatrix()
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length

        zoneout_cell = self.zoneout_tes(actual_batch_size, zc)  # shape=(n, 20)
        zoneout_hidd = self.zoneout_tes(actual_batch_size, zh)

        h0 = T.alloc(self.h0, actual_batch_size, n_hidden)      # shape=(n, 20)
        c0 = T.alloc(self.c0, actual_batch_size, n_hidden)      # shape=(n, 20)
        bi = T.alloc(self.bi, actual_batch_size, 4, n_hidden)   # shape=(n, 3, 20), 原维度放在后边
        bi = bi.dimshuffle(1, 2, 0)                             # shape=(3, 20, n)

        # 隐层是1个GRU Unit：都可以用这个统一的格式。
        pidxs = T.imatrix()
        ps = self.trained_items[pidxs]      # shape((actual_batch_size, seq_length, n_hidden))
        ps = ps.dimshuffle(1, 0, 2)         # shape=(seq_length, batch_size, n_hidden)=(157, n, 20)

        def recurrence(p_t, c_t_pre1, h_t_pre1):
            # 特征、隐层都处理成shape=(batch_size, n_hidden)=(n, 20)
            gates = T.dot(ui, p_t.T) + T.dot(wh, h_t_pre1.T) + bi  # shape=(4, 20, n)
            i, f, g, o = sigmoid(gates[0]).T, sigmoid(gates[1]).T, tanh(gates[2]).T, sigmoid(gates[3]).T
            c_t = f * c_t_pre1 + i * g
            h_t = o * tanh(c_t)   # shape=(n, 20)

            c_t = c_t_pre1 * zoneout_cell + c_t * (1. - zoneout_cell)
            h_t = h_t_pre1 * zoneout_hidd + h_t * (1. - zoneout_hidd)

            return [c_t, h_t]
        [c, h], _ = theano.scan(         # h.shape=(157, n, 20)
            fn=recurrence,
            sequences=ps,
            outputs_info=[c0, h0],
            n_steps=seq_length)

        # 得到batch_hts.shape=(n, 20)，就是这个batch里每个用户的表达ht。
        # 必须要用T.sum()，不然无法建模到theano的graph里、报length not known的错
        hs = h.dimshuffle(1, 0, 2)                      # shape=(batch_size, seq_length, n_hidden)
        hts = hs[                                       # shape=(n, n_hidden)
            T.arange(actual_batch_size),                # 行. 花式索引a[[1,2,3],[2,5,6]]，需给定行列的表示
            T.sum(tra_mask, axis=1) - 1]                # 列。需要mask是'int32'型的

        # givens给数据
        start_end = T.ivector()
        self.seq_predict = theano.function(
            inputs=[start_end],
            outputs=hts,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                tra_mask: self.tra_masks[start_end]})


@exe_time  # 放到待调用函数的定义的上一行
def main():
    print('... construct the class: GRU')


if '__main__' == __name__:
    main()

