#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:       Qiang Cui:  <cuiqiang1990[at]hotmail.com>
# Descripton:   
# Create Date:  2017-11-14 15:00:00
# Modify Date:  2017-00-00 00:00:00
# Modify Disp:

import datetime
import numpy as np
from numpy.random import uniform
import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh
from theano.tensor.extra_ops import Unique
from theano.tensor.nnet import softmax
from GRU import GruBasic
__docformat__ = 'restructedtext en'


def exe_time(func):
    def new_func(*args, **args2):
        name = func.__name__
        start = datetime.datetime.now()
        print("-- {%s} start: @ %ss" % (name, start.strftime("%Y.%m.%d_%H.%M.%S")))
        back = func(*args, **args2)
        end = datetime.datetime.now()
        print("-- {%s} start: @ %ss" % (name, start.strftime("%Y.%m.%d_%H.%M.%S")))
        print("-- {%s} end:   @ %ss" % (name, end.strftime("%Y.%m.%d_%H.%M.%S")))
        total = (end - start).total_seconds()
        print("-- {%s} total: @ %.2fs = %.2fh" % (name, total, total / 3600.0))
        return back
    return new_func


# ======================================================================================================================
class HcaGruX(GruBasic):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden,
                 window_input, window_hidden):
        super(HcaGruX, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        rang = 0.5
        self.window_input = theano.shared(borrow=True, value=np.asarray(window_input, dtype='int32'))
        self.window_hidden = theano.shared(borrow=True, value=np.asarray(window_hidden, dtype='int32'))
        # context_x
        vc = uniform(-rang, rang, (3, n_hidden, n_in))
        qx = uniform(-rang, rang, (n_in, n_in))
        rx = uniform(-rang, rang, (n_in, 1))
        self.vc = theano.shared(borrow=True, value=vc.astype(theano.config.floatX))
        self.qx = theano.shared(borrow=True, value=qx.astype(theano.config.floatX))
        self.rx = theano.shared(borrow=True, value=rx.astype(theano.config.floatX))
        self.params = [
            self.ui, self.wh, self.bi,
            self.qx, self.rx, self.vc]  # self.lt单独进行更新。
        self.l2_sqr = (
            T.sum(self.lt ** 2) +
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__(n_in, n_hidden)
        self.__theano_predict__(n_in, n_hidden)

    def __theano_train__(self, n_in, n_hidden):
        """
        训练阶段跑一遍训练序列
        """
        # self.alpha_lambda = ['alpha', 'lambda']
        ui, wh, bi = self.ui, self.wh, self.bi
        qx, rx, vc = self.qx, self.rx, self.vc
        winx = self.window_input

        tra_mask = T.imatrix()          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length
        mask = tra_mask.T                               # shape=(157, n)

        bi = T.alloc(self.bi, actual_batch_size, 3, n_hidden)   # shape=(n, 3, 20), 原维度放在后边
        bi = bi.dimshuffle(1, 2, 0)                             # shape=(3, 20, n)

        pidxs, qidxs = T.imatrix(), T.imatrix()
        xps, xqs = self.lt[pidxs], self.lt[qidxs]   # shape((actual_batch_size, seq_length, n_in))
        xps = xps.dimshuffle(1, 0, 2)               # shape=(seq_length, batch_size, n_in)=(157, n, 20)
        xqs = xqs.dimshuffle(1, 0, 2)

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
        def recurrence(xp_t, xp_t1, xq_t1, mask_t, h_t_pre1, cxs):
            # 特征、隐层都处理成shape=(batch_size, n_hidden)=(n, 20)
            # (n, winx, 20) = T.concatenate((((n, winx-1, 20)), ((n, 1, 20))), axis=1)
            # context_x
            # 定义定长矩阵，xp_t拼接到最底下，删除首行, 矩阵维度不变。
            cxs = T.concatenate((cxs[:, 1:, :],             # shape=(n, winx-1, 20)
                                 xp_t.dimshuffle(0, 'x', 1)),    # shape=(n, 1, 20)
                                axis=1)                     # shape=(n, winx, 20)
            exs = T.dot(tanh(T.dot(cxs, qx)), rx)           # shape=(n, winx, 1)
            exs = T.Rebroadcast((2, True))(exs)             # axis=2进行broadcast, 使其可被丢掉
            axs0 = softmax(exs.dimshuffle(0, 1))            # shape=(n, winx)，降一维。因为softmax按行处理。
            axs = axs0.dimshuffle(0, 1, 'x')                # shape=(n, winx, 1), 升一维。还原回去。
            axs = T.Rebroadcast((2, True))(axs)             # axis=2进行broadcast, 使其可做乘法。
            # (n, 20) = T.sum((n, winx, 20) * (n, winx, 1), axis=1)
            xc = T.sum(cxs * axs, axis=1)                     # shape=(n, 20)
            # gru unit
            z_r = sigmoid(T.dot(ui[:2], xp_t.T) +
                          T.dot(vc[:2], xc.T) +
                          T.dot(wh[:2], h_t_pre1.T) + bi[:2])
            z, r = z_r[0].T, z_r[1].T                           # shape=(n, 20)
            c = tanh(T.dot(ui[2], xp_t.T) +
                     T.dot(vc[2], xc.T) +
                     T.dot(wh[2], (r * h_t_pre1).T) + bi[2])
            h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c.T     # shape=(n, 20)
            # loss
            upq_t = T.sum(h_t * (xp_t1 - xq_t1), axis=1)    # shape=(n, ), h(t) * (xp(t+1) - xq(t+1)), 正负样本训练。
            loss_t = T.log(sigmoid(upq_t))
            loss_t *= mask_t                                # 只在损失这里乘一下0/1向量就可以了
            return [h_t, cxs, loss_t]

        batch_h0 = T.alloc(self.h0, actual_batch_size, n_hidden)
        cumx = T.alloc(self.lt[-1], actual_batch_size, winx, n_in)         # concatenate
        [_, _, loss], _ = theano.scan(               # h是h1~ht。loss是h0~ht-1和x1~xt计算得到的。
            fn=recurrence,
            sequences=[xps, xps[1:], xqs[1:], mask],
            outputs_info=[batch_h0, cumx, None],
            n_steps=seq_length-1,
            truncate_gradient=-1)

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        seq_l2_sq = (
            T.sum([T.sum(par ** 2) for par in [xps, xqs, ui, wh, qx, rx, vc]]) +
            T.sum([T.sum(par ** 2) for par in [bi]]) / actual_batch_size)
        upq = T.sum(loss)
        seq_costs = (
            - upq / actual_batch_size +
            0.5 * l2 * seq_l2_sq)
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        update_x = T.set_subtensor(uiq_x, uiq_x - lr * T.grad(seq_costs, self.lt)[uiq_pqs])
        seq_updates.append((self.lt, update_x))     # 会直接更改到seq_updates里
        # ----------------------------------------------------------------------------

        # givens给数据
        start_end = T.ivector()
        self.seq_train = theano.function(
            inputs=[start_end],
            outputs=-upq,
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                qidxs: self.tra_buys_neg_masks[start_end],
                tra_mask: self.tra_masks[start_end]})

    def __theano_predict__(self, n_in, n_hidden):
        """
        测试阶段再跑一遍训练序列得到各个隐层。用全部数据一次性得出所有用户的表达
        """
        ui, wh = self.ui, self.wh
        qx, rx, vc = self.qx, self.rx, self.vc
        winx = self.window_input

        tra_mask = T.imatrix()          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length

        bi = T.alloc(self.bi, actual_batch_size, 3, n_hidden)   # shape=(n, 3, 20), 原维度放在后边
        bi = bi.dimshuffle(1, 2, 0)                             # shape=(3, 20, n)

        pidxs = T.imatrix()
        xps = self.lt[pidxs]                # shape((actual_batch_size, seq_length, n_in))
        xps = xps.dimshuffle(1, 0, 2)       # shape=(seq_length, batch_size, n_in)=(157, n, 20)

        def recurrence(xp_t, h_t_pre1, cxs):
            # 特征、隐层都处理成shape=(batch_size, n_hidden)=(n, 20)
            # (n, winx, 20) = T.concatenate((((n, winx-1, 20)), ((n, 1, 20))), axis=1)
            # context_x
            # 定义定长矩阵，xp_t拼接到最底下，删除首行, 矩阵维度不变。
            cxs = T.concatenate((cxs[:, 1:, :],             # shape=(n, winx-1, 20)
                                 xp_t.dimshuffle(0, 'x', 1)),    # shape=(n, 1, 20)
                                axis=1)                     # shape=(n, winx, 20)
            exs = T.dot(tanh(T.dot(cxs, qx)), rx)           # shape=(n, winx, 1)
            exs = T.Rebroadcast((2, True))(exs)             # axis=2进行broadcast, 使其可被丢掉
            axs0 = softmax(exs.dimshuffle(0, 1))            # shape=(n, winx)，降一维。因为softmax按行处理。
            axs = axs0.dimshuffle(0, 1, 'x')                # shape=(n, winx, 1), 升一维。还原回去。
            axs = T.Rebroadcast((2, True))(axs)             # axis=2进行broadcast, 使其可做乘法。
            # (n, 20) = T.sum((n, winx, 20) * (n, winx, 1), axis=1)
            xc = T.sum(cxs * axs, axis=1)                     # shape=(n, 20)
            # gru unit
            z_r = sigmoid(T.dot(ui[:2], xp_t.T) +
                          T.dot(vc[:2], xc.T) +
                          T.dot(wh[:2], h_t_pre1.T) + bi[:2])
            z, r = z_r[0].T, z_r[1].T                           # shape=(n, 20)
            c = tanh(T.dot(ui[2], xp_t.T) +
                     T.dot(vc[2], xc.T) +
                     T.dot(wh[2], (r * h_t_pre1).T) + bi[2])
            h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c.T     # shape=(n, 20)
            return [h_t, cxs]

        batch_h0 = T.alloc(self.h0, actual_batch_size, n_hidden)
        cumx = T.alloc(self.lt[-1], actual_batch_size, winx, n_in)
        [h, _], _ = theano.scan(           # h.shape=(157, n, 20)
            fn=recurrence,
            sequences=xps,
            outputs_info=[batch_h0, cumx],
            n_steps=seq_length)

        # 每个用户最后一个隐层。
        hs = h.dimshuffle(1, 0, 2)              # shape=(n, 157, 20)
        hts = hs[                               # shape=(n, 20)
            T.arange(actual_batch_size),        # 行。 行列用花式索引a[[1,2,3],[2,5,6]]，需给定行列的表示
            T.sum(tra_mask, axis=1) - 1]        # 列。需要mask是'int32'型的。不能使用a[:, [1,2,3]]。

        # givens给数据
        start_end = T.ivector()
        self.seq_predict = theano.function(
            inputs=[start_end],
            outputs=hts,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                tra_mask: self.tra_masks[start_end]})

    def train(self, idxs):
        return self.seq_train(idxs)


# ======================================================================================================================
class HcaGruH(GruBasic):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden,
                 window_input, window_hidden):
        super(HcaGruH, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        rang = 0.5
        self.window_input = theano.shared(borrow=True, value=np.asarray(window_input, dtype='int32'))
        self.window_hidden = theano.shared(borrow=True, value=np.asarray(window_hidden, dtype='int32'))
        # context_h
        qh = uniform(-rang, rang, (n_hidden, n_hidden))
        rh = uniform(-rang, rang, (n_hidden, 1))
        e = uniform(-rang, rang, (n_hidden, n_hidden))
        f = uniform(-rang, rang, (n_hidden, n_hidden))
        self.qh = theano.shared(borrow=True, value=qh.astype(theano.config.floatX))
        self.rh = theano.shared(borrow=True, value=rh.astype(theano.config.floatX))
        self.e = theano.shared(borrow=True, value=e.astype(theano.config.floatX))
        self.f = theano.shared(borrow=True, value=f.astype(theano.config.floatX))
        self.params = [
            self.ui, self.wh, self.bi,
            self.qh, self.rh, self.e, self.f]  # self.lt单独进行更新。
        self.l2_sqr = (
            T.sum(self.lt ** 2) +
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__(n_in, n_hidden)
        self.__theano_predict__(n_in, n_hidden)

    def __theano_train__(self, n_in, n_hidden):
        """
        训练阶段跑一遍训练序列
        """
        # self.alpha_lambda = ['alpha', 'lambda']
        ui, wh, bi = self.ui, self.wh, self.bi
        qh, rh, e, f = self.qh, self.rh, self.e, self.f
        winh = self.window_hidden

        tra_mask = T.imatrix()          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length
        mask = tra_mask.T                               # shape=(157, n)

        bi = T.alloc(self.bi, actual_batch_size, 3, n_hidden)   # shape=(n, 3, 20), 原维度放在后边
        bi = bi.dimshuffle(1, 2, 0)                             # shape=(3, 20, n)

        pidxs, qidxs = T.imatrix(), T.imatrix()
        xps, xqs = self.lt[pidxs], self.lt[qidxs]   # shape((actual_batch_size, seq_length, n_in))
        xps = xps.dimshuffle(1, 0, 2)               # shape=(seq_length, batch_size, n_in)=(157, n, 20)
        xqs = xqs.dimshuffle(1, 0, 2)

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
        def recurrence(xp_t, xp_t1, xq_t1, mask_t, h_t_pre1, chs):
            # gru unit
            z_r = sigmoid(T.dot(ui[:2], xp_t.T) +
                          T.dot(wh[:2], h_t_pre1.T) + bi[:2])
            z, r = z_r[0].T, z_r[1].T                           # shape=(n, 20)
            c = tanh(T.dot(ui[2], xp_t.T) +
                     T.dot(wh[2], (r * h_t_pre1).T) + bi[2])
            h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c.T     # shape=(n, 20)
            # context_h
            # 定义定长矩阵，h_t拼接到最底下，删除首行, 矩阵维度不变。
            chs = T.concatenate((chs[:, 1:, :],             # shape=(n, winh-1, 20)
                                 h_t.dimshuffle(0, 'x', 1)),    # shape=(n, 1, 20)
                                axis=1)                     # shape=(n, winh, 20)
            ehs = T.dot(tanh(T.dot(chs, qh)), rh)           # shape=(n, winh, 1)
            ehs = T.Rebroadcast((2, True))(ehs)             # axis=2进行broadcast, 使其可被丢掉
            ahs0 = softmax(ehs.dimshuffle(0, 1))            # shape=(n, winh)，降一维。因为softmax按行处理。
            ahs = ahs0.dimshuffle(0, 1, 'x')                # shape=(n, winh, 1), 升一维。还原回去
            ahs = T.Rebroadcast((2, True))(ahs)             # axis=2进行broadcast, 使其可做乘法。
            hcs = T.sum(chs * ahs, axis=1)                  # shape=(n, 20)
            # 整体表达hws，融合当前hts、上下文hcs
            hws = tanh(T.dot(h_t, e.T) + T.dot(hcs, f.T))   # shape=(n, 20)
            # loss
            upq_t = T.sum(hws * (xp_t1 - xq_t1), axis=1)    # shape=(n, ), h(t) * (xp(t+1) - xq(t+1)), 正负样本训练。
            loss_t = T.log(sigmoid(upq_t))
            loss_t *= mask_t                                # 只在损失这里乘一下0/1向量就可以了
            return [h_t, chs, loss_t]

        batch_h0 = T.alloc(self.h0, actual_batch_size, n_hidden)
        cumh = T.alloc(self.h0, actual_batch_size, winh, n_hidden)
        [_, _, loss], _ = theano.scan(               # h是h1~ht。loss是h0~ht-1和x1~xt计算得到的。
            fn=recurrence,
            sequences=[xps, xps[1:], xqs[1:], mask],
            outputs_info=[batch_h0, cumh, None],
            n_steps=seq_length-1,
            truncate_gradient=-1)

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        seq_l2_sq = (
            T.sum([T.sum(par ** 2) for par in [xps, xqs, ui, wh, qh, rh, e, f]]) +
            T.sum([T.sum(par ** 2) for par in [bi]]) / actual_batch_size)
        upq = T.sum(loss)
        seq_costs = (
            - upq / actual_batch_size +
            0.5 * l2 * seq_l2_sq)
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        update_x = T.set_subtensor(uiq_x, uiq_x - lr * T.grad(seq_costs, self.lt)[uiq_pqs])
        seq_updates.append((self.lt, update_x))     # 会直接更改到seq_updates里
        # ----------------------------------------------------------------------------

        # givens给数据
        start_end = T.ivector()
        self.seq_train = theano.function(
            inputs=[start_end],
            outputs=-upq,
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                qidxs: self.tra_buys_neg_masks[start_end],
                tra_mask: self.tra_masks[start_end]})

    def __theano_predict__(self, n_in, n_hidden):
        """
        测试阶段再跑一遍训练序列得到各个隐层。用全部数据一次性得出所有用户的表达
        """
        ui, wh = self.ui, self.wh
        qh, rh, e, f = self.qh, self.rh, self.e, self.f
        winh = self.window_hidden

        tra_mask = T.imatrix()          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length

        bi = T.alloc(self.bi, actual_batch_size, 3, n_hidden)   # shape=(n, 3, 20), 原维度放在后边
        bi = bi.dimshuffle(1, 2, 0)                             # shape=(3, 20, n)

        pidxs = T.imatrix()
        xps = self.lt[pidxs]                # shape((actual_batch_size, seq_length, n_in))
        xps = xps.dimshuffle(1, 0, 2)       # shape=(seq_length, batch_size, n_in)=(157, n, 20)

        def recurrence(xp_t, h_t_pre1):
            # gru unit
            z_r = sigmoid(T.dot(ui[:2], xp_t.T) +
                          T.dot(wh[:2], h_t_pre1.T) + bi[:2])
            z, r = z_r[0].T, z_r[1].T                           # shape=(n, 20)
            c = tanh(T.dot(ui[2], xp_t.T) +
                     T.dot(wh[2], (r * h_t_pre1).T) + bi[2])
            h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c.T     # shape=(n, 20)
            return h_t

        batch_h0 = T.alloc(self.h0, actual_batch_size, n_hidden)
        h, _ = theano.scan(           # h.shape=(157, n, 20)
            fn=recurrence,
            sequences=xps,
            outputs_info=batch_h0,
            n_steps=seq_length)

        # 每个用户最后一个隐层.
        hs = h.dimshuffle(1, 0, 2)              # shape=(n, 157, 20)
        hts = hs[                               # shape=(n, 20)
            T.arange(actual_batch_size),        # 行。 行列用花式索引a[[1,2,3],[2,5,6]]，需给定行列的表示
            T.sum(tra_mask, axis=1) - 1]        # 列。需要mask是'int32'型的。不能使用a[:, [1,2,3]]。

        # 每个用户最后 winh 个隐层.
        # 每个用户的表达shape=(winh, 20), 最终hwins.shape=(n, winh, 20)
        def extract(hu, mask):
            le = T.sum(mask)
            return hu[le-winh: le]              # shape=(winh, 20)
        chs, _ = theano.scan(                   # shape=(n, winh, 20)
            fn=extract,
            sequences=[hs, tra_mask],
            outputs_info=None,
            n_steps=actual_batch_size)
        # 用每个用户的5个隐层计算context
        ehs = T.dot(tanh(T.dot(chs, qh)), rh)           # shape=(n, winh, 1)
        ehs = T.Rebroadcast((2, True))(ehs)             # axis=2进行broadcast, 使其可被丢掉
        ahs0 = softmax(ehs.dimshuffle(0, 1))            # shape=(n, winh)，降一维。因为softmax按行处理。
        ahs = ahs0.dimshuffle(0, 1, 'x')                # shape=(n, winx, 1), 升一维。还原回去
        ahs = T.Rebroadcast((2, True))(ahs)             # axis=2进行broadcast, 使其可做乘法。
        hcs = T.sum(chs * ahs, axis=1)                  # shape=(n, 20)
        # 整体表达hws，融合当前hts、上下文hcs
        hws = tanh(T.dot(hts, e.T) + T.dot(hcs, f.T))   # shape=(n, 20)

        # givens给数据
        start_end = T.ivector()
        self.seq_predict = theano.function(
            inputs=[start_end],
            outputs=hws,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                tra_mask: self.tra_masks[start_end]})

    def train(self, idxs):
        return self.seq_train(idxs)


# ======================================================================================================================
class HcaGru(GruBasic):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden,
                 window_input, window_hidden):
        super(HcaGru, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        rang = 0.5
        self.window_input = theano.shared(borrow=True, value=np.asarray(window_input, dtype='int32'))
        self.window_hidden = theano.shared(borrow=True, value=np.asarray(window_hidden, dtype='int32'))
        # context_x
        vc = uniform(-rang, rang, (3, n_hidden, n_in))
        qx = uniform(-rang, rang, (n_in, n_in))
        rx = uniform(-rang, rang, (n_in, 1))
        self.vc = theano.shared(borrow=True, value=vc.astype(theano.config.floatX))
        self.qx = theano.shared(borrow=True, value=qx.astype(theano.config.floatX))
        self.rx = theano.shared(borrow=True, value=rx.astype(theano.config.floatX))
        # context_h
        qh = uniform(-rang, rang, (n_hidden, n_hidden))
        rh = uniform(-rang, rang, (n_hidden, 1))
        e = uniform(-rang, rang, (n_hidden, n_hidden))
        f = uniform(-rang, rang, (n_hidden, n_hidden))
        self.qh = theano.shared(borrow=True, value=qh.astype(theano.config.floatX))
        self.rh = theano.shared(borrow=True, value=rh.astype(theano.config.floatX))
        self.e = theano.shared(borrow=True, value=e.astype(theano.config.floatX))
        self.f = theano.shared(borrow=True, value=f.astype(theano.config.floatX))
        self.params = [
            self.ui, self.wh, self.bi,
            self.qx, self.rx, self.vc,
            self.qh, self.rh, self.e, self.f]  # self.lt单独进行更新。
        self.l2_sqr = (
            T.sum(self.lt ** 2) +
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__(n_in, n_hidden)
        self.__theano_predict__(n_in, n_hidden)

    def __theano_train__(self, n_in, n_hidden):
        """
        训练阶段跑一遍训练序列
        """
        # self.alpha_lambda = ['alpha', 'lambda']
        ui, wh, bi = self.ui, self.wh, self.bi
        qx, rx, vc = self.qx, self.rx, self.vc
        qh, rh, e, f = self.qh, self.rh, self.e, self.f
        winx, winh = self.window_input, self.window_hidden

        tra_mask = T.imatrix()          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length
        mask = tra_mask.T                               # shape=(157, n)

        bi = T.alloc(self.bi, actual_batch_size, 3, n_hidden)   # shape=(n, 3, 20), 原维度放在后边
        bi = bi.dimshuffle(1, 2, 0)                             # shape=(3, 20, n)

        pidxs, qidxs = T.imatrix(), T.imatrix()
        xps, xqs = self.lt[pidxs], self.lt[qidxs]   # shape((actual_batch_size, seq_length, n_in))
        xps = xps.dimshuffle(1, 0, 2)               # shape=(seq_length, batch_size, n_in)=(157, n, 20)
        xqs = xqs.dimshuffle(1, 0, 2)

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
        def recurrence(xp_t, xp_t1, xq_t1, mask_t, h_t_pre1, cxs, chs):
            # 特征、隐层都处理成shape=(batch_size, n_hidden)=(n, 20)
            # (n, winx, 20) = T.concatenate((((n, winx-1, 20)), ((n, 1, 20))), axis=1)
            # context_x
            # 定义定长矩阵，xp_t拼接到最底下，删除首行, 矩阵维度不变。
            cxs = T.concatenate((cxs[:, 1:, :],             # shape=(n, winx-1, 20)
                                 xp_t.dimshuffle(0, 'x', 1)),    # shape=(n, 1, 20)
                                axis=1)                     # shape=(n, winx, 20)
            exs = T.dot(tanh(T.dot(cxs, qx)), rx)           # shape=(n, winx, 1)
            exs = T.Rebroadcast((2, True))(exs)             # axis=2进行broadcast, 使其可被丢掉
            axs0 = softmax(exs.dimshuffle(0, 1))            # shape=(n, winx)，降一维。因为softmax按行处理。
            axs = axs0.dimshuffle(0, 1, 'x')                # shape=(n, winx, 1), 升一维。还原回去。
            axs = T.Rebroadcast((2, True))(axs)             # axis=2进行broadcast, 使其可做乘法。
            # (n, 20) = T.sum((n, winx, 20) * (n, winx, 1), axis=1)
            xc = T.sum(cxs * axs, axis=1)                     # shape=(n, 20)
            # gru unit
            z_r = sigmoid(T.dot(ui[:2], xp_t.T) +
                          T.dot(vc[:2], xc.T) +
                          T.dot(wh[:2], h_t_pre1.T) + bi[:2])
            z, r = z_r[0].T, z_r[1].T                           # shape=(n, 20)
            c = tanh(T.dot(ui[2], xp_t.T) +
                     T.dot(vc[2], xc.T) +
                     T.dot(wh[2], (r * h_t_pre1).T) + bi[2])
            h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c.T     # shape=(n, 20)
            # context_h
            # 定义定长矩阵，h_t拼接到最底下，删除首行, 矩阵维度不变。
            chs = T.concatenate((chs[:, 1:, :],             # shape=(n, winh-1, 20)
                                 h_t.dimshuffle(0, 'x', 1)),    # shape=(n, 1, 20)
                                axis=1)                     # shape=(n, winh, 20)
            ehs = T.dot(tanh(T.dot(chs, qh)), rh)           # shape=(n, winh, 1)
            ehs = T.Rebroadcast((2, True))(ehs)             # axis=2进行broadcast, 使其可被丢掉
            ahs0 = softmax(ehs.dimshuffle(0, 1))            # shape=(n, winh)，降一维。因为softmax按行处理。
            ahs = ahs0.dimshuffle(0, 1, 'x')                # shape=(n, winh, 1), 升一维。还原回去
            ahs = T.Rebroadcast((2, True))(ahs)             # axis=2进行broadcast, 使其可做乘法。
            hcs = T.sum(chs * ahs, axis=1)                  # shape=(n, 20)
            # 整体表达hws，融合当前hts、上下文hcs
            hws = tanh(T.dot(h_t, e.T) + T.dot(hcs, f.T))   # shape=(n, 20)
            # loss
            upq_t = T.sum(hws * (xp_t1 - xq_t1), axis=1)    # shape=(n, ), h(t) * (xp(t+1) - xq(t+1)), 正负样本训练。
            loss_t = T.log(sigmoid(upq_t))
            loss_t *= mask_t                                # 只在损失这里乘一下0/1向量就可以了
            return [h_t, cxs, chs, loss_t]

        batch_h0 = T.alloc(self.h0, actual_batch_size, n_hidden)
        cumx = T.alloc(self.lt[-1], actual_batch_size, winx, n_in)         # concatenate
        cumh = T.alloc(self.h0, actual_batch_size, winh, n_hidden)
        [_, _, _, loss], _ = theano.scan(               # h是h1~ht。loss是h0~ht-1和x1~xt计算得到的。
            fn=recurrence,
            sequences=[xps, xps[1:], xqs[1:], mask],
            outputs_info=[batch_h0, cumx, cumh, None],
            n_steps=seq_length-1,
            truncate_gradient=-1)

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        seq_l2_sq = (
            T.sum([T.sum(par ** 2) for par in [xps, xqs, ui, wh, qx, rx, vc, qh, rh, e, f]]) +
            T.sum([T.sum(par ** 2) for par in [bi]]) / actual_batch_size)
        upq = T.sum(loss)
        seq_costs = (
            - upq / actual_batch_size +
            0.5 * l2 * seq_l2_sq)
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        update_x = T.set_subtensor(uiq_x, uiq_x - lr * T.grad(seq_costs, self.lt)[uiq_pqs])
        seq_updates.append((self.lt, update_x))     # 会直接更改到seq_updates里
        # ----------------------------------------------------------------------------

        # givens给数据
        start_end = T.ivector()
        self.seq_train = theano.function(
            inputs=[start_end],
            outputs=-upq,
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                qidxs: self.tra_buys_neg_masks[start_end],
                tra_mask: self.tra_masks[start_end]})

    def __theano_predict__(self, n_in, n_hidden):
        """
        测试阶段再跑一遍训练序列得到各个隐层。用全部数据一次性得出所有用户的表达
        """
        ui, wh = self.ui, self.wh
        qx, rx, vc = self.qx, self.rx, self.vc
        qh, rh, e, f = self.qh, self.rh, self.e, self.f
        winx, winh = self.window_input, self.window_hidden

        tra_mask = T.imatrix()          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length

        bi = T.alloc(self.bi, actual_batch_size, 3, n_hidden)   # shape=(n, 3, 20), 原维度放在后边
        bi = bi.dimshuffle(1, 2, 0)                             # shape=(3, 20, n)

        pidxs = T.imatrix()
        xps = self.lt[pidxs]                # shape((actual_batch_size, seq_length, n_in))
        xps = xps.dimshuffle(1, 0, 2)       # shape=(seq_length, batch_size, n_in)=(157, n, 20)

        def recurrence(xp_t, h_t_pre1, cxs):
            # 特征、隐层都处理成shape=(batch_size, n_hidden)=(n, 20)
            # (n, winx, 20) = T.concatenate((((n, winx-1, 20)), ((n, 1, 20))), axis=1)
            # context_x
            # 定义定长矩阵，xp_t拼接到最底下，删除首行, 矩阵维度不变。
            cxs = T.concatenate((cxs[:, 1:, :],             # shape=(n, winx-1, 20)
                                 xp_t.dimshuffle(0, 'x', 1)),    # shape=(n, 1, 20)
                                axis=1)                     # shape=(n, winx, 20)
            exs = T.dot(tanh(T.dot(cxs, qx)), rx)           # shape=(n, winx, 1)
            exs = T.Rebroadcast((2, True))(exs)             # axis=2进行broadcast, 使其可被丢掉
            axs0 = softmax(exs.dimshuffle(0, 1))            # shape=(n, winx)，降一维。因为softmax按行处理。
            axs = axs0.dimshuffle(0, 1, 'x')                # shape=(n, winx, 1), 升一维。还原回去。
            axs = T.Rebroadcast((2, True))(axs)             # axis=2进行broadcast, 使其可做乘法。
            # (n, 20) = T.sum((n, winx, 20) * (n, winx, 1), axis=1)
            xc = T.sum(cxs * axs, axis=1)                     # shape=(n, 20)
            # gru unit
            z_r = sigmoid(T.dot(ui[:2], xp_t.T) +
                          T.dot(vc[:2], xc.T) +
                          T.dot(wh[:2], h_t_pre1.T) + bi[:2])
            z, r = z_r[0].T, z_r[1].T                           # shape=(n, 20)
            c = tanh(T.dot(ui[2], xp_t.T) +
                     T.dot(vc[2], xc.T) +
                     T.dot(wh[2], (r * h_t_pre1).T) + bi[2])
            h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c.T     # shape=(n, 20)
            return [h_t, cxs, axs0]  # 每处位置的权重也返回, shape=(n, winx)

        batch_h0 = T.alloc(self.h0, actual_batch_size, n_hidden)
        cumx = T.alloc(self.lt[-1], actual_batch_size, winx, n_in)
        [h, _, att], _ = theano.scan(           # h.shape=(157, n, 20)
            fn=recurrence,
            sequences=xps,
            outputs_info=[batch_h0, cumx, None],
            n_steps=seq_length)

        # 逐行获取每个用户最后的 winh 组输入的权重。
        # 因为是对 winh 个 h 做 context，而每个 h 下有 winx 个输入做 context，所以每个用户取出(winh, winx)
        # 每个用户的权重shape=(winh, winx)，最终shape=(n, winh, winx)
        atts = att.dimshuffle(1, 0, 2)          # shape=(n, 157, winx)

        def extract(attu, mask):
            le = T.sum(mask)
            winxh = attu[le-winh: le]
            return winxh.reshape((winh * winxh.shape[1], ))   # shape=(winx*winh, )，当前时间的权重在向量的最右侧
        att_winxh, _ = theano.scan(             # shape=(n, winx*winh)
            fn=extract,
            sequences=[atts, tra_mask],
            outputs_info=None,
            n_steps=actual_batch_size)

        # 每个用户最后一个隐层.
        hs = h.dimshuffle(1, 0, 2)              # shape=(n, 157, 20)
        hts = hs[                               # shape=(n, 20)
            T.arange(actual_batch_size),        # 行。 行列用花式索引a[[1,2,3],[2,5,6]]，需给定行列的表示
            T.sum(tra_mask, axis=1) - 1]        # 列。需要mask是'int32'型的。不能使用a[:, [1,2,3]]。

        # 每个用户最后 winh 个隐层.
        # 每个用户的表达shape=(winh, 20), 最终hwins.shape=(n, winh, 20)
        def extract(hu, mask):
            le = T.sum(mask)
            return hu[le-winh: le]              # shape=(winh, 20)
        chs, _ = theano.scan(                   # shape=(n, winh, 20)
            fn=extract,
            sequences=[hs, tra_mask],
            outputs_info=None,
            n_steps=actual_batch_size)
        # 用每个用户的5个隐层计算context
        ehs = T.dot(tanh(T.dot(chs, qh)), rh)           # shape=(n, winh, 1)
        ehs = T.Rebroadcast((2, True))(ehs)             # axis=2进行broadcast, 使其可被丢掉
        ahs0 = softmax(ehs.dimshuffle(0, 1))            # shape=(n, winh)，降一维。因为softmax按行处理。
        ahs = ahs0.dimshuffle(0, 1, 'x')                # shape=(n, winx, 1), 升一维。还原回去
        ahs = T.Rebroadcast((2, True))(ahs)             # axis=2进行broadcast, 使其可做乘法。
        hcs = T.sum(chs * ahs, axis=1)                  # shape=(n, 20)
        # 整体表达hws，融合当前hts、上下文hcs
        hws = tanh(T.dot(hts, e.T) + T.dot(hcs, f.T))   # shape=(n, 20)

        # 序列尾端的：winh组winx个的输入权重，winh个隐层权重。shape=(n, winx*winh+winh)
        all_att_winxh_winh = T.concatenate((att_winxh, ahs0), axis=1)

        # givens给数据
        start_end = T.ivector()
        self.seq_predict = theano.function(
            inputs=[start_end],
            outputs=[hws, all_att_winxh_winh],
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                tra_mask: self.tra_masks[start_end]})

    def train(self, idxs):
        return self.seq_train(idxs)


@exe_time
def main():
    print('... construct the class: HCA-GRU')


if '__main__' == __name__:
    main()
