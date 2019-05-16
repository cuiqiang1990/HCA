#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:       Qiang Cui:  <cuiqiang1990[at]hotmail.com>
# Descripton:   construct the class BPR
# Create Date:  2017-02-12 21:30:00
# Modify Date:  2016-00-00 00:00:00
# Modify Disp:

from __future__ import print_function
import time
import numpy as np
from numpy.random import uniform
import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid, relu
from theano.tensor import tanh
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.extra_ops import Unique
from theano.tensor.nnet.nnet import softmax     # 作用于2d-matrix，按行处理。
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


from theano.tensor import exp
def hsoftmax(x):
    # 竖直方向取softmax。
    # theano里的是作用于2d-matrix，按行处理。我文中scan里有一步是处理(n,)，直接用会报错，所以要重写。
    # 按axis=0处理(n, )，也就是按第一个axis做，会超级方便。
    e_x = exp(x - x.max(axis=0, keepdims=True))
    out = e_x / e_x.sum(axis=0, keepdims=True)
    return out


# ======================================================================================================================
class MArankBasic(object):
    def __init__(self, train, test, train_set, alpha_lambda, n_user, n_item, n_in, n_hidden, layer):
        """
        构建 模型参数
        :param train: 添加mask后的
        :param test: 添加mask后的
        :param n_user: 用户的真实数目
        :param n_item: 商品items的真正数目，init()里补全一个商品作为填充符
        :param n_in: rnn输入向量的维度
        :param n_hidden: rnn隐层向量的维度
        :return:
        """
        # 来自于theano官网的dAE部分。
        rng = np.random.RandomState(123)
        self.thea_rng = RandomStreams(rng.randint(2 ** 30))     # 旗下随机函数可在GPU下运行。
        # 用mask进行补全后的train/test
        tra_buys_masks, tra_masks, tra_buys_neg_masks = train
        tes_buys_masks, tes_masks, tes_buys_neg_masks = test
        self.tra_buys_masks = theano.shared(borrow=True, value=np.asarray(tra_buys_masks, dtype='int32'))
        self.tes_buys_masks = theano.shared(borrow=True, value=np.asarray(tes_buys_masks, dtype='int32'))
        self.tra_masks = theano.shared(borrow=True, value=np.asarray(tra_masks, dtype='int32'))
        self.tes_masks = theano.shared(borrow=True, value=np.asarray(tes_masks, dtype='int32'))
        self.tra_buys_neg_masks = theano.shared(borrow=True, value=np.asarray(tra_buys_neg_masks, dtype='int32'))
        self.tes_buys_neg_masks = theano.shared(borrow=True, value=np.asarray(tes_buys_neg_masks, dtype='int32'))
        # 把训练集中的每个user sequence打包成一个一个的set。也要用mask做成等长，后边predict时用。
        # 比如set_size=3, 则一个user seq每个时刻都用3个item生成小set，即每个user seq变成嵌套list。
        self.tra_set_masks = theano.shared(borrow=True, value=np.asarray(train_set, dtype='int32'))
        # 把超参数shared
        self.alpha_lambda = theano.shared(borrow=True, value=np.asarray(alpha_lambda, dtype=theano.config.floatX))
        # ===========================================================================================
        # item是两套参数，但tra、tes一套就可以，因为他们是idx。索引时注意从两套分别取出即可。
        # 初始化，先定义局部变量，再self.修饰成实例变量
        rang = 0.5
        ux = uniform(-rang, rang, (n_user, n_in))       # 直接定义
        uc = uniform(-rang, rang, (n_user, n_in))       # 由模型中计算得到
        lx = uniform(-rang, rang, (n_item + 1, n_in))   # shape=(n_item, 20)
        lc = uniform(-rang, rang, (n_item + 1, n_in))   # 对应上下文建模的部分。
        self.ux = theano.shared(borrow=True, value=ux.astype(theano.config.floatX))
        self.uc = theano.shared(borrow=True, value=uc.astype(theano.config.floatX))
        self.lx = theano.shared(borrow=True, value=lx.astype(theano.config.floatX))
        self.lc = theano.shared(borrow=True, value=lc.astype(theano.config.floatX))
        # 各种权重矩阵 ====================================================================================
        # ResNet迭代矩阵. 先设置为各层共享。
        self.layer = layer                          # resnet做几层。
        wru = uniform(-rang, rang, (layer, n_hidden, n_hidden))
        wrc = uniform(-rang, rang, (layer, n_hidden, n_hidden))
        wrl = uniform(-rang, rang, (layer, n_hidden, n_hidden))
        # bru = np.zeros((layer, n_hidden), dtype=theano.config.floatX)
        # brc = np.zeros((layer, n_hidden, n_hidden), dtype=theano.config.floatX)
        self.wru = theano.shared(borrow=True, value=wru.astype(theano.config.floatX))
        self.wrc = theano.shared(borrow=True, value=wrc.astype(theano.config.floatX))
        self.wrl = theano.shared(borrow=True, value=wrl.astype(theano.config.floatX))
        # 一阶att的权重矩阵，所用于ResNet各层。很明确，是各层共享的。
        wa1 = uniform(-rang, rang, (n_hidden, ))
        wa2 = uniform(-rang, rang, (n_hidden, n_hidden))
        wa3 = uniform(-rang, rang, (n_hidden, n_hidden))
        # ba1 = np.zeros((n_hidden, ), dtype=theano.config.floatX)
        # ba2 = np.zeros((n_hidden, ), dtype=theano.config.floatX)
        self.wa1 = theano.shared(borrow=True, value=wa1.astype(theano.config.floatX))
        self.wa2 = theano.shared(borrow=True, value=wa2.astype(theano.config.floatX))
        self.wa3 = theano.shared(borrow=True, value=wa3.astype(theano.config.floatX))
        # 二阶att的权重矩阵
        wb1 = uniform(-rang, rang, (n_hidden, ))
        wb2 = uniform(-rang, rang, (n_hidden, n_hidden))
        # bb1 = np.zeros((n_hidden, ), dtype=theano.config.floatX)
        # bb2 = np.zeros((n_hidden, ), dtype=theano.config.floatX)
        self.wb1 = theano.shared(borrow=True, value=wb1.astype(theano.config.floatX))
        self.wb2 = theano.shared(borrow=True, value=wb2.astype(theano.config.floatX))
        # 存放训练好的users、items表达。用于计算所有users对所有items的评分：users * items
        trained_usr_x = uniform(-rang, rang, (n_user, n_hidden))
        trained_usr_c = uniform(-rang, rang, (n_user, n_hidden))
        trained_usr_l = uniform(-rang, rang, (n_user, n_hidden))
        trained_itm_x = uniform(-rang, rang, (n_item + 1, n_hidden))
        trained_itm_c = uniform(-rang, rang, (n_item + 1, n_hidden))
        self.trained_usr_x = theano.shared(borrow=True, value=trained_usr_x.astype(theano.config.floatX))
        self.trained_usr_c = theano.shared(borrow=True, value=trained_usr_c.astype(theano.config.floatX))
        self.trained_usr_l = theano.shared(borrow=True, value=trained_usr_l.astype(theano.config.floatX))
        self.trained_itm_x = theano.shared(borrow=True, value=trained_itm_x.astype(theano.config.floatX))
        self.trained_itm_c = theano.shared(borrow=True, value=trained_itm_c.astype(theano.config.floatX))

    def update_neg_masks(self, tra_buys_neg_masks, tes_buys_neg_masks):
        # 每个epoch都更新负样本
        self.tra_buys_neg_masks.set_value(np.asarray(tra_buys_neg_masks, dtype='int32'), borrow=True)
        self.tes_buys_neg_masks.set_value(np.asarray(tes_buys_neg_masks, dtype='int32'), borrow=True)

    def update_trained_items(self):
        # 更新最终的items表达
        lx = self.lx.get_value(borrow=True)         # self.lt是shared，用get_value()。
        lc = self.lc.get_value(borrow=True)
        self.trained_itm_x.set_value(np.asarray(lx, dtype=theano.config.floatX), borrow=True)     # update
        self.trained_itm_c.set_value(np.asarray(lc, dtype=theano.config.floatX), borrow=True)

    def update_trained_users(self):
        # 更新最终的users表达
        ux = self.ux.get_value(borrow=True)
        self.trained_usr_x.set_value(np.asarray(ux, dtype=theano.config.floatX), borrow=True)  # update

    def update_trained_users_att(self, uc, ul):
        self.trained_usr_c.set_value(np.asarray(uc, dtype=theano.config.floatX), borrow=True)
        self.trained_usr_l.set_value(np.asarray(ul, dtype=theano.config.floatX), borrow=True)

    def compute_sub_all_scores(self, start_end):    # 其实可以直接传过来实数参数
        # 计算users * items，每个用户对所有商品的评分(需去掉填充符)
        sub_all_scores_x = T.dot(self.trained_usr_x[start_end], self.trained_itm_x[:-1].T)
        sub_all_scores_c = T.dot(self.trained_usr_c[start_end], self.trained_itm_c[:-1].T)
        sub_all_scores_l = T.dot(self.trained_usr_l[start_end], self.trained_itm_c[:-1].T)
        subs = (sub_all_scores_x + sub_all_scores_c + sub_all_scores_l).eval()     # shape=(sub_n_user, n_item)
        return subs

    def compute_sub_auc_preference(self, start_end):
        # items.shape=(n_item+1, 20)，因为是mask形式，所以需要填充符。
        # 注意矩阵形式的索引方式。
        tes_itm_x = self.trained_itm_x[self.tes_buys_masks[start_end]]  # shape=(sub_n_user, len(tes_mask[0]), n_hidden)
        tes_itm_c = self.trained_itm_c[self.tes_buys_masks[start_end]]
        tes_itm_neg_x = self.trained_itm_x[self.tes_buys_neg_masks[start_end]]
        tes_itm_neg_c = self.trained_itm_c[self.tes_buys_neg_masks[start_end]]
        usr_x = self.trained_usr_x[start_end]
        usr_c = self.trained_usr_c[start_end]
        usr_l = self.trained_usr_l[start_end]
        shp0, shp2 = usr_x.shape        # shape=(sub_n_user, n_hidden)
        # 利用性质：(n_user, 1, n_hidden) * (n_user, len, n_hidden) = (n_user, len, n_hidden)，即broadcast
        # 利用性质：np.sum((n_user, len, n_hidden), axis=2) = (n_user, len)，
        #         即得到各用户对test里正负样本的偏好值
        all_upqs_x = T.sum(usr_x.reshape((shp0, 1, shp2)) * (tes_itm_x - tes_itm_neg_x), axis=2)
        all_upqs_c = T.sum(usr_c.reshape((shp0, 1, shp2)) * (tes_itm_c - tes_itm_neg_c), axis=2)
        all_upqs_l = T.sum(usr_l.reshape((shp0, 1, shp2)) * (tes_itm_c - tes_itm_neg_c), axis=2)
        all_upqs = all_upqs_x + all_upqs_c + all_upqs_l
        all_upqs *= self.tes_masks[start_end]       # 只保留原先test items对应有效位置的偏好值
        return all_upqs.eval() > 0                  # 将>0的标为True, 也就是1

    def get_corrupted_input_whole(self, inp, corruption_prob):
        # 处理2D矩阵：randomly set whole feature to zero. Matrix.shape=(n, m)
        # denoising方式0：随机将某些图、文特征整体性置为0
        # 比如原先一条序列的图像特征是(num, 1024); 那么0/1概率矩阵是(num, 1), T.Rebroadcast，再相乘
        # if corruption_prob < 0. or corruption_prob >= 1.:
        #     raise Exception('Drop prob must be in interval [0, 1)')
        retain_prob = 1. - corruption_prob
        randoms = self.thea_rng.binomial(
            size=(inp.shape[0], 1),                 # shape=(num, 1)
            n=1,
            p=retain_prob,                          # p是得1的概率。
            dtype=theano.config.floatX)
        randoms = T.Rebroadcast((1, True))(randoms)
        return inp * randoms                        # shape=(num, 1024)

    def get_corrupted_input_whole_minibatch(self, inp, corruption_prob):
        # 亲测表明：在序列前做data的corruption，效果更好更稳定。
        # 处理3D矩阵
        retain_prob = 1. - corruption_prob
        randoms = self.thea_rng.binomial(
            size=(inp.shape[0], inp.shape[1], 1),   # shape=(seq_length, batch_size, 1)
            n=1,
            p=retain_prob,                          # p是得1的概率。
            dtype=theano.config.floatX)
        randoms = T.Rebroadcast((2, True))(randoms)
        return inp * randoms                        # shape=(seq_length, batch_size, 1024)

    def dropout(self, inp, drop_prob):
        # 处理向量：randomly set some positions to zero. Vector.shape=(n, )
        # 例如一个向量20维，就有20个位置，也就是有20个神经元。
        # train时做dropout，test时还是全连接。
        # if drop_prob < 0. or drop_prob >= 1.:
        #     raise Exception('Drop prob must be in interval [0, 1)')
        retain_prob = 1. - drop_prob        # 取0.5就可以了。
        randoms = self.thea_rng.binomial(
            size=inp.shape,                 # 生成与向量inp同样维度的0、1向量
            n=1,                            # 每个神经元实验一次
            p=retain_prob)                  # 每个神经元*1的概率为p/retain_prob。*0的概率为drop_prob
        inp *= randoms                      # 屏蔽某些神经元，重置为0
        inp /= retain_prob                  # drop完后需要rescale，以维持inp在dropout前后的数值和(sum)不变。
        return inp                          # 直接本地修改inp，所以调用时'self.dropout(x, 0.5)'即可直接本地修改输入x。

    def __theano_predict__(self):
        """
        测试阶段再跑一遍训练序列得到各个隐层。用全部数据一次性得出所有用户的表达
        """
        # 各种权重矩阵。【注意：统一格式，权重 * 变量】
        lay = self.layer
        wru, wrc, wrl = self.wru, self.wrc, self.wrl     # resnet
        wa1, wa2, wa3 = self.wa1, self.wa2, self.wa3                    # 一阶att
        wb1, wb2 = self.wb1, self.wb2                                   # 二阶att

        # givens给数据
        start_end = T.ivector()
        tra_mask = T.imatrix()          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        # user vector
        urxs = T.fmatrix()                      # shape=(batch_size, d)
        cps_idxs = T.itensor3()                 # shape=(batch_size, 各用户set形式的序列)
        cpt_idxs = cps_idxs[                    # shape=(batch_size, set_size)
            T.arange(actual_batch_size),        # 花式索引，取出每个用户序列的最后一组item_idxs。
            T.sum(tra_mask, axis=1) - 1]
        # item vector (每个user一个set)
        cpts = self.lc[cpt_idxs]                 # shape=(batch_size, set_size, d)

        # ==============================================================================================================
        # 得分2   # 第0层的att, 获得(batch_size, d)的att vector。
        urx_emb = T.dot(wa2, urxs.T).T.dimshuffle(0, 'x', 1)    # shape=(batch_size, 1, d)
        e0 = T.dot(tanh(urx_emb + T.dot(cpts, wa3)), wa1)       # shape=(batch_size, set_size)
        a0 = softmax(e0)                                        # (batch_size, set_size)
        c0 = T.sum(cpts * a0.dimshuffle(0, 1, 'x'), axis=1)     # shape=(batch_size, d), broadcast
        # 得分2   # ResNet里的att
        def recurrence1(wrut, wrct, urx_pre1, cpt_pre1):
            # ResNet更新
            ur_t = relu(T.dot(wrut, urx_pre1.T).T + urx_pre1)    # (batch_size, d)
            cp_t = relu(T.dot(cpt_pre1, wrct) + cpt_pre1)        # (batch_size, set_size, d)
            # att计算生成上下文向量
            ur_t_emb = T.dot(wa2, ur_t.T).T.dimshuffle(0, 'x', 1)
            e_t = T.dot(tanh(ur_t_emb + T.dot(cp_t, wa3)), wa1) # shape=(batch_size, set_size)
            a_t = softmax(e_t)
            c_t = T.sum(cp_t * a_t.dimshuffle(0, 1, 'x'), axis=1)

            return [ur_t, cp_t, c_t]        # (batch_size, d), (batch_size, set_size, d), (batch_size, d)
        [urs, cps, cs], _ = theano.scan(    # cs.shape = (layer, batch_size, d)
            fn=recurrence1,
            sequences=[wru, wrc],
            outputs_info=[urxs, cpts, None],
            n_steps=lay,
            truncate_gradient=-1)

        # 得分2   # 二阶att
        c0 = c0.dimshuffle(0, 'x', 1)               # (batch_size, 1, d)
        cs = cs.dimshuffle(1, 0, 2)                 # (batch_size, layer, d)
        context = T.concatenate((c0, cs), axis=1)   # (batch_size, layer+1, d)
        e1 = T.dot(tanh(T.dot(context, wb2)), wb1)  # shape=(batch_size, layer+1)
        a1 = softmax(e1)
        c1 = T.sum(context * a1.dimshuffle(0, 1, 'x'), axis=1)  # shape=(batch_size, d)

        # ==============================================================================================================
        # 得分3   # 以resnet的输出c1重新计算一个新的resnet
        def recurrence2(wrlt, h_pre1):
            # ResNet更新
            hl_t = relu(T.dot(wrlt, h_pre1.T).T + h_pre1)   # shape=(batch_size, d)
            return hl_t
        hls, _ = theano.scan(
            fn=recurrence2,
            sequences=wrl,
            outputs_info=c1,
            n_steps=lay,
            truncate_gradient=-1)

        # ==============================================================================================================
        # 最终总的user vector。经由resnet计算得到的部分
        usr_vec_c = c1
        usr_vec_l = hls[-1]

        self.seq_predict = theano.function(
            inputs=[start_end],
            outputs=[usr_vec_c, usr_vec_l],             # shape=(batch_size, d)
            givens={
                urxs: self.trained_usr_x[start_end],    # shape=(batch_size, d)
                tra_mask: self.tra_masks[start_end],
                cps_idxs: self.tra_set_masks[start_end]})

    def predict(self, idxs):
        return self.seq_predict(idxs)


# ======================================================================================================================
class OboMArank(MArankBasic):
    def __init__(self, train, test, train_set, alpha_lambda, n_user, n_item, n_in, n_hidden, layer):
        super(OboMArank, self).__init__(train, test, train_set, alpha_lambda, n_user, n_item, n_in, n_hidden, layer)
        self.u_i = [self.ux, self.lx, self.lc]   # 两个都是单独进行更新。
        self.params = [
            self.wru, self.wrc, self.wrl,            # self.brc,  self.bru,
            self.wa1, self.wa2, self.wa3,   # self.ba1, self.ba2,
            self.wb1, self.wb2]             # , self.bb1, self.bb2
        self.l2_sqr = (
            T.sum([T.sum(param ** 2) for param in self.u_i]) +
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__()
        self.__theano_predict__()

    def __theano_train__(self):
        """
        训练阶段跑一遍训练序列
        """
        # 各种usr_itm输入
        uidx, pqidx, cidx = T.iscalar(), T.ivector(), T.ivector()
        urx = self.ux[uidx]     # shape=(n_in, )                        # user全局变量。
        xpq = self.lx[pqidx]    # shape=(2, n_in), 这俩肯定不是同一个item。 # t+1时刻的正负样本
        cpt = self.lc[cidx]     # shape=(set_size, d)                   # t时刻的items输入。需要unique后再做更新。
        cpq = self.lc[pqidx]    # shape=(2, d)                          # t+1时刻的正负样本（该方法中两套item表达）
        # 每时刻输入一个item_set，做unique
        cpqs = T.concatenate((cidx, pqidx))          # 先拼接
        uiq_cps = Unique(False, False, False)(cpqs)  # 去重
        uiq_c = self.lc[uiq_cps]                     # 相应的items特征
        # 各种权重矩阵。【注意：统一格式，权重 * 变量】
        lay = self.layer
        wru, wrc, wrl = self.wru, self.wrc, self.wrl                    # resnet
        wa1, wa2, wa3 = self.wa1, self.wa2, self.wa3                    # 一阶att
        wb1, wb2 = self.wb1, self.wb2                                   # 二阶att

        """
        输入t时刻正负样本，计算当前损失并更新user/正负样本. 公式里省略了时刻t
        # 根据性质：T.dot((n, ), (n, ))得到(1, 1)
            uij  = user * (xp - xq)
            upq = log(sigmoid(uij))
        """
        # ==============================================================================================================
        # 得分1
        uij_x = T.dot(urx, xpq[0] - xpq[1])

        # ==============================================================================================================
        # 得分2, ResNet部分。
        # -----------------------------------------------------------
        # # check: 向量 + 矩阵, (5, ) + (3, 5) -> (3, 5)
        # rang = 0.5
        # wi = uniform(-rang, rang, (5, 5))    # d * d
        # ii = uniform(-rang, rang, (3, 5))    # 3 itm * d
        # wu = uniform(-rang, rang, (5, 5))    # d * d
        # uu = uniform(-rang, rang, (5, ))     # (d, )
        # import numpy as np
        # a = np.dot(wu, uu)          # (5, )
        # b = np.dot(ii, wi)          # (3, 5)
        # c = np.dot(wi.T, ii.T).T    # b = c
        # d = a + b   # (3, 5)，a可以正常添加到b的每一行中
        # -----------------------------------------------------------
        # 得分2   # 第0层的att
        e0 = T.dot(tanh(T.dot(wa2, urx) + T.dot(cpt, wa3)), wa1)    # (size, )
        a0 = hsoftmax(e0)                                   # (size, )
        c0 = T.sum(cpt * a0.dimshuffle(0, 'x'), axis=0)     # (d, )
        # 得分2   # ResNet里的att
        def recurrence1(wrut, wrct, urx_pre1, cpt_pre1):
            # ResNet更新
            ur_t = relu(T.dot(wrut, urx_pre1) + urx_pre1)   # (d, )
            cp_t = relu(T.dot(cpt_pre1, wrct) + cpt_pre1)   # (size, d)
            # att计算生成上下文向量
            e_t = T.dot(tanh(T.dot(wa2, ur_t) + T.dot(cp_t, wa3)), wa1)
            a_t = hsoftmax(e_t)         # (size, )
            c_t = T.sum(cp_t * a_t.dimshuffle(0, 'x'), axis=0)      # (d, )

            return [ur_t, cp_t, c_t]
        [urs, cps, cs], _ = theano.scan(
            fn=recurrence1,
            sequences=[wru, wrc],    # bru, brc
            outputs_info=[urx, cpt, None],
            n_steps=lay,
            truncate_gradient=-1)
        # 得分2   # 二阶att
        c0 = c0.dimshuffle('x', 0)                      # (1, d)
        context = T.concatenate((c0, cs), axis=0)       # shape=(layer+1, d)
        e1 = T.dot(tanh(T.dot(context, wb2)), wb1)      # shape=(layer+1, )
        a1 = hsoftmax(e1)
        c1 = T.sum(context * a1.dimshuffle(0, 'x'), axis=0) # shape=(d, )
        # 得分2
        uij_c = T.dot(c1, cpq[0] - cpq[1])

        # ==============================================================================================================
        # 得分3   # 以resnet的输出c1重新计算一个新的resnet
        def recurrence2(wrlt, h_pre1):
            # ResNet更新
            hl_t = relu(T.dot(wrlt, h_pre1) + h_pre1)   # (d, )
            return hl_t
        hls, _ = theano.scan(
            fn=recurrence2,
            sequences=wrl,
            outputs_info=c1,
            n_steps=lay,
            truncate_gradient=-1)
        # 得分3
        uij_l = T.dot(hls[-1], cpq[0] - cpq[1])

        # ==============================================================================================================
        # 总的得分
        loss = T.log(sigmoid(uij_x + uij_c + uij_l))

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, L2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        l2_sqr = (
            T.sum([T.sum(par ** 2) for par in [urx, xpq, cpt, cpq,
                                               wru, wrc, wrl,
                                               wa1, wa2, wa3,
                                               wb1, wb2]]))
        upq = loss
        costs = (
            - upq +
            0.5 * l2 * l2_sqr)
        # self.params
        grads = T.grad(costs, self.params)
        updates = [(par, par - lr * gra) for par, gra in zip(self.params, grads)]
        # 1个user，2个items，这种更新求导是最快的。直接对sub求导，并非对par求导。
        subs_pars_idxs = [[urx, self.ux, uidx], [xpq, self.lx, pqidx], [uiq_c, self.lc, uiq_cps]]
        tmp = [(par, T.set_subtensor(sub, sub - lr * T.grad(costs, par)[idx]))
               for sub, par, idx in subs_pars_idxs]
        updates.extend(tmp)
        # ----------------------------------------------------------------------------

        # 输入用户、正负样本及其它参数后，更新变量，返回损失。
        self.train = theano.function(
            inputs=[uidx, pqidx, cidx],
            outputs=-upq,
            updates=updates,
            on_unused_input='warning')

    def train(self, u_idx, pq_idx):
        return self.train(u_idx, pq_idx)


# ======================================================================================================================
class MArank(MArankBasic):
    def __init__(self, train, test, train_set, alpha_lambda, n_user, n_item, n_in, n_hidden, layer):
        super(MArank, self).__init__(train, test, train_set, alpha_lambda, n_user, n_item, n_in, n_hidden, layer)
        self.u_i = [self.ux, self.lx, self.lc]   # 两个都是单独进行更新。
        self.params = [
            self.wru, self.wrc, self.wrl,            # self.brc,  self.bru,
            self.wa1, self.wa2, self.wa3,   # self.ba1, self.ba2,
            self.wb1, self.wb2]             # , self.bb1, self.bb2
        self.l2_sqr = (
            T.sum([T.sum(param ** 2) for param in self.u_i]) +
            T.sum([T.sum(param ** 2) for param in self.params]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr)
        self.__theano_train__()
        self.__theano_predict__()

    def __theano_train__(self):
        """
        训练阶段跑一遍训练序列
        """
        # self.alpha_lambda = ['alpha', 'lambda']
        # 各种usr_itm输入
        uidxs = T.ivector()     # n个用户
        pqidxs = T.imatrix()    # (2, n) 0行: n个正样本。1行: 负样本s。
        cidxs = T.imatrix()     # (n, set_size)
        mask = T.ivector()      # 当前时刻的mask，标明哪些用户的行为有效/无效。
        urxs = self.ux[uidxs]       # shape=(n, d)
        xpqs = self.lx[pqidxs]      # shape=(2, n, d)
        cpts = self.lc[cidxs]       # shape=(n, set_size, d)
        cpqs = self.lc[pqidxs]      # shape=(2, n, d)
        actual_batch_size = mask.shape[0]
        # 每时刻输入一个item_set，做unique
        ncpqs = T.concatenate((cidxs, pqidxs.T), axis=1)    # 先拼接, shape=(n, set_size+2)
        uiq_cps = Unique(False, False, False)(ncpqs)        # 去重
        uiq_c = self.lc[uiq_cps]                            # 相应的items特征
        # 各种权重矩阵。【注意：统一格式，权重 * 变量】
        lay = self.layer
        wru, wrc, wrl = self.wru, self.wrc, self.wrl                    # resnet
        wa1, wa2, wa3 = self.wa1, self.wa2, self.wa3                    # 一阶att
        wb1, wb2 = self.wb1, self.wb2                                   # 二阶att

        """
        输入t时刻正负样本，计算当前损失并更新user/正负样本. 公式里省略了时刻t
        # 根据性质：T.dot((n, ), (n, ))得到(1, 1)
            uij  = user * (xp - xq)
            upq = log(sigmoid(uij))
        """
        # ==============================================================================================================
        # 得分1
        uij_x = T.sum(urxs * (xpqs[0] - xpqs[1]), axis=1)   # shape=(n, )

        # ==============================================================================================================
        # 得分2   # 第0层的att, 获得(batch_size, d)的att vector。
        urx_emb = T.dot(wa2, urxs.T).T.dimshuffle(0, 'x', 1)    # shape=(batch_size, 1, d)
        e0 = T.dot(tanh(urx_emb + T.dot(cpts, wa3)), wa1)       # shape=(batch_size, set_size)
        a0 = softmax(e0)                                        # (batch_size, set_size)
        c0 = T.sum(cpts * a0.dimshuffle(0, 1, 'x'), axis=1)     # shape=(batch_size, d), broadcast
        # 得分2   # ResNet里的att
        def recurrence1(wrut, wrct, urx_pre1, cpt_pre1):
            # ResNet更新
            ur_t = relu(T.dot(wrut, urx_pre1.T).T + urx_pre1)    # (batch_size, d)
            cp_t = relu(T.dot(cpt_pre1, wrct) + cpt_pre1)        # (batch_size, set_size, d)
            # att计算生成上下文向量
            ur_t_emb = T.dot(wa2, ur_t.T).T.dimshuffle(0, 'x', 1)
            e_t = T.dot(tanh(ur_t_emb + T.dot(cp_t, wa3)), wa1) # shape=(batch_size, set_size)
            a_t = softmax(e_t)
            c_t = T.sum(cp_t * a_t.dimshuffle(0, 1, 'x'), axis=1)

            return [ur_t, cp_t, c_t]        # (batch_size, d), (batch_size, set_size, d), (batch_size, d)
        [urs, cps, cs], _ = theano.scan(    # cs.shape = (layer, batch_size, d)
            fn=recurrence1,
            sequences=[wru, wrc],
            outputs_info=[urxs, cpts, None],
            n_steps=lay,
            truncate_gradient=-1)
        # 得分2   # 二阶att
        c0 = c0.dimshuffle(0, 'x', 1)               # (batch_size, 1, d)
        cs = cs.dimshuffle(1, 0, 2)                 # (batch_size, layer, d)
        context = T.concatenate((c0, cs), axis=1)   # (batch_size, layer+1, d)
        e1 = T.dot(tanh(T.dot(context, wb2)), wb1)  # shape=(batch_size, layer+1)
        a1 = softmax(e1)
        c1 = T.sum(context * a1.dimshuffle(0, 1, 'x'), axis=1)  # shape=(batch_size, d)
        # 得分2
        uij_c = T.sum(c1 * (cpqs[0] - cpqs[1]), axis=1)         # shape=(n, )

        # ==============================================================================================================
        # 得分3   # 以resnet的输出c1重新计算一个新的resnet
        def recurrence2(wrlt, h_pre1):
            # ResNet更新
            hl_t = relu(T.dot(wrlt, h_pre1.T).T + h_pre1)       # shape=(batch_size, d)
            return hl_t
        hls, _ = theano.scan(
            fn=recurrence2,
            sequences=wrl,
            outputs_info=c1,
            n_steps=lay,
            truncate_gradient=-1)
        # 得分3
        uij_l = T.sum(hls[-1] * (cpqs[0] - cpqs[1]), axis=1)   # shape=(n, )

        # ==============================================================================================================
        # 总的得分
        loss = T.log(sigmoid(uij_x + uij_c + uij_l))    # shape=(n,)    #
        loss *= mask                    # 只在损失这里乘一下0/1向量就可以了

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, L2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        l2_sqr = (
            T.sum([T.sum(par ** 2) for par in [urxs, xpqs, cpts, cpqs,
                                               wru, wrc, wrl,
                                               wa1, wa2, wa3,
                                               wb1, wb2]]))
        upq = T.sum(loss) / actual_batch_size
        costs = (
            - upq +
            0.5 * l2 * l2_sqr)
        # self.params
        grads = T.grad(costs, self.params)
        updates = [(par, par - lr * gra) for par, gra in zip(self.params, grads)]
        # 1个user，2个items，这种更新求导是最快的。直接对sub求导，并非对par求导。
        subs_pars_idxs = [[urxs, self.ux, uidxs], [xpqs, self.lx, pqidxs], [uiq_c, self.lc, uiq_cps]]
        tmp = [(par, T.set_subtensor(sub, sub - lr * T.grad(costs, par)[idx]))
               for sub, par, idx in subs_pars_idxs]
        updates.extend(tmp)
        # ----------------------------------------------------------------------------

        # 输入用户、正负样本及其它参数后，更新变量，返回损失。
        self.train = theano.function(
            inputs=[uidxs, pqidxs, cidxs, mask],
            outputs=-upq,
            updates=updates,
            on_unused_input='warning')

    def train(self, u_idx, pq_idx):
        return self.train(u_idx, pq_idx)


@exe_time  # 放到待调用函数的定义的上一行
def main():
    pass


if '__main__' == __name__:
    main()
