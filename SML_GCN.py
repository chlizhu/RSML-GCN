# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import tensorflow as tf
import numpy as np
from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
from utils import dropout_sparse
from gcn_model import GCNModel
from opt import Optimizer
from scipy.sparse import lil_matrix, dok_matrix
import scipy.sparse as sp
from utils import *

class model(object):
    def __init__(self, hidden_size, user_num, prd_num, gama):
        self.hidden_size = hidden_size
        self.user_num = user_num
        self.prd_num = prd_num
        self.init = 1 / (self.hidden_size ** 0.5)
        self.gama = gama

        print("SML.")

    def build_work(self):

        # 定义SML输入数据的占位符
        self.userid = tf.placeholder(tf.int32, [None], name="user_id")
        self.prdid = tf.placeholder(tf.int32, [None], name="prd_id")
        self.neg_prdid = tf.placeholder(tf.int32, [None], name='neg_prdid')
        self.neg_userid = tf.placeholder(tf.int32, [None], name='neg_userid')

        # 定义变量
        U = tf.Variable(tf.random_normal([self.user_num, self.hidden_size], stddev=self.init), dtype=tf.float32)
        P = tf.Variable(tf.random_normal([self.prd_num, self.hidden_size], stddev=self.init), dtype=tf.float32)


        with tf.name_scope(name="B"):
            B = tf.Variable(np.array([1.0] * self.user_num), dtype=tf.float32, trainable=True)  # 初始化用户的margin全为1.0

            B1 = tf.Variable(np.array([1.0] * self.prd_num), dtype=tf.float32, trainable=True)

        #定义model
        bias = tf.nn.embedding_lookup(B, self.userid)  # 用户对应的margin
        user_embedding = tf.nn.embedding_lookup(U, self.userid)
        pbias = tf.nn.embedding_lookup(B1, self.prdid)  # 产品对应的margin
        prd_embedding = tf.nn.embedding_lookup(P, self.prdid)  #正的项目
        neg_prd_embedding = tf.nn.embedding_lookup(P, self.neg_prdid)   #用户对应负的项目
        neg_user_embedding = tf.nn.embedding_lookup(U, self.neg_userid)   #项目对应负的用户


        self.pred_distance = tf.reduce_sum(tf.square(user_embedding - prd_embedding), 1)    #正的用户与正的项目的距离（u，v）and 正的项目与正的用户的距离（v, u）

        self.pred_distance_neg = tf.reduce_sum(
            tf.multiply(user_embedding - neg_prd_embedding, user_embedding - neg_prd_embedding), 1)   #正的用户与负的项目的距离 (u, v-)

        self.pred_distance_UN = tf.reduce_sum(
            tf.multiply(prd_embedding - neg_user_embedding, prd_embedding - neg_user_embedding), 1)    #正的项目与负的用户的距离 (v, u-)

        a = tf.maximum(self.pred_distance - self.pred_distance_neg + bias, 0)
        b = tf.maximum(self.pred_distance - self.pred_distance_UN + pbias, 0)



        self.loss = tf.reduce_sum(a) + tf.reduce_sum(b)
        self.loss = self.loss - 1 * (self.gama * (tf.reduce_mean(bias) + tf.reduce_mean(pbias)))

        tf.add_to_collection('user_embedding', user_embedding)
        tf.add_to_collection('prd_embedding', prd_embedding)

        #定义最优化
        self.clip_U = tf.assign(U, tf.clip_by_norm(U, 1.0, axes=[1]))
        self.clip_P = tf.assign(P, tf.clip_by_norm(P, 1.0, axes=[1]))
        self.clip_B = tf.assign(B, tf.clip_by_value(B, 0, 1.0))
        self.clip_B1 = tf.assign(B1, tf.clip_by_value(B1, 0, 1.0))