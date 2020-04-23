"""
PageRank算法实现
"""

import numpy as np


class PageRank(object):
    def __init__(self, init_prob, trans_matrix):
        self.init_prob = init_prob
        self.trans_matrix = trans_matrix

    def get_page_rank_values(self, time_steps=None, set_init_prob=None, set_prob_trans_matrix=None):
        """
        计算time_steps后的概率分布，允许通过set_init_prob和set_prob_trans_matrix设置初始概率分布和概率转移矩阵
        :param time_steps:
        :param set_init_prob:
        :param set_prob_trans_matrix:
        :return:
        """
        init_prob = self.init_prob if set_init_prob is None else set_init_prob
        trans_matrix = self.trans_matrix if set_prob_trans_matrix is None else set_prob_trans_matrix
        for _ in range(0, time_steps):
            init_prob = trans_matrix.dot(init_prob)
            init_prob = init_prob / np.max(np.abs(init_prob))
        return init_prob / np.sum(init_prob)
