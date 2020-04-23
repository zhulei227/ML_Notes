"""
齐次时间、离散、有限状态、一阶马尔可夫链的实现
"""
import numpy as np


class SimpleMarkovModel(object):
    def __init__(self, status_num=None):
        # 初始状态向量
        self.pi = np.zeros(shape=(status_num, 1))
        # 状态转移概率矩阵
        self.P = np.zeros(shape=(status_num, status_num))

    def fit(self, x):
        """
        根据训练数据，统计计算初始状态向量以及状态转移概率矩阵
        :param x: x可以是单列表或者是列表的列表，比如[s1,s2,...,sn]或者[[s11,s12,...,s1m],[s21,s22,...,s2n],...],
                 计算初始状态向量的方式会有差异，单列表会统计所有所有状态作为初始状态分布，列表的列表会统计所有子列表开头
                 状态的分布
        :return:
        """
        if type(x[0]) == list:
            for clist in x:
                self.pi[clist[0]] += 1
                for cindex in range(0, len(clist) - 1):
                    self.P[clist[cindex + 1], clist[cindex]] += 1
        else:
            for index in range(0, len(x) - 1):
                self.pi[x[index]] += 1
                self.P[x[index + 1], x[index]] += 1
        # 归一化
        self.pi = self.pi / np.sum(self.pi)
        self.P = self.P / np.sum(self.P, axis=0)

    def predict_log_joint_prob(self, status_list):
        """
        计算联合概率的对数
        :param status_list:
        :return:
        """
        # 这里偷懒就不并行计算了...
        log_prob = np.log(self.pi[status_list[0], 0])
        for index in range(0, len(status_list) - 1):
            log_prob += np.log(self.P[status_list[index + 1], status_list[index]])
        return log_prob

    def predict_prob_distribution(self, time_steps=None, set_init_prob=None, set_prob_trans_matrix=None):
        """
        计算time_steps后的概率分布，允许通过set_init_prob和set_prob_trans_matrix设置初始概率分布和概率转移矩阵
        :param time_steps:
        :param set_init_prob:
        :param set_prob_trans_matrix:
        :return:
        """
        prob = self.pi if set_init_prob is None else set_init_prob
        trans_matrix = self.P if set_prob_trans_matrix is None else set_prob_trans_matrix
        for _ in range(0, time_steps):
            prob = trans_matrix.dot(prob)
        return prob

    def predict_next_step_prob_distribution(self, current_status=None):
        """
        预测下一时刻的状态分布
        :param current_status:
        :return:
        """
        return self.P[:, [current_status]]

    def predict_next_step_status(self, current_status=None):
        """
        预测下一个时刻最有可能的状态
        :param current_status:
        :return:
        """
        return np.argmax(self.predict_next_step_prob_distribution(current_status))

    def generate_status(self, step_times=10, stop_status=None, set_start_status=None, search_type="greedy", beam_num=5):
        """
        生成状态序列，包括greedy search和beam search两种方式
        :param step_times: 步长不超过 step_times
        :param stop_status: 中止状态列表
        :param set_start_status: 人为设置初始状态
        :param search_type: 搜索策略，包括greedy和beam
        :param beam_num: 只有在search_type="beam"时生效，保留前top个结果
        :return:
        """
        if stop_status is None:
            stop_status = []
        # 初始状态
        start_status = np.random.choice(len(self.pi.reshape(-1)),
                                        p=self.pi.reshape(-1)) if set_start_status is None else set_start_status
        if search_type == "greedy":
            # 贪婪搜索
            rst = [start_status]
            for _ in range(0, step_times):
                next_status = self.predict_next_step_status(current_status=start_status)
                rst.append(next_status)
                if next_status in stop_status:
                    break
        else:
            # beam search
            rst = [start_status]
            top_k_rst = [[start_status]]
            top_k_prob = [0.0]
            for _ in range(0, step_times):
                new_top_k_rst = []
                new_top_k_prob = []
                for k_index, k_rst in enumerate(top_k_rst):
                    k_rst_last_status = k_rst[-1]
                    # 获取前k大的idx
                    top_k_idx = self.P[:, k_rst_last_status].argsort()[::-1][0:beam_num]
                    for top_k_status in top_k_idx:
                        new_top_k_rst.append(k_rst + [top_k_status])
                        new_top_k_prob.append(top_k_prob[k_index] + np.log(1e-12+self.P[top_k_status, k_rst_last_status]))
                # 对所有的beam_num*beam_num个结果排序取前beam_num个结果
                top_rst_idx = np.asarray(new_top_k_prob).argsort()[::-1][0:beam_num]
                rst = new_top_k_rst[top_rst_idx[0]]
                # 更新
                top_k_rst = []
                top_k_prob = []
                for top_idx in top_rst_idx[:beam_num]:
                    if new_top_k_rst[top_idx][-1] in stop_status:
                        rst = new_top_k_rst[top_idx]
                        break
                    else:
                        top_k_rst.append(new_top_k_rst[top_idx])
                        top_k_prob.append(new_top_k_prob[top_idx])
        return rst
