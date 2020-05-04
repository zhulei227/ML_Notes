import numpy as np
from tqdm import tqdm

"""
线性链条件随机场的实现
"""

"""
1.实现特征函数的功能
"""


class CRFFeatureFunction(object):
    def __init__(self, unigram_rulers=None, bigram_rulers=None, output_status_num=None, input_status_num=None):
        """
        默认输入特征就一种类型
        :param unigram_rulers: 状态特征规则
        :param bigram_rulers: 状态转移规则
        :param output_status_num:输出状态数
        :param input_status_num:输入状态数
        """
        self.output_status_num = output_status_num
        self.input_status_num = input_status_num
        if unigram_rulers is None:
            self.unigram_rulers = [
                [0],  # 当前特征->标签
                [1],  # 后一个特征->标签
                [-1],  # 前一个特征->标签
                [0, 1],  # 当前特征和后一个特征->标签
                [-1, 0]  # 前一个特征和当前特征->标签
            ]
        else:
            self.unigram_rulers = unigram_rulers
        if bigram_rulers is None:
            self.bigram_rulers = [
                None,  # 不考虑当前特征，只考虑前一个标签和当前标签
                [0]  # 当前特征->前一个标签和当前标签
            ]
        else:
            self.bigram_rulers = bigram_rulers
        # 特征函数
        self.feature_funcs = None
        self.x_tol_bias = None
        self.tol_hash_weights = None
        self.max_range = None

    def fit_(self, x, y):
        """
        弃用，请使用fit
        构建特征函数，为了节省空间，训练集x,y中没有出现的特征和标签组合就不考虑了
        :param x: [[...],[...],...,[...]]
        :param y: [[...],[...],...,[...]]
        :return:
        """
        uni_cache = {}
        bi_cache = {}
        for i in range(0, len(x)):
            xi = x[i]
            yi = y[i]
            # 处理unigram_ruler
            for k, unigram_ruler in enumerate(self.unigram_rulers):
                if uni_cache.get(k) is None:
                    uni_cache[k] = []
                for j in range(max(0, 0 - np.min(unigram_ruler)), min(len(xi), len(xi) - np.max(unigram_ruler))):
                    key = "".join(str(item) for item in [xi[pos + j] for pos in unigram_ruler] + [yi[j]])
                    if key in uni_cache[k]:
                        continue
                    else:
                        self.feature_funcs.append([
                            'u',
                            unigram_ruler,
                            [xi[j + pos] for pos in unigram_ruler],
                            yi[j]
                        ])
                        uni_cache[k].append(key)
            # 处理 bigram_ruler
            for k, bigram_ruler in enumerate(self.bigram_rulers):
                if bi_cache.get(k) is None:
                    bi_cache[k] = []
                # B的情况
                if bigram_ruler is None:
                    for j in range(1, len(xi)):
                        key = "B" + "".join([str(yi[j - 1]), str(yi[j])])
                        if key in bi_cache[k]:
                            continue
                        else:
                            self.feature_funcs.append([
                                'B',
                                None,
                                None,
                                [yi[j - 1], yi[j]]
                            ])
                            bi_cache[k].append(key)
                    continue
                # 非B的情况
                for j in range(max(1, 0 - np.min(bigram_ruler)), min(len(xi), len(xi) - np.max(bigram_ruler))):
                    key = "".join(str(item) for item in [xi[pos + j] for pos in bigram_ruler] + [yi[j - 1], yi[j]])
                    if key in bi_cache[k]:
                        continue
                    else:
                        self.feature_funcs.append([
                            'b',
                            bigram_ruler,
                            [xi[j + pos] for pos in bigram_ruler],
                            [yi[j - 1], yi[j]]
                        ])
                        bi_cache[k].append(key)
        del uni_cache
        del bi_cache

    def map_(self, y_pre, y_cur, x_tol, i_cur):
        """
        弃用，请使用map
        返回是否match特征函数的list
        :param y_pre:
        :param y_cur:
        :param x_tol:
        :param i_cur:
        :return:
        """

        def map_func_(func):
            try:
                gram_type, ruler, xi, yi = func
                if gram_type == "u" and [x_tol[i + i_cur] for i in ruler] == xi and yi == y_cur:
                    return 1
                elif gram_type == "b" and [x_tol[i + i_cur] for i in ruler] == xi and yi == [y_pre, y_cur]:
                    return 1
                elif gram_type == "B" and yi == [y_pre, y_cur]:
                    return 1
                else:
                    return 0
            except:
                # 越界的情况，默认不匹配
                return 0

        return np.asarray(list(map(map_func_, self.feature_funcs)))

    def fit(self, x, y):
        """
        :param x:[[...],[...],...,[...]]
        :param y: [[...],[...],...,[...]]
        :return:
        """
        # 收集所有的x的bias项
        x_tol_bias = set()
        for ruler in self.unigram_rulers:
            x_tol_bias |= set(ruler)
        for ruler in self.bigram_rulers:
            if ruler is not None:
                x_tol_bias |= set(ruler)
        x_tol_bias = sorted(list(x_tol_bias))
        self.x_tol_bias = x_tol_bias
        x_bias_map = {}
        for index, value in enumerate(x_tol_bias):
            x_bias_map[value] = index
        """
        1.构建hash映射函数的权重
        """
        self.tol_hash_weights = []
        bias = 0
        for unigram_ruler in self.unigram_rulers:
            bias_append = 1
            weights = [0] * len(x_tol_bias)
            for index in range(0, len(unigram_ruler)):
                bias_append *= self.input_status_num
                weights[x_bias_map[unigram_ruler[index]]] = np.power(self.input_status_num, len(unigram_ruler) - index)
            weights.extend([1, 0])
            bias_append *= self.output_status_num
            weights.append(bias)
            # 前面n-1个为内积向量，最后一个为bias
            self.tol_hash_weights.append(weights)
            bias += bias_append
        for bigram_ruler in self.bigram_rulers:
            bias_append = 1
            weights = [0] * len(x_tol_bias)
            if bigram_ruler is None:
                weights = weights + [self.output_status_num, 1, bias]
                self.tol_hash_weights.append(weights)
                bias += self.output_status_num * self.output_status_num
            else:
                for index in range(0, len(bigram_ruler)):
                    bias_append *= self.input_status_num
                    weights[x_bias_map[bigram_ruler[index]]] = np.power(self.input_status_num, len(
                        bigram_ruler) - index) * self.output_status_num
                weights.extend([self.output_status_num, 1, bias])
                self.tol_hash_weights.append(weights)
                bias_append = bias_append * self.output_status_num * self.output_status_num
                bias += bias_append
        self.tol_hash_weights = np.asarray(self.tol_hash_weights)
        self.max_range = bias + 1
        """
        2.构建feature_funcs
        """
        print("构造特征函数...")
        self.feature_funcs = set()
        for i in tqdm(range(0, len(x))):
            xi = x[i]
            yi = y[i]
            tol_data = []
            # 添加x
            for pos in self.x_tol_bias:
                if pos < 0:
                    data = [self.max_range] * abs(pos) + xi[0:len(xi) + pos]
                elif pos > 0:
                    data = xi[0 + pos:len(xi)] + [self.max_range] * abs(pos)
                else:
                    data = xi
                tol_data.append(data)
            # 添加y
            tol_data.append(yi)
            tol_data.append([self.max_range] + yi[0:-1])
            # 添加bias
            tol_data.append([1] * len(tol_data[-1]))
            tol_data = np.asarray(tol_data)

            self.feature_funcs |= set(
                [item for item in tol_data.T.dot(self.tol_hash_weights.T).reshape(-1) if
                 item >= 0 and item < self.max_range])
        new_feature_func = {}
        for index, item in enumerate(self.feature_funcs):
            new_feature_func[item] = index
        self.feature_funcs = new_feature_func
        print("特征函数数量：", len(self.feature_funcs))

    def map(self, y_pre, y_cur, x_tol, i_cur):
        """
        即求f(y_{i-1},y_i,x,i)
        :param y_pre:
        :param y_cur:
        :param x_tol:
        :param i_cur:
        :return:
        """
        vec = []
        rst = np.zeros(len(self.feature_funcs))
        for pos in self.x_tol_bias:
            # 越界
            if pos + i_cur < 0 or pos + i_cur >= len(x_tol):
                vec.append(self.max_range)
            else:
                vec.append(x_tol[i_cur + pos])
        vec.extend([y_cur, y_pre, 1])
        vec = np.asarray([vec])
        tol_features = vec.dot(self.tol_hash_weights.T).reshape(-1)
        for feature in tol_features:
            feature_index = self.feature_funcs.get(feature)
            if feature_index:
                rst[feature_index] += 1
        return rst

    def map_sequence(self, y, x):
        """
        即求F(y,x)
        :param y:
        :param x:
        :return:
        """
        tol_data = []
        # 添加x
        for pos in self.x_tol_bias:
            if pos < 0:
                data = [self.max_range] * abs(pos) + x[0:len(x) + pos]
            elif pos > 0:
                data = x[0 + pos:len(x)] + [self.max_range] * abs(pos)
            else:
                data = x
            tol_data.append(data)
        # 添加y
        tol_data.append(y)
        tol_data.append([self.max_range] + y[0:-1])
        # 添加bias
        tol_data.append([1] * len(tol_data[-1]))
        tol_data = np.asarray(tol_data)
        # 与hashmap做内积
        tol_features = tol_data.T.dot(self.tol_hash_weights.T).reshape(-1)
        # 统计结果
        rst = np.zeros(len(self.feature_funcs))
        for feature in tol_features:
            feature_index = self.feature_funcs.get(feature)
            if feature_index:
                rst[feature_index] += 1
        return rst


"""
2.线性链CRF的实现
"""


class CRF(object):
    def __init__(self, epochs=10, lr=1e-3, output_status_num=None, input_status_num=None, unigram_rulers=None,
                 bigram_rulers=None):
        """
        :param epochs: 迭代次数
        :param lr: 学习率
        :param output_status_num:标签状态数
        :param input_status_num:输入状态数
        :param unigram_rulers: 状态特征规则
        :param bigram_rulers: 状态转移规则
        """
        self.epochs = epochs
        self.lr = lr
        # 为输入序列和标签状态序列添加一个头尾id
        self.output_status_num = output_status_num + 2
        self.input_status_num = input_status_num + 2
        self.input_status_head_tail = [input_status_num, input_status_num + 1]
        self.output_status_head_tail = [output_status_num, output_status_num + 1]
        # 特征函数
        self.FF = CRFFeatureFunction(unigram_rulers, bigram_rulers, input_status_num=self.input_status_num,
                                     output_status_num=self.output_status_num)
        # 模型参数
        self.w = None

    def fit(self, x, y, if_drop_p=True):
        """
        :param x: [[...],[...],...,[...]]
        :param y: [[...],[...],...,[...]]
        :param if_drop_p:是否去掉P_w(x)项
        :return
        """
        # 为 x,y加头尾
        x = [[self.input_status_head_tail[0]] + xi + [self.input_status_head_tail[1]] for xi in x]
        y = [[self.output_status_head_tail[0]] + yi + [self.output_status_head_tail[1]] for yi in y]
        self.FF.fit(x, y)
        self.w = np.ones(len(self.FF.feature_funcs)) * 1e-5
        if if_drop_p:
            print("训练进度...")
            for i in tqdm(range(0, len(x))):
                xi = x[i]
                yi = y[i]
                self.w = self.w + self.epochs * self.lr * self.FF.map_sequence(yi, xi)
        else:
            for epoch in range(0, self.epochs):
                # 偷个懒，用随机梯度下降
                print("\n 模型训练,epochs:" + str(epoch + 1) + "/" + str(self.epochs))
                for i in tqdm(range(0, len(x))):
                    xi = x[i]
                    yi = y[i]
                    """
                    1.求F(yi \mid xi)以及P_w(yi \mid xi)
                    """
                    F_y_x = self.FF.map_sequence(yi, xi)
                    Z_x = np.ones(shape=(self.output_status_num, 1)).T
                    for j in range(1, len(xi)):
                        # 构建M矩阵
                        M = np.zeros(shape=(self.output_status_num, self.output_status_num))
                        for k in range(0, self.output_status_num):
                            for t in range(0, self.output_status_num):
                                M[k, t] = np.exp(np.dot(self.w, self.FF.map(k, t, xi, j)))
                        # 前向算法求 Z(x)
                        Z_x = Z_x.dot(M)
                    Z_x = np.sum(Z_x)
                    P_w = np.exp(np.dot(self.w, F_y_x)) / Z_x
                    """
                    2.求梯度,并更新
                    """
                    dw = (P_w - 1) * F_y_x
                    self.w = self.w - self.lr * dw

    def predict(self, x):
        """
        维特比求解最优的y
        :param x:[...]
        :return:
        """
        # 为x加头尾
        x = [self.input_status_head_tail[0]] + x + [self.input_status_head_tail[1]]
        # 初始化
        delta = np.asarray([np.dot(self.w, self.FF.map(self.output_status_head_tail[0], j, x, 1)) for j in
                            range(0, self.output_status_num)])
        psi = [[0] * self.output_status_num]
        # 递推
        for visible_index in range(2, len(x) - 1):
            new_delta = np.zeros_like(delta)
            new_psi = []
            # 当前节点
            for i in range(0, self.output_status_num):
                best_pre_index_i = -1
                best_pre_index_value_i = 0
                delta_i = 0
                # 上一轮节点
                for j in range(0, self.output_status_num):
                    delta_i_j = delta[j] + np.dot(self.w, self.FF.map(j, i, x, visible_index))
                    if delta_i_j > delta_i:
                        delta_i = delta_i_j
                    best_pre_index_value_i_j = delta[j] + np.dot(self.w, self.FF.map(j, i, x, visible_index))
                    if best_pre_index_value_i_j > best_pre_index_value_i:
                        best_pre_index_value_i = best_pre_index_value_i_j
                        best_pre_index_i = j
                new_delta[i] = delta_i
                new_psi.append(best_pre_index_i)
            delta = new_delta
            psi.append(new_psi)
        # 回溯
        best_hidden_status = [np.argmax(delta)]
        for psi_index in range(len(x) - 3, 0, -1):
            next_status = psi[psi_index][best_hidden_status[-1]]
            best_hidden_status.append(next_status)
        best_hidden_status.reverse()
        return best_hidden_status
