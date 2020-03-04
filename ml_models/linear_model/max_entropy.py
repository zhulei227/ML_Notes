"""
最大熵模型
"""
import numpy as np
from .. import utils


class SimpleFeatureFunction(object):
    def __init__(self):
        """
        记录特征函数
        {
            (x_index,x_value,y_index)
        }
        """
        self.feature_funcs = set()

    def build_feature_funcs(self, X, y):
        """
        构建特征函数
        :param X:
        :param y:
        :return:
        """
        n_sample, _ = X.shape
        for index in range(0, n_sample):
            x = X[index, :].tolist()
            for feature_index in range(0, len(x)):
                self.feature_funcs.add(tuple([feature_index, x[feature_index], y[index]]))

    def get_feature_funcs_num(self):
        """
        获取特征函数总数
        :return:
        """
        return len(self.feature_funcs)

    def match_feature_funcs_indices(self, x, y):
        """
        返回命中的特征函数index
        :param x:
        :param y:
        :return:
        """
        match_indices = []
        index = 0
        for feature_index, feature_value, feature_y in self.feature_funcs:
            if feature_y == y and x[feature_index] == feature_value:
                match_indices.append(index)
            index += 1
        return match_indices


class MaxEnt(object):
    def __init__(self, feature_func, epochs=5, eta=0.01):
        self.feature_func = feature_func
        self.epochs = epochs
        self.eta = eta

        self.class_num = None
        """
        记录联合概率分布:
        {
            (x_0,x_1,...,x_p,y_index):p
        }
        """
        self.Pxy = {}
        """
        记录边缘概率分布:
        {
            (x_0,x_1,...,x_p):p
        }
        """
        self.Px = {}

        """
        w[i]-->feature_func[i]
        """
        self.w = None

    def init_params(self, X, y):
        """
        初始化相应的数据
        :return:
        """
        n_sample, n_feature = X.shape
        self.class_num = np.max(y) + 1

        # 初始化联合概率分布、边缘概率分布、特征函数
        for index in range(0, n_sample):
            range_indices = X[index, :].tolist()

            if self.Px.get(tuple(range_indices)) is None:
                self.Px[tuple(range_indices)] = 1
            else:
                self.Px[tuple(range_indices)] += 1

            if self.Pxy.get(tuple(range_indices + [y[index]])) is None:
                self.Pxy[tuple(range_indices + [y[index]])] = 1
            else:
                self.Pxy[tuple(range_indices + [y[index]])] = 1

        for key, value in self.Pxy.items():
            self.Pxy[key] = 1.0 * self.Pxy[key] / n_sample
        for key, value in self.Px.items():
            self.Px[key] = 1.0 * self.Px[key] / n_sample

        # 初始化参数权重
        self.w = np.zeros(self.feature_func.get_feature_funcs_num())

    def _sum_exp_w_on_all_y(self, x):
        """
        sum_y exp(self._sum_w_on_feature_funcs(x))
        :param x:
        :return:
        """
        sum_w = 0
        for y in range(0, self.class_num):
            tmp_w = self._sum_exp_w_on_y(x, y)
            sum_w += np.exp(tmp_w)
        return sum_w

    def _sum_exp_w_on_y(self, x, y):
        tmp_w = 0
        match_feature_func_indices = self.feature_func.match_feature_funcs_indices(x, y)
        for match_feature_func_index in match_feature_func_indices:
            tmp_w += self.w[match_feature_func_index]
        return tmp_w

    def fit(self, X, y, sample_weight=None):
        n_sample = X.shape[0]
        if sample_weight is None:
            sample_weight = np.asarray([1.0] * n_sample)
        # check sample_weight
        if len(sample_weight) != n_sample:
            raise Exception('sample_weight size error:', len(sample_weight))
        self.eta = max(1.0 / np.sqrt(X.shape[0]), self.eta)
        self.init_params(X, y)
        x_y = np.c_[X, y]
        for epoch in range(self.epochs):
            count = 0
            np.random.shuffle(x_y)
            for index in range(x_y.shape[0]):
                count += 1
                x_point = x_y[index, :-1]
                y_point = x_y[index, -1:][0]
                # 获取联合概率分布
                p_xy = self.Pxy.get(tuple(x_point.tolist() + [y_point]))
                # 获取边缘概率分布
                p_x = self.Px.get(tuple(x_point))
                # 更新w
                dw = np.zeros(shape=self.w.shape)
                match_feature_func_indices = self.feature_func.match_feature_funcs_indices(x_point, y_point)
                if len(match_feature_func_indices) == 0:
                    continue
                if p_xy is not None:
                    for match_feature_func_index in match_feature_func_indices:
                        dw[match_feature_func_index] = p_xy
                if p_x is not None:
                    sum_w = self._sum_exp_w_on_all_y(x_point)
                    for match_feature_func_index in match_feature_func_indices:
                        dw[match_feature_func_index] -= (
                                                            p_x * np.exp(
                                                                self._sum_exp_w_on_y(x_point, y_point)) + 1e-7) / (
                                                            1e-7 + sum_w)
                # 考虑sample_weight
                dw = dw * sample_weight[index]
                # 更新
                self.w += self.eta * dw
                # 打印训练进度
                if count % (X.shape[0] // 4) == 0:
                    print("processing:\tepoch:" + str(epoch + 1) + "/" + str(self.epochs) + ",percent:" + str(
                        count) + "/" + str(X.shape[0]))

    def predict_proba(self, x):
        """
        预测为y的概率分布
        :param x:
        :return:
        """
        y = []
        for x_point in x:
            y_tmp = []
            for y_index in range(0, self.class_num):
                match_feature_func_indices = self.feature_func.match_feature_funcs_indices(x_point, y_index)
                tmp = 0
                for match_feature_func_index in match_feature_func_indices:
                    tmp += self.w[match_feature_func_index]
                y_tmp.append(tmp)
            y.append(y_tmp)
        return utils.softmax(np.asarray(y))

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)
