import numpy as np


class DataBinWrapper(object):
    def __init__(self, max_bins=10):
        # 分段数
        self.max_bins = max_bins
        # 记录x各个特征的分段区间
        self.XrangeMap = None

    def fit(self, x):
        n_sample, n_feature = x.shape
        # 构建分段数据
        self.XrangeMap = [[] for _ in range(0, n_feature)]
        for index in range(0, n_feature):
            tmp = sorted(x[:, index])
            for percent in range(1, self.max_bins):
                percent_value = np.percentile(tmp, (1.0 * percent / self.max_bins) * 100.0 // 1)
                self.XrangeMap[index].append(percent_value)
            self.XrangeMap[index] = sorted(list(set(self.XrangeMap[index])))

    def transform(self, x):
        """
        抽取x_bin_index
        :param x:
        :return:
        """
        if x.ndim == 1:
            return np.asarray([np.digitize(x[i], self.XrangeMap[i]) for i in range(0, x.size)])
        else:
            return np.asarray([np.digitize(x[:, i], self.XrangeMap[i]) for i in range(0, x.shape[1])]).T
