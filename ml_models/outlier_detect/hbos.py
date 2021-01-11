import numpy as np


class HBOS(object):
    def __init__(self, bins=10, thresh=0.01):
        """
        :param bins:分箱数
        :param thresh:
        """
        self.bins = bins
        self.thresh = thresh
        self.thresh_value = None
        self.hist_bins = {}

    def fit_transform(self, X):
        # 计算直方图概率
        hist_X = np.zeros_like(X)
        for i in range(0, X.shape[1]):
            hist, bins = np.histogram(X[:, i])
            hist = hist / hist.sum()
            hist_X[:, i] = np.asarray([hist[idx - 1] for idx in np.digitize(X[:, i], bins[:-1])])
            self.hist_bins[i] = (hist, bins)
        # 计算HBOS异常值
        hbos = np.zeros_like(hist_X[:, 0])
        for i in range(0, hist_X.shape[1]):
            hbos += np.log(1.0 / (hist_X[:, i] + 1e-7))
        # 计算异常阈值
        self.thresh_value = sorted(hbos)[int(len(hbos) * (1 - self.thresh))]
        return (hbos >= self.thresh_value).astype(int)

    def transform(self, X):
        # 计算直方图概率
        hist_X = np.zeros_like(X)
        for i in range(0, X.shape[1]):
            hist, bins = self.hist_bins[i]
            hist_X[:, i] = np.asarray([hist[idx - 1] for idx in np.digitize(X[:, i], bins[:-1])])
        # 计算HBOS异常值
        hbos = np.zeros_like(hist_X[:, 0])
        for i in range(0, hist_X.shape[1]):
            hbos += np.log(1.0 / (hist_X[:, i] + 1e-7))
        return (hbos >= self.thresh_value).astype(int)
