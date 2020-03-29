"""
xgboost基模型：回归树的实现
"""
import numpy as np
from ..wrapper_models import DataBinWrapper


class XGBoostBaseTree(object):
    class Node(object):
        """
        树节点，用于存储节点信息以及关联子节点
        """

        def __init__(self, feature_index: int = None, feature_value=None, y_hat=None, score=None,
                     left_child_node=None, right_child_node=None, num_sample: int = None):
            """
            :param feature_index: 特征id
            :param feature_value: 特征取值
            :param y_hat: 预测值
            :param score: 损失函数值
            :param left_child_node: 左孩子结点
            :param right_child_node: 右孩子结点
            :param num_sample:样本量
            """
            self.feature_index = feature_index
            self.feature_value = feature_value
            self.y_hat = y_hat
            self.score = score
            self.left_child_node = left_child_node
            self.right_child_node = right_child_node
            self.num_sample = num_sample

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, gamma=1e-2, lamb=1e-1,
                 max_bins=10):
        """
        :param max_depth:树的最大深度
        :param min_samples_split:当对一个内部结点划分时，要求该结点上的最小样本数，默认为2
        :param min_samples_leaf:设置叶子结点上的最小样本数，默认为1
        :param gamma:即损失函数中的gamma
        :param lamb:即损失函数中lambda
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.gamma = gamma
        self.lamb = lamb

        self.root_node: self.Node = None
        self.dbw = DataBinWrapper(max_bins=max_bins)

    def _score(self, g, h):
        """
        计算损失损失评分
        :param g:一阶导数
        :param h: 二阶导数
        :return:
        """
        G = np.sum(g)
        H = np.sum(h)
        return -0.5 * G ** 2 / (H + self.lamb) + self.gamma

    def _build_tree(self, current_depth, current_node: Node, x, g, h):
        """
        递归进行特征选择，构建树
        :param x:
        :param y:
        :param sample_weight:
        :return:
        """
        rows, cols = x.shape
        # 计算G和H
        G = np.sum(g)
        H = np.sum(h)
        # 计算当前的预测值
        current_node.y_hat = -1 * G / (H + self.lamb)
        current_node.num_sample = rows
        # 判断停止切分的条件
        current_node.score = self._score(g, h)

        if rows < self.min_samples_split:
            return

        if self.max_depth is not None and current_depth > self.max_depth:
            return

        # 寻找最佳的特征以及取值
        best_index = None
        best_index_value = None
        best_criterion_value = 0
        for index in range(0, cols):
            for index_value in sorted(set(x[:, index])):
                left_indices = np.where(x[:, index] <= index_value)
                right_indices = np.where(x[:, index] > index_value)
                criterion_value = current_node.score - self._score(g[left_indices], h[left_indices]) - self._score(
                    g[right_indices], h[right_indices])
                if criterion_value > best_criterion_value:
                    best_criterion_value = criterion_value
                    best_index = index
                    best_index_value = index_value

        # 如果减少不够则停止
        if best_index is None:
            return
        # 切分
        current_node.feature_index = best_index
        current_node.feature_value = best_index_value
        selected_x = x[:, best_index]

        # 创建左孩子结点
        left_selected_index = np.where(selected_x <= best_index_value)
        # 如果切分后的点太少，以至于都不能做叶子节点，则停止分割
        if len(left_selected_index[0]) >= self.min_samples_leaf:
            left_child_node = self.Node()
            current_node.left_child_node = left_child_node
            self._build_tree(current_depth + 1, left_child_node, x[left_selected_index], g[left_selected_index],
                             h[left_selected_index])
        # 创建右孩子结点
        right_selected_index = np.where(selected_x > best_index_value)
        # 如果切分后的点太少，以至于都不能做叶子节点，则停止分割
        if len(right_selected_index[0]) >= self.min_samples_leaf:
            right_child_node = self.Node()
            current_node.right_child_node = right_child_node
            self._build_tree(current_depth + 1, right_child_node, x[right_selected_index], g[right_selected_index],
                             h[right_selected_index])

    def fit(self, x, g, h):
        # 构建空的根节点
        self.root_node = self.Node()

        # 对x分箱
        self.dbw.fit(x)

        # 递归构建树
        self._build_tree(1, self.root_node, self.dbw.transform(x), g, h)

    # 检索叶子节点的结果
    def _search_node(self, current_node: Node, x):
        if current_node.left_child_node is not None and x[current_node.feature_index] <= current_node.feature_value:
            return self._search_node(current_node.left_child_node, x)
        elif current_node.right_child_node is not None and x[current_node.feature_index] > current_node.feature_value:
            return self._search_node(current_node.right_child_node, x)
        else:
            return current_node.y_hat

    def predict(self, x):
        # 计算结果
        x = self.dbw.transform(x)
        rows = x.shape[0]
        results = []
        for row in range(0, rows):
            results.append(self._search_node(self.root_node, x[row]))
        return np.asarray(results)
