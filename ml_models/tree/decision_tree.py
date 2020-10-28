"""
ID3和C4.5决策树分类器的实现
"""
import numpy as np
from .. import utils
from ..wrapper_models import DataBinWrapper


class DecisionTreeClassifier(object):
    class Node(object):
        """
        树节点，用于存储节点信息以及关联子节点
        """

        def __init__(self, feature_index: int = None, target_distribute: dict = None, weight_distribute: dict = None,
                     children_nodes: dict = None, num_sample: int = None):
            """
            :param feature_index: 特征id
            :param target_distribute: 目标分布
            :param weight_distribute:权重分布
            :param children_nodes: 孩子节点
            :param num_sample:样本量
            """
            self.feature_index = feature_index
            self.target_distribute = target_distribute
            self.weight_distribute = weight_distribute
            self.children_nodes = children_nodes
            self.num_sample = num_sample

    def __init__(self, criterion='c4.5', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0, max_bins=10):
        """
        :param criterion:划分标准，包括id3,c4.5，默认为c4.5
        :param max_depth:树的最大深度
        :param min_samples_split:当对一个内部结点划分时，要求该结点上的最小样本数，默认为2
        :param min_samples_leaf:设置叶子结点上的最小样本数，默认为1
        :param min_impurity_decrease:打算划分一个内部结点时，只有当划分后不纯度(可以用criterion参数指定的度量来描述)减少值不小于该参数指定的值，才会对该结点进行划分，默认值为0
        """
        self.criterion = criterion
        if criterion == 'c4.5':
            self.criterion_func = utils.info_gain_rate
        else:
            self.criterion_func = utils.muti_info
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

        self.root_node: self.Node = None
        self.dbw = DataBinWrapper(max_bins=max_bins)

    def _build_tree(self, current_depth, current_node: Node, x, y, sample_weight):
        """
        递归进行特征选择，构建树
        :param x:
        :param y:
        :param sample_weight:
        :return:
        """
        rows, cols = x.shape
        # 计算y分布以及其权重分布
        target_distribute = {}
        weight_distribute = {}
        for index, tmp_value in enumerate(y):
            if tmp_value not in target_distribute:
                target_distribute[tmp_value] = 0.0
                weight_distribute[tmp_value] = []
            target_distribute[tmp_value] += 1.0
            weight_distribute[tmp_value].append(sample_weight[index])
        for key, value in target_distribute.items():
            target_distribute[key] = value / rows
            weight_distribute[key] = np.mean(weight_distribute[key])
        current_node.target_distribute = target_distribute
        current_node.weight_distribute = weight_distribute
        current_node.num_sample = rows
        # 判断停止切分的条件

        if len(target_distribute) <= 1:
            return

        if rows < self.min_samples_split:
            return

        if self.max_depth is not None and current_depth > self.max_depth:
            return

        # 寻找最佳的特征
        best_index = None
        best_criterion_value = 0
        for index in range(0, cols):
            criterion_value = self.criterion_func(x[:, index], y)
            if criterion_value > best_criterion_value:
                best_criterion_value = criterion_value
                best_index = index

        # 如果criterion_value减少不够则停止
        if best_index is None:
            return
        if best_criterion_value <= self.min_impurity_decrease:
            return
        # 切分
        current_node.feature_index = best_index
        children_nodes = {}
        current_node.children_nodes = children_nodes
        selected_x = x[:, best_index]
        for item in set(selected_x):
            selected_index = np.where(selected_x == item)
            # 如果切分后的点太少，以至于都不能做叶子节点，则停止分割
            if len(selected_index[0]) < self.min_samples_leaf:
                continue
            child_node = self.Node()
            children_nodes[item] = child_node
            self._build_tree(current_depth + 1, child_node, x[selected_index], y[selected_index],
                             sample_weight[selected_index])

    def fit(self, x, y, sample_weight=None):
        # check sample_weight
        n_sample = x.shape[0]
        if sample_weight is None:
            sample_weight = np.asarray([1.0] * n_sample)
        # check sample_weight
        if len(sample_weight) != n_sample:
            raise Exception('sample_weight size error:', len(sample_weight))

        # 构建空的根节点
        self.root_node = self.Node()

        # 对x分箱
        self.dbw.fit(x)

        # 递归构建树
        self._build_tree(1, self.root_node, self.dbw.transform(x), y, sample_weight)

    # 检索叶子节点的结果
    def _search_node(self, current_node: Node, x, class_num):
        if current_node.feature_index is None or current_node.children_nodes is None or len(
                current_node.children_nodes) == 0 or current_node.children_nodes.get(
            x[current_node.feature_index]) is None:
            result = []
            total_value = 0.0
            for index in range(0, class_num):
                value = current_node.target_distribute.get(index, 0) * current_node.weight_distribute.get(index, 1.0)
                result.append(value)
                total_value += value
            # 归一化
            for index in range(0, class_num):
                result[index] = result[index] / total_value
            return result
        else:
            return self._search_node(current_node.children_nodes.get(x[current_node.feature_index]), x, class_num)

    def predict_proba(self, x):
        # 计算结果概率分布
        x = self.dbw.transform(x)
        rows = x.shape[0]
        results = []
        class_num = len(self.root_node.target_distribute)
        for row in range(0, rows):
            results.append(self._search_node(self.root_node, x[row], class_num))
        return np.asarray(results)

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)

    def _prune_node(self, current_node: Node, alpha):
        # 如果有子结点,先对子结点部分剪枝
        if current_node.children_nodes is not None and len(current_node.children_nodes) != 0:
            for child_node in current_node.children_nodes.values():
                self._prune_node(child_node, alpha)

        # 再尝试对当前结点剪枝
        if current_node.children_nodes is not None and len(current_node.children_nodes) != 0:
            # 避免跳层剪枝
            for child_node in current_node.children_nodes.values():
                # 当前剪枝的层必须是叶子结点的层
                if child_node.children_nodes is not None and len(child_node.children_nodes) > 0:
                    return
            # 计算剪枝的前的损失值
            pre_prune_value = alpha * len(current_node.children_nodes)
            for child_node in current_node.children_nodes.values():
                for key, value in child_node.target_distribute.items():
                    pre_prune_value += -1 * child_node.num_sample * value * np.log(
                        value) * child_node.weight_distribute.get(key, 1.0)
            # 计算剪枝后的损失值
            after_prune_value = alpha
            for key, value in current_node.target_distribute.items():
                after_prune_value += -1 * current_node.num_sample * value * np.log(
                    value) * current_node.weight_distribute.get(key, 1.0)

            if after_prune_value <= pre_prune_value:
                # 剪枝操作
                current_node.children_nodes = None
                current_node.feature_index = None

    def prune(self, alpha=0.01):
        """
        决策树剪枝 C(T)+alpha*|T|
        :param alpha:
        :return:
        """
        # 递归剪枝
        self._prune_node(self.root_node, alpha)
