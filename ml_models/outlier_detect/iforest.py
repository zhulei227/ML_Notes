import numpy as np


# 函数C(n)
def _C(n):
    return 2 * np.log(n) - 2 / n - 0.8455686702


# 孤立节点
class _INode(object):
    def __init__(self, idx=None, threshold=None, size=None, leftNode=None, rightNode=None, deep=None):
        """
        :param idx: 维度
        :param threshold: 阈值
        :param size: 样本量
        :param leftNode: 左节点
        :param rightNode: 右节点
        :param deep:深度
        """
        self.idx = idx
        self.threshold = threshold
        self.size = size
        self.leftNode = leftNode
        self.rightNode = rightNode
        self.deep = deep


# 构建孤立树
class _ITree(object):
    def __init__(self, max_deep=10):
        """
        :param max_deep: 树的最大深度
        """
        self.max_deep = max_deep
        self.rootNode = None

    # 递归构建
    def _build(self, node: _INode, data, current_deep):
        rows, cols = data.shape
        node.deep = current_deep
        # 判断中止条件(1):达到最大深度或者样本量为1
        if current_deep == self.max_deep or rows == 1:
            node.size = 1
            return
        # 判断中止条件(2):所有样本取值都一样
        available_cols = []
        for col in range(0, cols):
            if np.std(data[:, col]) > 1e-7:
                available_cols.append(col)
        if len(available_cols) == 0:
            node.size = rows
            return
        node.size = rows
        # 随机选择一个维度
        idx = np.random.choice(available_cols, 1)[0]
        node.idx = idx
        # 随机构建一个阈值
        max_value = np.max(data[:, idx])
        min_value = np.min(data[:, idx])
        threshold = np.random.random() * (max_value - min_value) + min_value
        node.threshold = threshold
        # 切分数据
        left_data = data[(data[:, idx] < threshold).flatten()]
        right_data = data[(data[:, idx] >= threshold).flatten()]
        node.leftNode = _INode()
        node.rightNode = _INode()
        self._build(node.leftNode, left_data, current_deep + 1)
        self._build(node.rightNode, right_data, current_deep + 1)

    def fit(self, X):
        self.rootNode = _INode()
        self._build(self.rootNode, X, 0)

    # 递归查询
    def _search(self, node: _INode, x):
        if node.leftNode is None or node.rightNode is None:
            return node.deep + _C(node.size)
        else:
            if x[node.idx] < node.threshold:
                return self._search(node.leftNode, x)
            else:
                return self._search(node.rightNode, x)

    def predict_score(self, x):
        # 估计分值h(x)=e+C(T.size)
        return self._search(self.rootNode, x)


# 构建孤立森林
class IForest(object):
    def __init__(self, sub_sample=0.66, n_estimators=100):
        """
        :param sub_sample: 采样比例
        :param n_estimators: 孤立树数量
        """
        self.sub_sample = sub_sample
        self.n_estimators = n_estimators
        self.trees = []
        self.c_phi = None  # 即归一化因子

    def fit(self, X):
        rows, cols = X.shape
        trn_num = int(rows * self.sub_sample)
        self.c_phi = _C(trn_num)
        max_deep = int(np.log2(trn_num)) + 1
        indices = list(range(0, rows))
        for _ in range(0, self.n_estimators):
            tree = _ITree(max_deep=max_deep)
            np.random.shuffle(indices)
            tree.fit(X[indices[:trn_num]])
            self.trees.append(tree)

    def predict(self, X):
        rst = []
        rows, _ = X.shape
        for row in range(0, rows):
            x = X[row]
            hs = []
            for tree in self.trees:
                hs.append(tree.predict_score(x))
            rst.append(np.exp2(-1 * np.mean(hs) / self.c_phi))
        return np.asarray(rst)
