"""
DART提升分类树的实现（ps,其实也不一定非要决策树，所以代码实现上还是支持异质的情况）
"""
from ml_models.tree import CARTRegressor
from ml_models import utils
import copy
import numpy as np


class DARTClassifier(object):
    def __init__(self, base_estimator=None, n_estimators=10, dropout=0.5):
        """
        :param base_estimator: 基学习器，允许异质；异质的情况下使用列表传入比如[estimator1,estimator2,...,estimator10],这时n_estimators会失效；
                                同质的情况，单个estimator会被copy成n_estimators份
        :param n_estimators: 基学习器迭代数量
        :param dropout: dropout概率
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.dropout = dropout
        if self.base_estimator is None:
            # 默认使用决策树桩
            self.base_estimator = CARTRegressor(max_depth=2)
        # 同质分类器
        if type(base_estimator) != list:
            estimator = self.base_estimator
            self.base_estimator = [copy.deepcopy(estimator) for _ in range(0, self.n_estimators)]
        # 异质分类器
        else:
            self.n_estimators = len(self.base_estimator)

        # 扩展class_num组分类器
        self.expand_base_estimators = []

        # 记录权重
        self.weights = None

    def _dropout(self, y_pred_score_):
        y_pred_score_results = []
        ks = []
        for class_index in range(0, self.class_num):
            dropout_indices = []
            no_dropout_indices = []
            for index in range(0, len(y_pred_score_[class_index])):
                if np.random.random() <= self.dropout:
                    dropout_indices.append(index)
                else:
                    no_dropout_indices.append(index)
            if len(dropout_indices) == 0:
                np.random.shuffle(no_dropout_indices)
                dropout_indices.append(no_dropout_indices.pop())
            k = len(dropout_indices)
            # 调整对应的weights
            for index in dropout_indices:
                self.weights[class_index][index] *= (1.0 * k / (k + 1))
            # 返回新的pred结果以及dropout掉的数量
            y_pred_result = np.zeros_like(y_pred_score_[class_index][0])
            for no_dropout_index in no_dropout_indices:
                y_pred_result += y_pred_score_[class_index][no_dropout_index] * self.weights[class_index][
                    no_dropout_index]
            y_pred_score_results.append(y_pred_result)
            ks.append(k)
        return y_pred_score_results, ks

    def fit(self, x, y):
        # 将y转one-hot编码
        class_num = np.amax(y) + 1
        self.class_num = class_num
        y_cate = np.zeros(shape=(len(y), class_num))
        y_cate[np.arange(len(y)), y] = 1

        self.weights = [[] for _ in range(0, class_num)]

        # 扩展分类器
        self.expand_base_estimators = [copy.deepcopy(self.base_estimator) for _ in range(class_num)]

        # 拟合第一个模型
        y_pred_score_ = [[] for _ in range(0, self.class_num)]
        # TODO:并行优化
        for class_index in range(0, class_num):
            self.expand_base_estimators[class_index][0].fit(x, y_cate[:, class_index])
            y_pred_score_[class_index].append(self.expand_base_estimators[class_index][0].predict(x))
            self.weights[class_index].append(1.0)
        y_pred_result, ks = self._dropout(y_pred_score_)
        y_pred_result = np.c_[y_pred_result].T
        # 计算负梯度
        new_y = y_cate - utils.softmax(y_pred_result)
        # 训练后续模型
        for index in range(1, self.n_estimators):
            for class_index in range(0, class_num):
                self.expand_base_estimators[class_index][index].fit(x, new_y[:, class_index])
                y_pred_score_[class_index].append(self.expand_base_estimators[class_index][index].predict(x))
                self.weights[class_index].append(1.0 / (ks[class_index] + 1))
            y_pred_result, ks = self._dropout(y_pred_score_)
            y_pred_result = np.c_[y_pred_result].T
            new_y = y_cate - utils.softmax(y_pred_result)

    def predict_proba(self, x):
        # TODO:并行优化
        y_pred_score = []
        for class_index in range(0, len(self.expand_base_estimators)):
            estimator_of_index = self.expand_base_estimators[class_index]
            y_pred_score.append(
                np.sum(
                    [estimator_of_index[0].predict(x)* self.weights[class_index][0]] +
                    [self.weights[class_index][i] * estimator_of_index[i].predict(x) for i in
                     range(1, self.n_estimators - 1)] +
                    [estimator_of_index[self.n_estimators - 1].predict(x) * self.weights[class_index][-1]]
                    , axis=0)
            )
        return utils.softmax(np.c_[y_pred_score].T)

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)
