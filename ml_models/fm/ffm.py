"""
FFM因子分解机的实现
"""
import numpy as np
from ml_models import utils


class FFM(object):
    def __init__(self, epochs=1, lr=1e-3, adjust_lr=True, batch_size=1, hidden_dim=4, lamb=1e-3, alpha=1e-3,
                 normal=True, solver='adam', rho_1=0.9, rho_2=0.999, early_stopping_rounds=100,
                 objective="squarederror", tweedie_p=1.5):
        """

        :param epochs: 迭代轮数
        :param lr: 学习率
        :param adjust_lr:是否根据特征数量再次调整学习率 max(lr,1/n_feature)
        :param batch_size:
        :param hidden_dim:隐变量维度
        :param lamb:l2正则项系数
        :param alpha:l1正则项系数
        :param normal:是否归一化，默认用min-max归一化
        :param solver:优化方式，包括sgd,adam,默认adam
        :param rho_1:adam的rho_1的权重衰减,solver=adam时生效
        :param rho_2:adam的rho_2的权重衰减,solver=adam时生效
        :param early_stopping_rounds:对early_stopping进行支持，默认100，使用rmse作为回归任务评估指标，使用错误率（1-accuray）作为分类任务的评估指标
        :param objective:损失函数，回归任务支持squarederror,poisson,gamma,tweedie，分类任务支持logistic
        :param tweedie_p:teweedie的超参数，objective=tweedie时生效
        """
        self.epochs = epochs
        self.lr = lr
        self.adjust_lr = adjust_lr
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.lamb = lamb
        self.alpha = alpha
        self.solver = solver
        self.rho_1 = rho_1
        self.rho_2 = rho_2
        self.early_stopping_rounds = early_stopping_rounds
        self.objective = objective
        self.tweedie_p = tweedie_p
        # 初始化参数
        self.w = None  # w_0,w_i
        self.V = None  # v_{i,f}
        # 归一化
        self.normal = normal
        if normal:
            self.xmin = None
            self.xmax = None
        # 功能性参数
        self.replace_ind = None  # 置换index
        self.positive_ind = None  # 参与特征组合的开始id
        self.fields = []  # replace_ind后的fields
        self.field_num = None

    def _y(self, X):
        """
        实现y(x)的功能
        :param X:
        :return:
        """
        # 去掉第一列bias以及非组合特征
        X_ = X[:, self.positive_ind + 1:]
        n_sample, n_feature = X_.shape
        pol = np.zeros(n_sample)
        for i in range(0, n_feature - 1):
            for j in range(i + 1, n_feature):
                pol += X_[:, i] * X_[:, j] * np.dot(self.V[i, self.fields[self.positive_ind + j]],
                                                    self.V[j, self.fields[self.positive_ind + i]])
        linear_rst = X @ self.w.reshape(-1) + pol
        return linear_rst

    def fit(self, X, y, eval_set=None, show_log=False, fields=None):
        """
        :param X:
        :param y:
        :param eval_set:
        :param show_log:
        :param fields: 为None时，退化为FM
        :return:
        """
        X_o = X.copy()

        # 归一化
        if self.normal:
            self.xmin = X.min(axis=0)
            self.xmax = X.max(axis=0) + 1e-7
            X = (X - self.xmin) / self.xmax

        n_sample, n_feature = X.shape
        # 处理fields
        if fields is None:
            self.replace_ind = list(range(0, n_feature))
            self.positive_ind = 0
            self.fields = [0] * n_feature
            self.field_num = 1
        else:
            self.replace_ind = np.argsort(fields).tolist()
            self.positive_ind = np.sum([1 if item < 0 else 0 for item in fields])
            self.fields = sorted(fields)
            self.field_num = len(set(self.fields[self.positive_ind:]))

        # reshape X
        X = X[:, self.replace_ind]

        x_y = np.c_[np.ones(n_sample), X, y]
        # 记录loss
        train_losses = []
        eval_losses = []
        # 调整一下学习率
        if self.adjust_lr:
            self.lr = max(self.lr, 1 / n_feature)
        # 初始化参数
        self.w = np.random.random((n_feature + 1, 1)) * 1e-3
        self.V = np.random.random((n_feature - self.positive_ind, self.field_num, self.hidden_dim)) * 1e-3
        if self.solver == 'adam':
            # 缓存梯度一阶，二阶估计
            w_1 = np.zeros_like(self.w)
            V_1 = np.zeros_like(self.V)
            w_2 = np.zeros_like(self.w)
            V_2 = np.zeros_like(self.V)
        # 更新参数
        count = 0
        for epoch in range(self.epochs):
            # 验证集记录
            best_eval_value = np.power(2., 1023)
            eval_count = 0
            np.random.shuffle(x_y)
            for index in range(x_y.shape[0] // self.batch_size):
                count += 1
                batch_x_y = x_y[self.batch_size * index:self.batch_size * (index + 1)]
                batch_x = batch_x_y[:, :-1]
                batch_y = batch_x_y[:, -1:]
                # 计算链式求导第一层梯度
                if self.objective == "squarederror":
                    y_x_t = self._y(batch_x).reshape((-1, 1)) - batch_y
                elif self.objective == "poisson":
                    y_x_t = np.exp(self._y(batch_x).reshape((-1, 1))) - batch_y
                elif self.objective == "gamma":
                    y_x_t = 1.0 - batch_y * np.exp(-1.0 * self._y(batch_x).reshape((-1, 1)))
                elif self.objective == 'tweedie':
                    if self.tweedie_p == 1:
                        y_x_t = np.exp(self._y(batch_x).reshape((-1, 1))) - batch_y
                    elif self.tweedie_p == 2:
                        y_x_t = 1.0 - batch_y * np.exp(-1.0 * self._y(batch_x).reshape((-1, 1)))
                    else:
                        y_x_t = np.exp(self._y(batch_x).reshape((-1, 1)) * (2.0 - self.tweedie_p)) \
                                - batch_y * np.exp(self._y(batch_x).reshape((-1, 1)) * (1.0 - self.tweedie_p))
                else:
                    # 二分类
                    y_x_t = utils.sigmoid(self._y(batch_x).reshape((-1, 1))) - batch_y

                # 更新w
                if self.solver == 'sgd':
                    self.w = self.w - (self.lr * (np.sum(y_x_t * batch_x, axis=0) / self.batch_size).reshape(
                        (-1, 1)) + self.lamb * self.w + self.alpha * np.where(self.w > 0, 1, 0))
                elif self.solver == 'adam':
                    w_reg = self.lamb * self.w + self.alpha * np.where(self.w > 0, 1, 0)
                    w_grad = (np.sum(y_x_t * batch_x, axis=0) / self.batch_size).reshape(
                        (-1, 1)) + w_reg
                    w_1 = self.rho_1 * w_1 + (1 - self.rho_1) * w_grad
                    w_2 = self.rho_2 * w_2 + (1 - self.rho_2) * w_grad * w_grad
                    w_1_ = w_1 / (1 - np.power(self.rho_1, count))
                    w_2_ = w_2 / (1 - np.power(self.rho_2, count))
                    self.w = self.w - (self.lr * w_1_) / (np.sqrt(w_2_) + 1e-8)

                # 更新 V
                batch_x_ = batch_x[:, 1 + self.positive_ind:]
                # 逐元素更新
                for i in range(0, batch_x_.shape[1] - 1):
                    for j in range(i + 1, batch_x_.shape[1]):
                        for k in range(0, self.hidden_dim):
                            v_reg_l = self.lamb * self.V[i, self.fields[self.positive_ind + j], k] + \
                                      self.alpha * (self.V[i, self.fields[self.positive_ind + j], k] > 0)

                            v_grad_l = np.sum(y_x_t.reshape(-1) * batch_x_[:, i] * batch_x_[:, j] *
                                              self.V[
                                                  j, self.fields[self.positive_ind + i], k]) / self.batch_size + v_reg_l

                            v_reg_r = self.lamb * self.V[j, self.fields[self.positive_ind + i], k] + \
                                      self.alpha * (self.V[j, self.fields[self.positive_ind + i], k] > 0)

                            v_grad_r = np.sum(y_x_t.reshape(-1) * batch_x_[:, i] * batch_x_[:, j] *
                                              self.V[
                                                  i, self.fields[self.positive_ind + j], k]) / self.batch_size + v_reg_r

                            if self.solver == "sgd":
                                self.V[i, self.fields[self.positive_ind + j], k] -= self.lr * v_grad_l
                                self.V[j, self.fields[self.positive_ind + i], k] -= self.lr * v_grad_r
                            elif self.solver == "adam":
                                V_1[i, self.fields[self.positive_ind + j], k] = self.rho_1 * V_1[
                                    i, self.fields[self.positive_ind + j], k] + (1 - self.rho_1) * v_grad_l
                                V_2[i, self.fields[self.positive_ind + j], k] = self.rho_2 * V_2[
                                    i, self.fields[self.positive_ind + j], k] + (1 - self.rho_2) * v_grad_l * v_grad_l
                                v_1_l = V_1[i, self.fields[self.positive_ind + j], k] / (
                                    1 - np.power(self.rho_1, count))
                                v_2_l = V_2[i, self.fields[self.positive_ind + j], k] / (
                                    1 - np.power(self.rho_2, count))

                                V_1[j, self.fields[self.positive_ind + i], k] = self.rho_1 * V_1[
                                    j, self.fields[self.positive_ind + i], k] + (1 - self.rho_1) * v_grad_r
                                V_2[j, self.fields[self.positive_ind + i], k] = self.rho_2 * V_2[
                                    j, self.fields[self.positive_ind + i], k] + (1 - self.rho_2) * v_grad_r * v_grad_r
                                v_1_r = V_1[j, self.fields[self.positive_ind + i], k] / (
                                    1 - np.power(self.rho_1, count))
                                v_2_r = V_2[j, self.fields[self.positive_ind + i], k] / (
                                    1 - np.power(self.rho_2, count))

                                self.V[i, self.fields[self.positive_ind + j], k] -= (self.lr * v_1_l) / (
                                    np.sqrt(v_2_l) + 1e-8)

                                self.V[j, self.fields[self.positive_ind + i], k] -= (self.lr * v_1_r) / (
                                    np.sqrt(v_2_r) + 1e-8)

                # 计算eval loss
                eval_loss = None
                if eval_set is not None:
                    eval_x, eval_y = eval_set
                    if self.objective == 'logistic':
                        eval_loss = np.mean(eval_y != self.predict(eval_x))
                    else:
                        eval_loss = np.std(eval_y - self.predict(eval_x))
                    eval_losses.append(eval_loss)
                # 是否显示
                if show_log:
                    if self.objective == 'logistic':
                        train_loss = np.mean(y != self.predict(X_o))
                    else:
                        train_loss = np.std(y - self.predict(X_o))
                    print("epoch:", epoch + 1, "/", self.epochs, ",samples:", (index + 1) * self.batch_size, "/",
                          n_sample,
                          ",train loss:",
                          train_loss, ",eval loss:", eval_loss)
                    train_losses.append(train_loss)
                # 是否早停
                if eval_loss is not None and self.early_stopping_rounds is not None:
                    if eval_loss < best_eval_value:
                        eval_count = 0
                        best_eval_value = eval_loss
                    else:
                        eval_count += 1
                    if eval_count >= self.early_stopping_rounds:
                        print("---------------early_stopping-----------------------------")
                        break

        return train_losses, eval_losses

    def predict_proba(self, X):
        """
        logistic regression用
        :param X:
        :return:
        """
        # 归一化
        if self.normal:
            X = (X - self.xmin) / self.xmax
        # reshape
        X = X[:, self.replace_ind]
        # 去掉第一列bias以及非组合特征
        X_ = X[:, self.positive_ind:]
        n_sample, n_feature = X_.shape
        pol = np.zeros(n_sample)
        for i in range(0, n_feature - 1):
            for j in range(i + 1, n_feature):
                pol += X_[:, i] * X_[:, j] * np.dot(self.V[i, self.fields[self.positive_ind + j]],
                                                    self.V[j, self.fields[self.positive_ind + i]])
        pos_proba = utils.sigmoid(np.c_[np.ones(n_sample), X] @ self.w.reshape(-1) + pol)
        return np.c_[1.0 - pos_proba, pos_proba]

    def predict(self, X):
        """
        :param X:
        :return:
        """
        # 归一化
        if self.normal:
            X = (X - self.xmin) / self.xmax
        # reshape
        X = X[:, self.replace_ind]
        # 去掉第一列bias以及非组合特征
        X_ = X[:, self.positive_ind:]
        n_sample, n_feature = X_.shape
        pol = np.zeros(n_sample)
        for i in range(0, n_feature - 1):
            for j in range(i + 1, n_feature):
                pol += X_[:, i] * X_[:, j] * np.dot(self.V[i, self.fields[self.positive_ind + j]],
                                                    self.V[j, self.fields[self.positive_ind + i]])

        linear_rst = np.c_[np.ones(n_sample), X] @ self.w.reshape(-1) + pol
        if self.objective == "squarederror":
            return linear_rst
        elif self.objective in ["poisson", "gamma", "tweedie"]:
            return np.exp(linear_rst)
        else:
            return utils.sigmoid(linear_rst) > 0.5
