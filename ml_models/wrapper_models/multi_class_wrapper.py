import threading
import copy
import numpy as np

"""
继承Thread,获取函数的返回值
"""


class MyThread(threading.Thread):
    def __init__(self, target, args, kwargs, name=''):
        threading.Thread.__init__(self)
        self.name = name
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.result = self.target(*self.args, **self.kwargs)

    def get_result(self):
        try:
            return self.result
        except:
            return None


class MultiClassWrapper(object):
    def __init__(self, base_classifier, mode='ovr'):
        """
        :param base_classifier: 实例化后的分类器
        :param mode: 'ovr'表示one-vs-rest方式,'ovo'表示one-vs-one方式
        """
        self.base_classifier = base_classifier
        self.mode = mode

    @staticmethod
    def fit_base_classifier(base_classifier, x, y, **kwargs):
        base_classifier.fit(x, y, **kwargs)

    @staticmethod
    def predict_proba_base_classifier(base_classifier, x):
        return base_classifier.predict_proba(x)

    def fit(self, x, y, **kwargs):
        # 对y分组并行fit
        self.n_class = np.max(y)
        if self.mode == 'ovr':
            # 打包数据
            self.classifiers = []

            for cls in range(0, self.n_class + 1):
                self.classifiers.append(copy.deepcopy(self.base_classifier))
            # 并行训练
            tasks = []
            for cls in range(len(self.classifiers)):
                task = MyThread(target=self.fit_base_classifier,
                                args=(self.classifiers[cls], x, (y == cls).astype('int')), kwargs=kwargs)
                task.start()
                tasks.append(task)
            for task in tasks:
                task.join()
        elif self.mode == "ovo":
            # 打包数据
            self.classifiers = {}
            for first_cls in range(0, self.n_class):
                for second_cls in range(first_cls + 1, self.n_class + 1):
                    self.classifiers[(first_cls, second_cls)] = copy.deepcopy(self.base_classifier)
            # 并行训练
            tasks = {}
            for first_cls in range(0, self.n_class):
                for second_cls in range(first_cls + 1, self.n_class + 1):
                    index = np.where(y == first_cls)[0].tolist() + np.where(y == second_cls)[0].tolist()
                    new_x = x[index, :]
                    new_y = y[index]
                    task = MyThread(target=self.fit_base_classifier,
                                    args=(self.classifiers[(first_cls, second_cls)], new_x,
                                          (new_y == first_cls).astype('int')), kwargs=kwargs)
                    task.start()
                    tasks[(first_cls, second_cls)] = task
            for first_cls in range(0, self.n_class):
                for second_cls in range(first_cls + 1, self.n_class + 1):
                    tasks[(first_cls, second_cls)].join()

    def predict_proba(self, x, **kwargs):
        if self.mode == 'ovr':
            tasks = []
            probas = []
            for cls in range(len(self.classifiers)):
                task = MyThread(target=self.predict_proba_base_classifier, args=(self.classifiers[cls], x),
                                kwargs=kwargs)
                task.start()
                tasks.append(task)
            for task in tasks:
                task.join()
            for task in tasks:
                probas.append(task.get_result()[:,1])
            total_probas = np.asarray(probas).T
            # 归一化
            return total_probas / total_probas.sum(axis=1, keepdims=True)
        elif self.mode == 'ovo':
            tasks = {}
            probas = {}
            for first_cls in range(0, self.n_class):
                for second_cls in range(first_cls + 1, self.n_class + 1):
                    task = MyThread(target=self.predict_proba_base_classifier,
                                    args=(self.classifiers[(first_cls, second_cls)], x), kwargs=kwargs)
                    task.start()
                    tasks[(first_cls, second_cls)] = task
            for first_cls in range(0, self.n_class):
                for second_cls in range(first_cls + 1, self.n_class + 1):
                    tasks[(first_cls, second_cls)].join()
            for first_cls in range(0, self.n_class):
                for second_cls in range(first_cls + 1, self.n_class + 1):
                    probas[(first_cls, second_cls)] = tasks[(first_cls, second_cls)].get_result()
                    probas[(second_cls, first_cls)] = 1.0 - probas[(first_cls, second_cls)]
            # 统计概率
            total_probas = []
            for first_cls in range(0, self.n_class + 1):
                temp = []
                for second_cls in range(0, self.n_class + 1):
                    if first_cls != second_cls:
                        temp.append(probas[(first_cls, second_cls)])
                temp = np.sum(temp,axis=0)[:,1]
                total_probas.append(temp)
            # 归一化
            total_probas = np.asarray(total_probas).T
            return total_probas / total_probas.sum(axis=1, keepdims=True)

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)
