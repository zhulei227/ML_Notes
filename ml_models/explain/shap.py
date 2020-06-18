"""
对论文：Consistent feature attribution for tree ensembles中的两个算法进行实现
注意：论文中的公式有误，将(M-|S|!-1)修改为(M-|S|-1)!
"""

import numpy as np
import itertools
import copy


class Shap(object):
    def __init__(self, total_tree):
        self.total_tree = total_tree

    def EXPVALUE(self, x, S, tree):
        """
        对应论文中的EXPVALUE函数
        :param x: 数据
        :param S: 特征集合
        :param tree: 树
        :return:
        """

        # 对应论文中的G函数
        def G(tree, w):
            if tree.get('leaf_value') is not None:
                return w * tree['leaf_value']
            else:
                if tree['split_feature'] in S:
                    if x[tree['split_feature']] <= tree['threshold']:
                        return G(tree['left_child'], w)
                    else:
                        return G(tree['right_child'], w)
                else:
                    left_weight = tree['left_child'].get('internal_count') if tree['left_child'].get(
                        'internal_count') is not None else tree['left_child'].get('leaf_count')
                    right_weight = tree['right_child'].get('internal_count') if tree['right_child'].get(
                        'internal_count') is not None else tree['right_child'].get('leaf_count')
                    return G(tree['left_child'], w * left_weight / (left_weight + right_weight)) + G(
                        tree['right_child'], w * right_weight / (left_weight + right_weight))

        return G(tree, 1)

    def pred_one_contrib(self, x):
        n_feature = len(x)
        # 1.获取所有的组合
        combine_sets = []
        for num in range(0, n_feature + 1):
            combine_sets += itertools.combinations(list(range(0, n_feature)), r=num)
        # 2.遍历每一个颗树，计算对应组合的预测值
        phi = np.zeros(len(x) + 1)
        for item in self.total_tree['tree_info']:
            # 2.1 计算当前树的所有集合上的取值
            tree = item['tree_structure']
            combine_set_value = {}
            for S in combine_sets:
                v = self.EXPVALUE(x, S, tree)
                combine_set_value[S] = v

            # 2.2 计算每个特征的重要性
            phi[-1] += combine_set_value[()]  # 最后一个存储空集结果/即根节点的结果
            for f_num in range(0, n_feature):
                for S in combine_sets:
                    if f_num in S:
                        new_S = S[:S.index(f_num)] + S[S.index(f_num) + 1:]
                        phi[f_num] += np.math.factorial(len(new_S)) * np.math.factorial(
                            max(0, n_feature - len(new_S) - 1)) / np.math.factorial(n_feature) * (
                                          combine_set_value[S] - combine_set_value[new_S])
        return phi


class TreeShap(object):
    def __init__(self, total_tree):
        self.total_tree = total_tree

    def pred_one_contrib(self, x):
        n_feature = len(x) + 1
        phi = np.zeros(n_feature)

        # 对应论文中的RECURSE函数
        def RECURSE(tree, m, p_z, p_o, p_i):
            m = EXTEND(m, p_z, p_o, p_i)
            if tree.get('leaf_value') is not None:
                for i in range(1, len(m) + 1):
                    w = 0.0
                    for _, item in UNWIND(m, i).items():
                        w += item['w']
                    phi[m[i]['d']] += w * (m[i]['o'] - m[i]['z']) * tree['leaf_value']
            else:
                h, c = (tree['left_child'], tree['right_child']) if x[tree['split_feature']] <= tree['threshold'] else (
                    tree['right_child'], tree['left_child'])
                i_z = i_o = 1
                k = FINDFIRST(m, tree['split_feature'])
                if k is not None:
                    i_z, i_o = m[k]['z'], m[k]['o']
                    m = UNWIND(m, k)
                r_h = h.get('internal_count') if h.get('internal_count') is not None else h.get('leaf_count')
                r_c = c.get('internal_count') if c.get('internal_count') is not None else c.get('leaf_count')
                r_j = r_h + r_c
                RECURSE(h, m, i_z * r_h / r_j, i_o, tree['split_feature'])
                RECURSE(c, m, i_z * r_c / r_j, 0, tree['split_feature'])

        # 对应论文中的EXTEND函数
        def EXTEND(m, p_z, p_o, p_i):
            l = len(m)
            m_ = copy.deepcopy(m)
            m_[l + 1] = {}
            m_[l + 1]['d'] = p_i
            m_[l + 1]['z'] = p_z
            m_[l + 1]['o'] = p_o
            m_[l + 1]['w'] = 1 if l == 0 else 0
            for i in range(l - 1, 0, -1):
                if m_.get(i + 1) is None:
                    m_[i + 1] = {}
                    m_[i + 1]['w'] = 0
                m_[i + 1]['w'] += p_o * m_[i]['w'] * (i + 1) / (l + 1)

                if m_.get(i) is None:
                    m_[i] = {}
                    m_[i]['w'] = 0
                m_[i]['w'] = p_z * m_[i]['w'] * (l - i) / (l + 1)

            return m_

        # 对应论文中的UNWIND函数
        def UNWIND(m, i):
            l = len(m)
            n = m[l]['w']
            m_ = {}
            for i in range(1, l + 1):
                m_[i] = copy.deepcopy(m[i])
            for j in range(l - 1, 0, -1):
                if m_[i]['o'] != 0:
                    t = m_[j]['w']
                    m_[j]['w'] = n * (l + 1) / ((j + 1) * m_[i]['o'])
                    n = t - m_[j]['w'] * m_[i]['z'] * (l - j) / (l + 1)
                else:
                    m_[j]['w'] = m_[j]['w'] * (l + 1) / (m_[i]['z'] * (l - j))
            for j in range(i, l):
                m_[j]['d'] = m_[j + 1]['d']
                m_[j]['z'] = m_[j + 1]['z']
                m_[j]['o'] = m_[j + 1]['o']
            return m_

        # 对应FINDFIRST函数
        def FINDFIRST(m, d_j):
            for i in range(1, len(m) + 1):
                if m[i].get('d') == d_j:
                    return i
            return None

        for item in self.total_tree['tree_info']:
            tree = item['tree_structure']
            m = {}
            RECURSE(tree, m, 1, 1, 0)
        return phi