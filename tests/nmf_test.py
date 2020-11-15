import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

# 造伪样本
docs = [
    ["有", "微信", "红包", "的", "软件"],
    ["微信", "支付", "不行", "的"],
    ["我们", "需要", "稳定的", "微信", "支付", "接口"],
    ["申请", "公众号", "认证"],
    ["这个", "还有", "几天", "放", "垃圾", "流量"],
    ["可以", "提供", "聚合", "支付", "系统"]
]
word2id = {}
idx = 0
W = []
for doc in docs:
    tmp = []
    for word in doc:
        if word in word2id:
            tmp.append(word2id[word])
        else:
            word2id[word] = idx
            idx += 1
            tmp.append(word2id[word])
    W.append(tmp)

data = np.zeros(shape=(len(docs), len(word2id)))
for idx, w in enumerate(W):
    for i in w:
        data[idx][i] = 1

from ml_models.decomposition import NMF

nmf = NMF(n_components=4)
trans = nmf.fit_transform(data)


def cosine(x1, x2):
    return x1.dot(x2) / (np.sqrt(np.sum(np.power(x1, 2))) * np.sqrt(np.sum(np.power(x2, 2))))


print(cosine(trans[1], trans[2]))
print(cosine(trans[1], trans[3]))
print(cosine(trans[1], trans[4]))
print(cosine(trans[1], trans[0]))