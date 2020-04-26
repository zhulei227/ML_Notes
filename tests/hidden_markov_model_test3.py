import numpy as np
from ml_models.pgm import HMM

pi = np.asarray([[0.2], [0.4], [0.4]])
A = np.asarray([[0.5, 0.2, 0.3],
                [0.3, 0.5, 0.2],
                [0.2, 0.3, 0.5]])
B = np.asarray([[0.5, 0.5],
                [0.4, 0.6],
                [0.7, 0.3]])

hmm = HMM(hidden_status_num=3, visible_status_num=2)
hmm.pi = pi
hmm.A = A
hmm.B = B
print(hmm.predict_hidden_status([0, 1, 0]))
