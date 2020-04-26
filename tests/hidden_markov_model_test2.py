import numpy as np
from ml_models.pgm import HMM

O = [
    [1, 2, 3, 0, 1, 3, 4],
    [1, 2, 3],
    [0, 2, 4, 2],
    [4, 3, 2, 1],
    [3, 1, 1, 1, 1],
    [2, 1, 3, 2, 1, 3, 4]
]
I = O

print("------------------有监督学习----------------------")
hmm = HMM(hidden_status_num=5, visible_status_num=5)
hmm.fit_with_hidden_status(visible_list=O, hidden_list=I)
print(hmm.pi)
print(hmm.A)
print(hmm.B)

print("\n------------------无监督学习----------------------")
hmm = HMM(hidden_status_num=5, visible_status_num=5)
hmm.fit_without_hidden_status(O[0] + O[1] + O[2] + O[3] + O[4] + O[5])
print(hmm.pi)
print(hmm.A)
print(hmm.B)
