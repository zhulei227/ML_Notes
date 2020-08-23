import numpy as np
# import os
#
# os.chdir('../')
from ml_models.latent_dirichlet_allocation import LDA

W = []
for _ in range(0, 10):
    W.append(np.random.choice(10, np.random.randint(low=8, high=20)).tolist())

lda = LDA(epochs=100,method='vi_em')
lda.fit(W)
print(lda.transform(W))
