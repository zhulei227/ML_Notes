from sklearn.datasets.samples_generator import make_blobs

X, y = make_blobs(n_samples=400, centers=4, cluster_std=0.85, random_state=0)
X = X[:, ::-1]

from ml_models.em import GMMCluster

gmm = GMMCluster(verbose=True, n_iter=100, n_components=4)
gmm.fit(X)
