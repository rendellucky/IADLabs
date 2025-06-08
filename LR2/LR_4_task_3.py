import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, AffinityPropagation
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances_argmin
from sklearn.preprocessing import StandardScaler
X = np.loadtxt("LR2/data_clustering.txt", delimiter=',')

bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels_ms = ms.labels_
cluster_centers = ms.cluster_centers_
n_clusters_ = len(np.unique(labels_ms))

plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels_ms, cmap='Accent', edgecolors='black', s=80)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=200, alpha=0.5)
plt.title(f"Mean Shift: кількість кластерів = {n_clusters_}")
plt.xticks([])
plt.yticks([])
plt.savefig('task3_meanshift.png', dpi=300)
plt.show()