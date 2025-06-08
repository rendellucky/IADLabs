import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, AffinityPropagation
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances_argmin
from sklearn.preprocessing import StandardScaler
X = np.loadtxt("PR1/data_clustering.txt", delimiter=',')

iris = load_iris()
X_iris = iris.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_iris)

ap = AffinityPropagation(random_state=42)
ap.fit(X_scaled)
labels_ap = ap.labels_
cluster_centers_indices = ap.cluster_centers_indices_
n_clusters_ap = len(cluster_centers_indices)

plt.figure(figsize=(6, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_ap, cmap='tab10', edgecolors='black', s=80)
plt.scatter(X_scaled[cluster_centers_indices, 0], X_scaled[cluster_centers_indices, 1],
            c='black', s=200, alpha=0.6, marker='x')
plt.title(f"Affinity Propagation: знайдено кластерів = {n_clusters_ap}")
plt.xticks([])
plt.yticks([])
plt.savefig('task4_affinity.png', dpi=300)
plt.show()