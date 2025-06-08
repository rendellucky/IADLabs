import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, AffinityPropagation
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances_argmin
from sklearn.preprocessing import StandardScaler
X = np.loadtxt("LR2/data_clustering.txt", delimiter=',')

iris = load_iris()
X_iris = iris.data
kmeans_iris = KMeans(n_clusters=3)
kmeans_iris.fit(X_iris)
y_kmeans = kmeans_iris.predict(X_iris)

plt.figure(figsize=(6, 6))
plt.scatter(X_iris[:, 0], X_iris[:, 1], c=y_kmeans, s=80, cmap='viridis', edgecolors='black')
centers = kmeans_iris.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title("Кластеризація Iris методом k-середніх")
plt.xticks([])
plt.yticks([])
plt.savefig('task2_iris.png', dpi=300)
plt.show()