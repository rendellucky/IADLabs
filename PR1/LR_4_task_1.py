import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, AffinityPropagation
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances_argmin
from sklearn.preprocessing import StandardScaler
X = np.loadtxt("PR1/data_clustering.txt", delimiter=',')

plt.figure(figsize=(6, 6))
plt.scatter(
    X[:, 0], X[:, 1],
    marker='o',
    facecolors='none',
    edgecolors='black',
    s=80
)
plt.title('Вхідні дані')

x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks([])
plt.yticks([])
plt.savefig('task1_input.png', dpi=300)
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=80, edgecolors='black')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.5)
plt.title("Результат кластеризації k-середніх")
plt.xticks([])
plt.yticks([])
plt.savefig('task1_result.png', dpi=300)
plt.show()