"""
DEC (Deep Embedded Clustering) with elbow method and silhouette coefficient to determine K=3.
This script simulates the clustering of stratigraphic samples (using synthetic features)
and outputs soft partition probabilities.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt

# ---------------------------
# Generate synthetic features (simulating deep autoencoder latent features)
# ---------------------------
np.random.seed(42)
n_samples = 2200  # 220 boreholes × 10 layers
n_features = 3    # latent dimension

# Simulate three clusters (shallow, middle, deep) with distinct means
cluster_centers_true = [
    [2.0, 1.5, 1.0],   # shallow assemblage
    [0.0, 0.0, 0.0],   # middle transition
    [-2.0, -1.5, -1.0] # deep coal-bearing
]
X = []
y_true = []
for i, center in enumerate(cluster_centers_true):
    n = n_samples // 3
    cluster_points = np.random.randn(n, n_features) * 0.5 + center
    X.append(cluster_points)
    y_true.extend([i] * n)
X = np.vstack(X)

# ---------------------------
# Elbow method and silhouette coefficient to determine optimal K
# ---------------------------
K_range = range(2, 11)
inertias = []
silhouette_scores = []
for k in K_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    if k >= 2:
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)

# Plot elbow + silhouette
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_xlabel('Number of clusters K')
axes[0].set_ylabel('Inertia (SSE)')
axes[0].set_title('Elbow Method')
axes[0].axvline(x=3, color='r', linestyle='--', label='K=3')
axes[0].legend()

axes[1].plot(list(K_range), silhouette_scores, 'ro-')
axes[1].set_xlabel('Number of clusters K')
axes[1].set_ylabel('Silhouette Coefficient')
axes[1].set_title('Silhouette Method')
axes[1].axvline(x=3, color='b', linestyle='--', label='K=3')
axes[1].legend()
plt.tight_layout()
plt.savefig('K_selection.png', dpi=300)
# plt.show()  # 注释掉，避免 PyCharm 报错

print(f"Optimal K from silhouette: {K_range[np.argmax(silhouette_scores)]}")

# ---------------------------
# Perform clustering with K=3 and compute soft partition probabilities
# ---------------------------
kmeans_final = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = kmeans_final.fit_predict(X)

# Compute soft assignment probabilities based on inverse distance
def soft_assignment(X, centers, alpha=1.0):
    distances = np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2)
    # t-distribution based similarity
    q = 1.0 / (1.0 + distances / alpha)
    q = q / q.sum(axis=1, keepdims=True)
    return q

soft_probs = soft_assignment(X, kmeans_final.cluster_centers_)

print("Soft partition probabilities for first 5 samples:\n", soft_probs[:5])
print("Hard labels (max probability) for first 5 samples:", np.argmax(soft_probs[:5], axis=1))

# Plot silhouette for K=3
fig, ax = plt.subplots(figsize=(8, 6))
sil_vals = silhouette_samples(X, labels)
y_lower = 0
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i in range(3):
    cluster_sil = sil_vals[labels == i]
    cluster_sil.sort()
    y_upper = y_lower + len(cluster_sil)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil, facecolor=colors[i], edgecolor='none')
    ax.text(-0.05, (y_lower + y_upper)/2, f'Cluster {i+1}', fontsize=10)
    y_lower = y_upper
ax.axvline(x=np.mean(sil_vals), color='red', linestyle='--', label=f'Mean silhouette: {np.mean(sil_vals):.3f}')
ax.set_xlabel('Silhouette coefficient')
ax.set_ylabel('Samples')
ax.set_title('Silhouette plot for K=3')
ax.legend()
plt.tight_layout()
plt.savefig('silhouette_K3.png', dpi=800)
# plt.show()