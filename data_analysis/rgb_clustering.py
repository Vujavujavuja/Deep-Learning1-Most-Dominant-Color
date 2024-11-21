import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

data = pd.read_csv(r'C:\Users\nvuji\OneDrive\Documents\GitHub\Deep-Learning1-Most-Dominant-Color\dataset\data\dataset.csv')

def to_rgb(color_string):
    clean_string = color_string.replace("(", "").replace(")", "").replace("np.int64", "").strip()
    return tuple(map(int, clean_string.split(",")))

data['RGB'] = data['Dominant Color'].apply(to_rgb)
rgb_values = np.array(data['RGB'].tolist())

num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(rgb_values)

data['Cluster'] = clusters

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.get_cmap('tab10', num_clusters)

for cluster_id in range(num_clusters):
    cluster_points = rgb_values[clusters == cluster_id]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
               label=f'Cluster {cluster_id}', color=colors(cluster_id), s=40, alpha=0.8)

cluster_centers = kmeans.cluster_centers_
ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
           color='black', s=100, label='Cluster Centers', edgecolor='k')

ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
ax.set_title('RGB Clustering')
ax.legend()
plt.show()

for i, center in enumerate(cluster_centers):
    center_normalized = center / 255
    plt.imshow([[center_normalized]], aspect='auto')
    plt.title(f'Cluster {i} Center: RGB {center.astype(int)}')
    plt.axis('off')
    plt.show()
