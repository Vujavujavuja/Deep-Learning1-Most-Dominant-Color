import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv(r'C:\Users\nvuji\OneDrive\Documents\GitHub\Deep-Learning1-Most-Dominant-Color\dataset\data\dataset.csv')

def to_rgb(color_string):
    clean_string = color_string.replace("(", "").replace(")", "").replace("np.int64", "").strip()
    return tuple(map(int, clean_string.split(",")))

data['RGB'] = data['Dominant Color'].apply(to_rgb)
rgb_values = np.array(data['RGB'].tolist())

num_colors = 10
kmeans = KMeans(n_clusters=num_colors, random_state=42)
clusters = kmeans.fit_predict(rgb_values)

unique_colors = kmeans.cluster_centers_

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('10 Most Unique Colors', fontsize=16)

for i, ax in enumerate(axes.flatten()):
    color = unique_colors[i] / 255
    ax.imshow([[color]], aspect='auto')
    ax.set_title(f'RGB: {unique_colors[i].astype(int)}', fontsize=10)
    ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
