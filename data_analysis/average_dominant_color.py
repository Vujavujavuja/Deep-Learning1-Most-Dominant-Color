import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv(r'C:\Users\nvuji\OneDrive\Documents\GitHub\Deep-Learning1-Most-Dominant-Color\dataset\data\dataset.csv')

def to_rgb(color_string):
    clean_string = color_string.replace("(", "").replace(")", "").replace("np.int64", "").strip()
    return tuple(map(int, clean_string.split(",")))

data['RGB'] = data['Dominant Color'].apply(to_rgb)

average_color = np.mean(data['RGB'].tolist(), axis=0)

print(f"Average RGB Color: {average_color}")


average_color_normalized = average_color / 255
plt.imshow([[average_color_normalized]])
plt.title(f"Average Color: {average_color.astype(int)}")
plt.axis('off')
plt.show()