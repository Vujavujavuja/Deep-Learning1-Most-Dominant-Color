import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\nvuji\OneDrive\Documents\GitHub\Deep-Learning1-Most-Dominant-Color\dataset\data\dataset.csv')

def to_rgb(color_string):
    clean_string = color_string.replace("(", "").replace(")", "").replace("np.int64", "").strip()
    return tuple(map(int, clean_string.split(",")))

data['RGB'] = data['Dominant Color'].apply(to_rgb)

def is_dark_color(rgb):
    return all(component < 1 for component in rgb)

dark_images = data[data['RGB'].apply(is_dark_color)]

dark_image_files = dark_images.iloc[:, 0].tolist()
print("Dark image files with average RGB < (10, 10, 10):")
count = 0
for file_name in dark_image_files:
    if file_name.endswith('.jpg'):
        count += 1
        print(file_name)

print(f"Total dark images: {count}")
