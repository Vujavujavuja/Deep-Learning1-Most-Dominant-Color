import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r'C:\Users\nvuji\OneDrive\Documents\GitHub\Deep-Learning1-Most-Dominant-Color\dataset\data\dataset.csv')

def to_rgb(color_string):
    clean_string = color_string.replace("(", "").replace(")", "").replace("np.int64", "").strip()
    return tuple(map(int, clean_string.split(",")))

data['RGB'] = data['Dominant Color'].apply(to_rgb)

data = data[data['RGB'].apply(lambda rgb: sum(rgb) >= 50)]

def count_dominant_color(rgb_values):
    red_count = sum(1 for rgb in rgb_values if rgb[0] > rgb[1] and rgb[0] > rgb[2])
    green_count = sum(1 for rgb in rgb_values if rgb[1] > rgb[0] and rgb[1] > rgb[2])
    blue_count = sum(1 for rgb in rgb_values if rgb[2] > rgb[0] and rgb[2] > rgb[1])
    return red_count, green_count, blue_count

red_count, green_count, blue_count = count_dominant_color(data['RGB'])

labels = ['Red', 'Green', 'Blue']
sizes = [red_count, green_count, blue_count]
colors = ['red', 'green', 'blue']

plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Proportion of Dominant Colors (R, G, B)')
plt.show()
