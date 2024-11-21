import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\nvuji\OneDrive\Documents\GitHub\Deep-Learning1-Most-Dominant-Color\dataset\data\dataset.csv')

def to_rgb(color_string):
    clean_string = color_string.replace("(", "").replace(")", "").replace("np.int64", "").strip()
    return tuple(map(int, clean_string.split(",")))

data['RGB'] = data['Dominant Color'].apply(to_rgb)

data = data[data['RGB'].apply(lambda rgb: sum(rgb) >= 20)]

def calculate_black_white_proximity(rgb):
    brightness = sum(rgb) / (255 * 3)
    return 'white' if brightness > 0.5 else 'black'

data['Category'] = data['RGB'].apply(calculate_black_white_proximity)

def is_red_dominant(rgb):
    return rgb[0] > rgb[1] and rgb[0] > rgb[2]

def is_green_dominant(rgb):
    return rgb[1] > rgb[0] and rgb[1] > rgb[2]

def is_blue_dominant(rgb):
    return rgb[2] > rgb[0] and rgb[2] > rgb[1]

red_dominant = data[data['RGB'].apply(is_red_dominant)]
green_dominant = data[data['RGB'].apply(is_green_dominant)]
blue_dominant = data[data['RGB'].apply(is_blue_dominant)]

def count_dark_light(data):
    dark_count = len(data[data['Category'] == 'black'])
    light_count = len(data[data['Category'] == 'white'])
    return dark_count, light_count

red_dark, red_light = count_dark_light(red_dominant)
green_dark, green_light = count_dark_light(green_dominant)
blue_dark, blue_light = count_dark_light(blue_dominant)

fig, ax = plt.subplots(figsize=(10, 6))

dark_heights = [red_dark, green_dark, blue_dark]
light_heights = [red_light, green_light, blue_light]

x = np.arange(len(dark_heights))
bar_width = 0.35

ax.bar(x - bar_width / 2, dark_heights, bar_width, label='Dark Colors', color=['red', 'green', 'blue'])

ax.bar(x + bar_width / 2, light_heights, bar_width, label='Light Colors', color=['pink', 'lightgreen', 'lightblue'])

ax.set_xticks(x)
ax.set_xticklabels(['Red Dominant', 'Green Dominant', 'Blue Dominant'])
ax.set_ylabel('Number of Colors')
ax.set_title('Dark and Light Colors by Dominance')
ax.legend()

plt.tight_layout()
plt.show()
