import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r'C:\Users\nvuji\OneDrive\Documents\GitHub\Deep-Learning1-Most-Dominant-Color\dataset\data\dataset.csv')

def to_rgb(color_string):
    clean_string = color_string.replace("(", "").replace(")", "").replace("np.int64", "").strip()
    return tuple(map(int, clean_string.split(",")))

data['RGB'] = data['Dominant Color'].apply(to_rgb)

def is_red_dominant(rgb):
    return rgb[0] > rgb[1] + rgb[2] and sum(rgb) > 50

def is_green_dominant(rgb):
    return rgb[1] > rgb[0] + rgb[2] and sum(rgb) > 50

def is_blue_dominant(rgb):
    return rgb[2] > rgb[0] + rgb[1] and sum(rgb) > 50

red_dominant = data[data['RGB'].apply(is_red_dominant)]
green_dominant = data[data['RGB'].apply(is_green_dominant)]
blue_dominant = data[data['RGB'].apply(is_blue_dominant)]

def normalize_colors(rgb_values):
    return [np.array(rgb) / 255 for rgb in rgb_values]

red_colors = normalize_colors(red_dominant['RGB'])
green_colors = normalize_colors(green_dominant['RGB'])
blue_colors = normalize_colors(blue_dominant['RGB'])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

if red_colors:
    axes[0].imshow([red_colors], aspect='auto')
    axes[0].set_title('Red > Green + Blue')
    axes[0].axis('off')
else:
    axes[0].text(0.5, 0.5, 'No Red Dominant Colors', horizontalalignment='center', verticalalignment='center')
    axes[0].axis('off')

if green_colors:
    axes[1].imshow([green_colors], aspect='auto')
    axes[1].set_title('Green > Red + Blue')
    axes[1].axis('off')
else:
    axes[1].text(0.5, 0.5, 'No Green Dominant Colors', horizontalalignment='center', verticalalignment='center')
    axes[1].axis('off')

if blue_colors:
    axes[2].imshow([blue_colors], aspect='auto')
    axes[2].set_title('Blue > Red + Green')
    axes[2].axis('off')
else:
    axes[2].text(0.5, 0.5, 'No Blue Dominant Colors', horizontalalignment='center', verticalalignment='center')
    axes[2].axis('off')

plt.tight_layout()
plt.show()
