import pandas as pd

csv_path = r'C:\Users\nvuji\OneDrive\Documents\GitHub\Deep-Learning1-Most-Dominant-Color\dataset\data\dataset.csv'
artifact_images = [143, 145, 158, 16, 164, 17, 198, 21, 213, 217, 235, 24, 249, 254, 261, 294, 307, 80, 89]
artifact_images_with_extension = [f"{img}.jpg" for img in artifact_images]

def to_rgb(color_string):
    clean_string = color_string.replace("(", "").replace(")", "").replace("np.int64", "").strip()
    return tuple(map(int, clean_string.split(",")))

data = pd.read_csv(csv_path)

data['Dominant Color'] = data['Dominant Color'].apply(to_rgb)


if 'Hex Color' in data.columns:
    data = data.drop(columns=['Hex Color'])

filtered_data = data[~data['Image Name'].isin(artifact_images_with_extension)]

new_csv_path = r'C:\Users\nvuji\OneDrive\Documents\GitHub\Deep-Learning1-Most-Dominant-Color\preprocessing\cleaned_dataset.csv'
filtered_data.to_csv(new_csv_path, index=False)

