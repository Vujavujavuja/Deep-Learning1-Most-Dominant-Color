import pandas as pd
import os
import shutil

artifact_images = [143, 145, 158, 16, 164, 17, 198, 21, 213, 217, 235, 24, 249, 254, 261, 294, 307, 80, 89]
artifact_images_with_extension = [f"{img}.jpg" for img in artifact_images]

data = pd.read_csv(r'C:\Users\nvuji\OneDrive\Documents\GitHub\Deep-Learning1-Most-Dominant-Color\dataset\data\dataset.csv')
path_to_images = r'C:\Users\nvuji\OneDrive\Documents\GitHub\Deep-Learning1-Most-Dominant-Color\dataset\images1'

destination_folder = r'C:\Users\nvuji\OneDrive\Documents\GitHub\Deep-Learning1-Most-Dominant-Color\preprocessing\images'

os.makedirs(destination_folder, exist_ok=True)

for _, row in data.iterrows():
    image_name = row['Image Name']
    if image_name not in artifact_images_with_extension:
        src_path = os.path.join(path_to_images, image_name)
        dst_path = os.path.join(destination_folder, image_name)
        shutil.copy(src_path, dst_path)
