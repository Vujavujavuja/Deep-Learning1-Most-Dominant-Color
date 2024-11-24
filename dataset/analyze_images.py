import os
import cv2
import shutil

dataset_path = r"C:\Users\nvuji\.cache\kagglehub\datasets\kacpergregorowicz\house-plant-species\versions\4\house_plant_species"
destination_folder = "images1"

os.makedirs(destination_folder, exist_ok=True)

total_images = 0
dimensions = []

image_counter = 0

for root, _, files in os.walk(dataset_path):
    for img_file in files:
        img_path = os.path.join(root, img_file)

        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(img_path)
            if img is not None:
                dimensions.append(img.shape[:2])
                total_images += 1

                new_image_name = f"{image_counter}.jpg"
                print(new_image_name)
                image_counter += 1

                shutil.copy(img_path, os.path.join(destination_folder, new_image_name))

print(f"Total images: {total_images}")
print(f"Unique dimensions: {set(dimensions)}")
