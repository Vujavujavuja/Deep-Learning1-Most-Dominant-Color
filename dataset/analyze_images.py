import os
import cv2

dataset_path = "Hyacinth (Hyacinthus orientalis)"
print(f"Total images: {len(os.listdir(dataset_path))}")

dimensions = []
for img_file in os.listdir(dataset_path):
    img = cv2.imread(os.path.join(dataset_path, img_file))
    dimensions.append(img.shape[:2])

print(f"Unique dimensions: {set(dimensions)}")