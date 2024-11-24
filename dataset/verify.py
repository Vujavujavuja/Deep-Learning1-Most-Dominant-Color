import os
from PIL import Image

dataset_path = "images1"
invalid_images = []

for img_file in os.listdir(dataset_path):
    try:
        with Image.open(os.path.join(dataset_path, img_file)) as img:
            img.verify()
    except Exception:
        invalid_images.append(img_file)

if invalid_images:
    print("Invalid images:", invalid_images)
else:
    print("All images are valid")
