import os
import csv
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from dataset.analyze import dataset_path

print(f"Number of GPUs available: {cuda.Device.count()}")
device = cuda.Device(0)
print(f"GPU Name: {device.name()}")

kernel_code = """
extern "C"
__global__ void count_rgb(unsigned char *image, int *counters, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < width && idy < height) {
        int pixel_index = (idy * width + idx) * 3; // RGB values
        int r = image[pixel_index];
        int g = image[pixel_index + 1];
        int b = image[pixel_index + 2];

        // Flatten RGB value into a single key
        int color_key = (r << 16) | (g << 8) | b; // Combine R, G, B into a single integer key
        atomicAdd(&counters[color_key], 1);       // Atomic increment of the corresponding color counter
    }
}
"""

mod = SourceModule(kernel_code)
count_rgb = mod.get_function("count_rgb")

def process_image(image_path):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    image_flat = image.astype(np.uint8).flatten()

    image_gpu = cuda.mem_alloc(image_flat.nbytes)
    cuda.memcpy_htod(image_gpu, image_flat)

    counters_gpu = cuda.mem_alloc(2**24 * np.int32().nbytes)
    counters = np.zeros(2**24, dtype=np.int32)
    cuda.memcpy_htod(counters_gpu, counters)

    block = (16, 16, 1)
    grid = ((width + block[0] - 1) // block[0], (height + block[1] - 1) // block[1])

    count_rgb(image_gpu, counters_gpu, np.int32(width), np.int32(height), block=block, grid=grid)

    cuda.memcpy_dtoh(counters, counters_gpu)

    image_gpu.free()
    counters_gpu.free()

    most_common_color_index = np.argmax(counters)
    r = (most_common_color_index >> 16) & 0xFF
    g = (most_common_color_index >> 8) & 0xFF
    b = most_common_color_index & 0xFF
    dominant_color = (r, g, b)
    hex_color = f"#{r:02X}{g:02X}{b:02X}"

    return dominant_color, hex_color

def process_dataset(dataset_path, output_csv):
    results = []

    for img_file in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, img_file)
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            try:
                dominant_color, hex_color = process_image(image_path)
                print(f"Processed {img_file}: Dominant Color = {dominant_color}")
                results.append((img_file, dominant_color, hex_color))
            except Exception as e:
                print(f"Failed to process {img_file}: {e}")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name", "Dominant Color", "Hex Color"])
        writer.writerows(results)

    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    dataset_path = "Hyacinth (Hyacinthus orientalis)"
    #dataset_path = "test"
    #output_csv = "test.csv"
    output_csv = "dataset.csv"
    process_dataset(dataset_path, output_csv)
