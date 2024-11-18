import pycuda.driver as cuda
import pycuda.autoinit

print(f"Number of GPUs available: {cuda.Device.count()}")
device = cuda.Device(0)
print(f"GPU Name: {device.name()}")

