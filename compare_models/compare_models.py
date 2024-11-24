

lenet_v1 = ['5.18%', '3.57%', '9.56%', '13.02%', '13.06%']
resnet_v1 = ['28.42%', '33.18%', '32.80%', '44.56%', '39.81%']
resnet_v2 = ['30.28%', '44.52%', '52.61%', '55.27%', '57.33%']


import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 5, 10, 15, 20])
y1 = np.array([float(i[:-1]) for i in lenet_v1])
y2 = np.array([float(i[:-1]) for i in resnet_v1])
y3 = np.array([float(i[:-1]) for i in resnet_v2])

plt.plot(x, y1, label='LeNet v1')
plt.plot(x, y2, label='ResNet v1')
plt.plot(x, y3, label='ResNet v2')

plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy')
plt.legend()

plt.show()