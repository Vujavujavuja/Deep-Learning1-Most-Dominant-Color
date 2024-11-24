import torch
import matplotlib.pyplot as plt
from torchvision import models

resnet1 = models.resnet18(pretrained=False)
resnet1.fc = torch.nn.Linear(resnet1.fc.in_features, 3)

resnet2 = models.resnet18(pretrained=False)
resnet2.fc = torch.nn.Linear(resnet2.fc.in_features, 3)

class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5, padding=2)
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2)
        self.fc1 = torch.nn.Linear(16 * 56 * 56, 120)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(120, 84)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc3 = torch.nn.Linear(84, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.avg_pool(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        return self.fc3(x)

lenet1 = LeNet()

resnet1.load_state_dict(torch.load(r'C:\Users\nvuji\OneDrive\Documents\GitHub\Deep-Learning1-Most-Dominant-Color\models\resnet_v1.pth'))
resnet2.load_state_dict(torch.load(r'C:\Users\nvuji\OneDrive\Documents\GitHub\Deep-Learning1-Most-Dominant-Color\models\resnet_v2.pth'))
lenet1.load_state_dict(torch.load(r'C:\Users\nvuji\OneDrive\Documents\GitHub\Deep-Learning1-Most-Dominant-Color\models\lenet_v1.pth'))

print(resnet1)
print(resnet2)
print(lenet1)

for (name1, param1), (name2, param2) in zip(resnet1.named_parameters(), resnet2.named_parameters()):
    if name1 == name2:
        distance = torch.dist(param1, param2, p=2).item()
        print(f"L2 distance between {name1}: {distance}")


def plot_weights(model, title):
    weights = []
    for param in model.parameters():
        if len(param.shape) > 1:
            weights.extend(param.flatten().cpu().numpy())
    plt.hist(weights, bins=50, alpha=0.7)
    plt.title(title)
    plt.show()

plot_weights(resnet1, "ResNet 1 Weights")
plot_weights(resnet2, "ResNet 2 Weights")
plot_weights(lenet1, "LeNet 1 Weights")
