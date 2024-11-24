import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd


class ColorDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = f"{self.image_dir}/{row['Image Name']}"
        image = Image.open(img_path).convert("RGB")
        target = eval(row["Dominant Color"])

        if self.transform:
            image = self.transform(image)

        target = torch.tensor(target, dtype=torch.float32)
        return image, target


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)

        self.avg_pool = nn.AvgPool2d(2, stride=2)

        self.fc1 = nn.Linear(16 * 56 * 56, 120)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.avg_pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.avg_pool(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        return self.fc3(x)


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Validation
        val_loss = validate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}")


def validate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(loader)


def test_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader.dataset)
    print(f"Test Set Mean Squared Error: {avg_loss:.4f}")
    return avg_loss


if __name__ == "__main__":
    csv_path = "C:\\Users\\nvuji\\OneDrive\\Documents\\GitHub\\Deep-Learning1-Most-Dominant-Color\\preprocessing\\cleaned_dataset.csv"
    image_dir = "C:\\Users\\nvuji\\OneDrive\\Documents\\GitHub\\Deep-Learning1-Most-Dominant-Color\\preprocessing\\images"

    batch_size = 32
    learning_rate = 0.001
    epochs = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load Dataset
    dataset = ColorDataset(csv_path, image_dir, transform=transform)
    train_size = int(0.6 * len(dataset))  # 60
    val_size = int(0.2 * len(dataset))    # 20
    test_size = len(dataset) - train_size - val_size  # 20
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LeNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)


    train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device)

    torch.save(model.state_dict(), "lenet_v1.pth")
    print("Model saved as lenet_v1.pth")

    test_model(model, test_loader, criterion, device)
