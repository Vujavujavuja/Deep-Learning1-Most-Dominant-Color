import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm


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
        image = Image.open(img_path).convert("RGBA").convert("RGB")
        target = eval(row["Dominant Color"])

        if self.transform:
            image = self.transform(image)

        target = torch.tensor(target, dtype=torch.float32) / 255.0
        return image, target


if __name__ == "__main__":
    csv_path = "C:\\Users\\nvuji\\OneDrive\\Documents\\GitHub\\Deep-Learning1-Most-Dominant-Color\\preprocessing\\cleaned_dataset.csv"
    image_dir = "C:\\Users\\nvuji\\OneDrive\\Documents\\GitHub\\Deep-Learning1-Most-Dominant-Color\\preprocessing\\images"

    batch_size = 64
    learning_rate = 0.001
    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = ColorDataset(csv_path, image_dir, transform=transform)
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        with tqdm(train_loader, desc=f"Epoch {epoch} [Training]") as pbar:
            for data, target in pbar:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix({"Batch Loss": loss.item()})

        train_loss /= len(train_loader)

        model.eval()
        val_loss, correct = 0, 0
        total_samples = 0
        with tqdm(val_loader, desc=f"Epoch {epoch} [Validation]") as pbar:
            with torch.no_grad():
                for data, target in pbar:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()
                    diff = torch.abs(output - target)
                    correct += (diff < 0.1).all(dim=1).sum().item()
                    total_samples += target.size(0)
                    pbar.set_postfix({"Batch Loss": loss.item()})

        val_loss /= len(val_loader)
        accuracy = 100.0 * correct / total_samples
        print(f"Epoch {epoch}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), "resnet_v1.pth")
    print("Model saved as resnet_v1.pth")
