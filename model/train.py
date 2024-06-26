import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn as nn

# 1. Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load and preprocess the dataset
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match model input
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Assuming you have your images in a folder called 'data/coins', with subfolders named '1ct', '2ct', etc.
dataset = ImageFolder(root='./data/categorised_one_coin', transform=data_transforms)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# 3. Modify the model
model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
# Adjust this number to match the number of your coin classes
model.fc = nn.Linear(num_ftrs, 8)  # We have 8 classes for 8 types of coins
model = model.to(device)

# 4. Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_model(num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 200:.4f}')
                running_loss = 0.0

    torch.save(model.state_dict(), '../model_results/model4.pth')
    print("Model saved to model4.pth")

if __name__ == "__main__":
    train_model(num_epochs=2)
