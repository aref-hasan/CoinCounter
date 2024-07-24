#############################Libraries#################################
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn as nn


######################################################################
######################################################################


# setup device: use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load and preprocess the dataset
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # resize images to 224x224 pixels
    transforms.ToTensor(),          # convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize images
])

# assuming you have your images in a folder called 'data/coins', with subfolders named '1ct', '2ct', etc.
dataset = ImageFolder(root='./data/categorised_one_coin', transform=data_transforms)
train_size = int(0.8 * len(dataset))  # 80% of the data for training
test_size = len(dataset) - train_size  # the rest 20% for testing
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # data loader for training
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)   # data loader for testing

# modify the model
model = torchvision.models.resnet18(pretrained=True)  # load pre-trained ResNet18 model
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 8)  # 8 classes for 8 types of coins
model = model.to(device)  # move model to the appropriate device (CPU or GPU)

# training
criterion = nn.CrossEntropyLoss()  # loss function
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # stochastic gradient descent optimizer

def train_model(num_epochs):
    model.train()  # set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # move inputs and labels to the appropriate device

            optimizer.zero_grad()  # zero the parameter gradients
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # compute the loss
            loss.backward()  # backpropagate the loss
            optimizer.step()  # update the model parameters

            running_loss += loss.item()  # accumulate the loss
            if i % 200 == 199:  # print every 200 mini-batches
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 200:.4f}')
                running_loss = 0.0

    torch.save(model.state_dict(), '../model_results/recognition_model_final.pth')  # save the trained model
    print("Model saved to recognition_model_final.pth")

if __name__ == "__main__":
    train_model(num_epochs=2)  # train the model for 2 epochs
