#############################Libraries#################################
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torch.nn as nn
######################################################################
######################################################################


# setup device: use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define data transformations: convert images to tensors and normalize them
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# load dataset from the specified folder and apply the transformations
dataset = ImageFolder(root='../data/categorised_one_coin', transform=data_transforms)
train_size = int(0.8 * len(dataset))  # 80% of the data for training
test_size = len(dataset) - train_size  # the rest 20% for testing
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# create data loader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# load pre-trained ResNet18 model and modify the final layer to match the number of classes
model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 8)  # assume there are 8 classes
model = model.to(device)  # move model to the appropriate device (CPU or GPU)

# load the trained model's state dictionary
model.load_state_dict(torch.load('../models/recognition_model_final.pth'))

# define loss function
criterion = nn.CrossEntropyLoss()

# function to test the model on the test dataset
def test_model():
    model.eval()  # set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # no need to compute gradients during testing
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # move inputs and labels to the appropriate device
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # get the predicted class with the highest score
            total += labels.size(0)  # total number of samples
            correct += (predicted == labels).sum().item()  # count correct predictions

    # print the accuracy of the model on the test set
    print(f'Accuracy on the test set: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    test_model()