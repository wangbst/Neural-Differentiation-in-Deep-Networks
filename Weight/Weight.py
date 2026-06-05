import os
import pathlib
from datetime import datetime
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torch.nn.utils as utils
import math
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    Cutout(n_holes=1, length=16)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

batch_size = 128
data_path = 'data'

trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)

testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=6)

classes = 10

# Define the model
model =

# Move the model to the device
model = model.to(device)

learning_rate = 
momentum =
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=)

# Get all convolutional layers
conv_modules = [module for module in model.modules() if isinstance(module, nn.Conv2d)]


# Function to calculate NDI for a layer
def calculate_NDI():
    
    return NDI_values

# Function to calculate global sparsity
def calculate_sparsity(model):
    
    return

# Function to prune weights based on NDI
def prune_weights_NDI(model, NDI_values, target_sparsity=):
    
    # Calculate current sparsity
    current_sparsity = calculate_sparsity(model)
    
    if current_sparsity >= target_sparsity:
        return
    
    # Collect all weights and their importance scores
    all_weights = []
    importance_scores = []
    
    for i, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, nn.Conv2d):
            
            else:
    
    # Apply pruning
    for i, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, nn.Conv2d):
            else:

val_losses = []
train_losses = []
test_losses = []
best_test_acc = 0.0
target_sparsity =

# Training Loop
for epoch in range(100):  # loop over the dataset
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    # Calculate NDI for all convolutional layers
    NDI_values = {}
    for i in range(len(conv_modules)):
        NDI_values[i] = calculate_NDI()
    
    # Calculate current sparsity
    current_sparsity = calculate_sparsity(model)
    
    # Prune if we haven't reached target sparsity
    if current_sparsity < target_sparsity:
        prune_weights_NDI(model, NDI_values, target_sparsity=target_sparsity)
        print(f"Pruned to sparsity: {calculate_sparsity(model):.4f}")
    
    # Training
    model.train()
    for i, data in enumerate(trainloader, 0):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # Use those GPUs!
        inputs, labels = inputs.to(device), labels.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()

        # Calculate accuracy
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()
        train_acc = 100 * correct_train / total_train

        # Print statistics
        running_loss += loss.item()

        if i % 200 == 199:  # Print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

    # Calculate training error
    train_loss = running_loss / len(trainloader)
    train_losses.append(train_loss)

    print(f'Training Error at Epoch {epoch + 1}: {train_loss}')

    # Validation
    model.eval()
    val_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted_test = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted_test == labels).sum().item()

    # Calculate test error
    test_loss = val_loss / len(testloader)
    test_acc = 100.0 * correct_test / total_test
    test_losses.append(test_loss)

    print(f'Test Error at Epoch {epoch + 1}: {test_loss:.4f}')
    print(f'Test Accuracy at Epoch {epoch + 1}: {test_acc:.2f}%')

    # Update the best test accuracy
    if test_acc > best_test_acc:
        best_test_acc = test_acc

    val_losses.append(test_loss)

# Close all hooks
for hook in hooks.values():
    hook.remove()

print('Finished Training')
print(f'Best Test Accuracy: {best_test_acc:.2f}%')
print(f'Final Sparsity: {calculate_sparsity(model):.4f}')
