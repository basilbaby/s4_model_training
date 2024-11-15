import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTNet
import json
import os
from pathlib import Path
import random
import matplotlib.pyplot as plt
import numpy as np

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)

# Initialize model, loss, and optimizer
model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training logs directory
Path('logs').mkdir(exist_ok=True)

def save_training_log(epoch, batch_idx, loss, accuracy):
    log = {
        'epoch': epoch,
        'batch': batch_idx,
        'loss': float(loss),
        'accuracy': float(accuracy)
    }
    with open('logs/training_log.json', 'w') as f:
        json.dump(log, f)

def train(epochs=10):
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = correct / len(data)
            
            if batch_idx % 10 == 0:
                save_training_log(epoch, batch_idx, loss.item(), accuracy)
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

def evaluate_random_samples():
    model.eval()
    # Get 10 random test samples
    test_data = []
    test_labels = []
    
    random_indices = random.sample(range(len(test_dataset)), 10)
    for idx in random_indices:
        image, label = test_dataset[idx]
        test_data.append(image)
        test_labels.append(label)
    
    test_data = torch.stack(test_data).to(device)
    
    with torch.no_grad():
        outputs = model(test_data)
        predictions = outputs.argmax(dim=1).cpu().numpy()
    
    # Plot results
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for idx in range(10):
        axes[idx].imshow(test_data[idx][0].cpu().numpy(), cmap='gray')
        axes[idx].axis('off')
        axes[idx].set_title(f'Pred: {predictions[idx]}\nTrue: {test_labels[idx]}')
    
    plt.savefig('logs/results.png')
    plt.close()

if __name__ == '__main__':
    train()
    evaluate_random_samples()
    torch.save(model.state_dict(), 'mnist_cnn.pth') 