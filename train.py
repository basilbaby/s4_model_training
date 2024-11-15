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
import platform
import time
from datetime import datetime, timedelta
import argparse

# Add this after imports, before device selection
def parse_args():
    parser = argparse.ArgumentParser(description='MNIST Training with Device Selection')
    parser.add_argument('--force-cpu', action='store_true', 
                       help='Force CPU usage even if GPU is available')
    return parser.parse_args()

# Update device selection logic
args = parse_args()

if args.force_cpu:
    device = torch.device('cpu')
    backend = 'CPU (Forced)'
    torch.set_num_threads(os.cpu_count())
    print(f"Forcing CPU usage with {os.cpu_count()} threads")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    backend = 'NVIDIA GPU'
    torch.backends.cudnn.benchmark = True
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
    backend = 'Apple Silicon GPU'
    if platform.processor() == 'arm':
        import subprocess
        cmd = "sysctl -n hw.perflevel0.gpu_count"
        try:
            gpu_cores = int(subprocess.check_output(cmd.split()).decode().strip())
            torch.set_num_threads(gpu_cores)
            print(f"Utilizing {gpu_cores} GPU cores on Apple Silicon")
        except:
            print("Could not determine GPU core count, using default settings")
else:
    device = torch.device('cpu')
    backend = 'CPU'
    torch.set_num_threads(os.cpu_count())

print(f"Using device: {backend} ({device})")

# Enable automatic mixed precision for better performance
if backend == 'Apple Silicon GPU':
    # Configure for mixed precision on MPS
    torch.set_float32_matmul_precision('high')

# Data preparation with optimized loading
def create_data_loaders(batch_size=64, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    global train_dataset, test_dataset  # Make datasets global
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # Optimize DataLoader for the specific backend
    if backend == 'Apple Silicon GPU':
        num_workers = min(4, os.cpu_count())
        persistent_workers = True
    else:
        persistent_workers = False

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True
    )

    return train_loader, test_loader

# Create optimized data loaders
train_loader, test_loader = create_data_loaders()

# Initialize model, loss, and optimizer
model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training logs directory
Path('logs').mkdir(exist_ok=True)

def save_training_log(epoch, batch_idx, loss, accuracy, elapsed_time=None):
    log = {
        'epoch': epoch,
        'batch': batch_idx,
        'loss': float(loss),
        'accuracy': float(accuracy),
        'device': backend,
        'cores': torch.get_num_threads() if backend == 'Apple Silicon GPU' else None,
        'elapsed_time': str(timedelta(seconds=int(elapsed_time))) if elapsed_time else None
    }
    with open('logs/training_log.json', 'w') as f:
        json.dump(log, f)

def train(epochs=10):
    print(f"Training on: {backend}")
    start_time = time.time()
    
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
                elapsed_time = time.time() - start_time
                save_training_log(epoch, batch_idx, loss.item(), accuracy, elapsed_time)
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}, '
                      f'Accuracy: {accuracy:.4f}, Time: {timedelta(seconds=int(elapsed_time))}')
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in: {timedelta(seconds=int(total_time))}")
    
    # Save final training time
    with open('logs/training_time.txt', 'w') as f:
        f.write(f"Total training time: {timedelta(seconds=int(total_time))}")

def evaluate_random_samples():
    model.eval()
    # Get 10 random test samples
    test_data = []
    test_labels = []
    
    # Now test_dataset is accessible
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