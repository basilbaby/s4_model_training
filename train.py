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
from uuid import uuid4

# Add this after imports, before device selection
def parse_args():
    parser = argparse.ArgumentParser(description='MNIST Training with Model Comparison')
    parser.add_argument('--force-cpu', action='store_true', 
                       help='Force CPU usage even if GPU is available')
    parser.add_argument('--model1', nargs=3, type=int, default=[16, 32, 64],
                       help='Kernel sizes for first model (3 integers)')
    parser.add_argument('--model2', nargs=3, type=int, default=[8, 8, 8],
                       help='Kernel sizes for second model (3 integers)')
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

# Training logs directory
Path('logs').mkdir(exist_ok=True)

# Add after imports
run_id = None  # Global variable for run_id

def save_training_log(epoch, batch_idx, model1_loss, model1_acc, model2_loss, model2_acc, elapsed_time=None, is_complete=False):
    global run_id
    log = {
        'run_id': run_id,
        'epoch': epoch,
        'batch': batch_idx,
        'model1': {
            'loss': float(model1_loss),
            'accuracy': float(model1_acc),
            'kernels': args.model1
        },
        'model2': {
            'loss': float(model2_loss),
            'accuracy': float(model2_acc),
            'kernels': args.model2
        },
        'device': backend,
        'cores': torch.get_num_threads() if backend == 'Apple Silicon GPU' else None,
        'elapsed_time': str(timedelta(seconds=int(elapsed_time))) if elapsed_time else None,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'training_complete': is_complete
    }
    with open('logs/training_log.json', 'w') as f:
        json.dump(log, f)

def train_models(epochs=10):
    global run_id  # Access the global run_id
    run_id = str(uuid4())[:8]  # Set the global run_id
    print(f"Starting training run: {run_id}")
    start_time = time.time()
    
    # Initialize both models
    model1 = MNISTNet(args.model1).to(device)
    model2 = MNISTNet(args.model2).to(device)
    
    # Print model parameters
    print(f"\nModel 1 ({args.model1}): {model1.get_parameter_count():,} parameters")
    print(f"Model 2 ({args.model2}): {model2.get_parameter_count():,} parameters\n")
    
    optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model1.train()
        model2.train()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Train model 1
            optimizer1.zero_grad()
            output1 = model1(data)
            loss1 = criterion(output1, target)
            loss1.backward()
            optimizer1.step()
            
            # Train model 2
            optimizer2.zero_grad()
            output2 = model2(data)
            loss2 = criterion(output2, target)
            loss2.backward()
            optimizer2.step()
            
            # Calculate accuracies
            pred1 = output1.argmax(dim=1, keepdim=True)
            pred2 = output2.argmax(dim=1, keepdim=True)
            acc1 = pred1.eq(target.view_as(pred1)).sum().item() / len(data)
            acc2 = pred2.eq(target.view_as(pred2)).sum().item() / len(data)
            
            if batch_idx % 10 == 0:
                elapsed_time = time.time() - start_time
                save_training_log(epoch, batch_idx, loss1.item(), acc1, 
                                loss2.item(), acc2, elapsed_time)
                print(f'Epoch: {epoch}, Batch: {batch_idx}')
                print(f'Model1 - Loss: {loss1.item():.4f}, Accuracy: {acc1:.4f}')
                print(f'Model2 - Loss: {loss2.item():.4f}, Accuracy: {acc2:.4f}')
                print(f'Time: {timedelta(seconds=int(elapsed_time))}\n')
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in: {timedelta(seconds=int(total_time))}")
    
    # Save final log with complete flag
    save_training_log(epochs-1, len(train_loader)-1, loss1.item(), acc1, 
                     loss2.item(), acc2, total_time, is_complete=True)
    
    return model1, model2

def evaluate_random_samples(model1, model2):
    global run_id  # Access the global run_id
    model1.eval()
    model2.eval()
    test_data = []
    test_labels = []
    
    random_indices = random.sample(range(len(test_dataset)), 10)
    for idx in random_indices:
        image, label = test_dataset[idx]
        test_data.append(image)
        test_labels.append(label)
    
    test_data = torch.stack(test_data).to(device)
    
    with torch.no_grad():
        outputs1 = model1(test_data)
        outputs2 = model2(test_data)
        pred1 = outputs1.argmax(dim=1).cpu().numpy()
        pred2 = outputs2.argmax(dim=1).cpu().numpy()
    
    # Plot results
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for idx in range(10):
        axes[idx].imshow(test_data[idx][0].cpu().numpy(), cmap='gray')
        axes[idx].axis('off')
        axes[idx].set_title(f'M1: {pred1[idx]} | M2: {pred2[idx]}\nTrue: {test_labels[idx]}')
    
    plt.suptitle(f'Test Results (Run ID: {run_id})')
    plt.savefig(f'logs/results_{run_id}.png')
    plt.close()

if __name__ == '__main__':
    args = parse_args()
    model1, model2 = train_models()
    evaluate_random_samples(model1, model2)
    torch.save({
        'run_id': run_id,
        'model1_state': model1.state_dict(),
        'model2_state': model2.state_dict(),
        'model1_kernels': args.model1,
        'model2_kernels': args.model2,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }, f'mnist_cnn_comparison_{run_id}.pth') 