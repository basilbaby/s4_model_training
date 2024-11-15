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

# Add these constants at the top of the file after imports
DEFAULT_MODELS = {
    'standard': [16, 32, 64],
    'light': [8, 8, 8],
    'heavy': [32, 64, 128],
    'pyramid': [64, 32, 16],
    'uniform': [32, 32, 32]
}

# Add these as global variables at the start of the file, after imports
run_id = None
args = None
device = None
backend = None
train_loader = None
test_loader = None
train_dataset = None
test_dataset = None
start_time = None

def parse_args():
    parser = argparse.ArgumentParser(description='MNIST Training with Model Comparison')
    parser.add_argument('--force-cpu', action='store_true', 
                       help='Force CPU usage even if GPU is available')
    
    # Update model arguments to handle both preset names and kernel sizes
    parser.add_argument('--model1', nargs='*', default=['standard'],
                       help=f'Model 1 configuration. Either 3 integers for kernel sizes or a preset name {list(DEFAULT_MODELS.keys())}')
    parser.add_argument('--model2', nargs='*', default=['light'],
                       help=f'Model 2 configuration. Either 3 integers for kernel sizes or a preset name {list(DEFAULT_MODELS.keys())}')
    
    # Add other training parameters
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs (default: 10)')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam',
                       help='Optimizer to use (adam or sgd)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum for SGD optimizer (default: 0.9)')
    
    args = parser.parse_args()
    
    # Process model configurations immediately
    args.model1 = get_model_config(args.model1)
    args.model2 = get_model_config(args.model2)
    
    return args

def get_model_config(model_arg):
    """Convert model argument to kernel configuration"""
    # Handle empty or None case
    if not model_arg:
        return DEFAULT_MODELS['standard']
    
    # Handle preset name
    if len(model_arg) == 1 and model_arg[0] in DEFAULT_MODELS:
        return DEFAULT_MODELS[model_arg[0]]
    
    # Handle kernel sizes
    try:
        kernels = [int(k) for k in model_arg]
        if len(kernels) != 3:
            print(f"Warning: Expected 3 kernel sizes, got {len(kernels)}. Using default configuration.")
            return DEFAULT_MODELS['standard']
        return kernels
    except ValueError:
        print(f"Warning: Invalid kernel sizes. Using default configuration.")
        return DEFAULT_MODELS['standard']

# Update device selection logic
def setup_device(args):
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
    return device, backend

# Add this function after setup_device
def create_data_loaders(batch_size, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # Optimize DataLoader for the specific backend
    if backend == 'Apple Silicon GPU':
        # Use appropriate number of workers for MPS
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

    return train_loader, test_loader, train_dataset, test_dataset

# Also add the train_models function
def train_models(epochs=10):
    global run_id
    run_id = str(uuid4())[:8]
    print(f"Starting training run: {run_id}")
    start_time = time.time()
    
    # Initialize both models
    model1 = MNISTNet(args.model1).to(device)
    model2 = MNISTNet(args.model2).to(device)
    
    print(f"\nModel 1 ({args.model1}): {model1.get_parameter_count():,} parameters")
    print(f"Model 2 ({args.model2}): {model2.get_parameter_count():,} parameters")
    
    # Initialize optimizers based on choice
    if args.optimizer == 'adam':
        optimizer1 = optim.Adam(model1.parameters(), lr=args.learning_rate)
        optimizer2 = optim.Adam(model2.parameters(), lr=args.learning_rate)
    else:  # SGD
        optimizer1 = optim.SGD(model1.parameters(), lr=args.learning_rate, momentum=args.momentum)
        optimizer2 = optim.SGD(model2.parameters(), lr=args.learning_rate, momentum=args.momentum)
    
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

# Add this function after the imports and before other functions
def save_training_log(epoch, batch_idx, model1_loss, model1_acc, model2_loss, model2_acc, elapsed_time=None, is_complete=False):
    global run_id, args, start_time
    
    # Print debug information
    print("\nSaving training log with parameters:")
    print(f"Optimizer: {args.optimizer}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    if args.optimizer == 'sgd':
        print(f"Momentum: {args.momentum}")
    
    log = {
        'run_id': run_id,
        'epoch': epoch,
        'batch': batch_idx,
        'model1': {
            'loss': float(model1_loss),
            'accuracy': float(model1_acc),
            'kernels': args.model1,
            'type': 'custom' if isinstance(args.model1[0], int) else args.model1[0]
        },
        'model2': {
            'loss': float(model2_loss),
            'accuracy': float(model2_acc),
            'kernels': args.model2,
            'type': 'custom' if isinstance(args.model2[0], int) else args.model2[0]
        },
        'training_params': {
            'optimizer': str(args.optimizer).lower(),  # Ensure it's a string and lowercase
            'learning_rate': float(args.learning_rate),
            'batch_size': int(args.batch_size),
            'epochs': int(args.epochs),
            'momentum': float(args.momentum) if args.optimizer == 'sgd' else None
        },
        'device': backend,
        'cores': torch.get_num_threads() if backend == 'Apple Silicon GPU' else None,
        'elapsed_time': str(timedelta(seconds=int(elapsed_time))) if elapsed_time else None,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'training_complete': is_complete
    }
    
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    # Save the log with pretty printing for debugging
    with open('logs/training_log.json', 'w') as f:
        json.dump(log, f, indent=4)
    
    # Print confirmation
    print(f"Training log saved with parameters: {log['training_params']}")

def evaluate_random_samples(model1, model2):
    global run_id, args  # Add args here to access training parameters
    model1.eval()
    model2.eval()
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
        outputs1 = model1(test_data)
        outputs2 = model2(test_data)
        predictions1 = outputs1.argmax(dim=1).cpu().numpy()
        predictions2 = outputs2.argmax(dim=1).cpu().numpy()
    
    # Plot results
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for idx in range(10):
        axes[idx].imshow(test_data[idx][0].cpu().numpy(), cmap='gray')
        axes[idx].axis('off')
        axes[idx].set_title(f'M1: {predictions1[idx]} | M2: {predictions2[idx]}\nTrue: {test_labels[idx]}')
    
    plt.savefig(f'logs/results_{run_id}.png')
    plt.close()

    # Save final training parameters again to ensure they're current
    save_training_log(
        args.epochs-1, 
        len(train_loader)-1, 
        0.0,  # placeholder for final loss
        0.0,  # placeholder for final accuracy
        0.0,  # placeholder for final loss
        0.0,  # placeholder for final accuracy
        time.time() - start_time, 
        is_complete=True
    )

if __name__ == '__main__':
    # Parse arguments once at the start
    args = parse_args()
    start_time = time.time()
    
    print("\nTraining Configuration:")
    print(f"Optimizer: {args.optimizer.upper()}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    if args.optimizer == 'sgd':
        print(f"Momentum: {args.momentum}")
    print()
    
    # Setup device
    device, backend = setup_device(args)
    
    # Create data loaders with specified batch size
    train_loader, test_loader, train_dataset, test_dataset = create_data_loaders(args.batch_size)
    
    # Save initial configuration
    save_training_log(0, 0, 0.0, 0.0, 0.0, 0.0, 0, False)
    
    # Train for specified number of epochs
    model1, model2 = train_models(args.epochs)
    evaluate_random_samples(model1, model2)
    
    # Save model and parameters
    torch.save({
        'run_id': run_id,
        'model1_state': model1.state_dict(),
        'model2_state': model2.state_dict(),
        'model1_kernels': args.model1,
        'model2_kernels': args.model2,
        'optimizer': args.optimizer,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'momentum': args.momentum if args.optimizer == 'sgd' else None,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }, f'mnist_cnn_comparison_{run_id}.pth') 