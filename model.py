import torch
import torch.nn as nn

class MNISTNet(nn.Module):
    def __init__(self, kernels=(16, 32, 64)):
        super(MNISTNet, self).__init__()
        
        self.kernels = kernels
        # Layer 1: Convolutional + ReLU + MaxPool
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, kernels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Layer 2: Convolutional + ReLU + MaxPool
        self.conv2 = nn.Sequential(
            nn.Conv2d(kernels[0], kernels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Layer 3: Convolutional + ReLU
        self.conv3 = nn.Sequential(
            nn.Conv2d(kernels[1], kernels[2], kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Layer 4: Fully Connected
        self.fc = nn.Linear(7 * 7 * kernels[2], 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())