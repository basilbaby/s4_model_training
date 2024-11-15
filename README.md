# MNIST CNN Training Visualizer

A real-time visualization tool for training a 4-layer Convolutional Neural Network (CNN) on the MNIST dataset. The project includes live training metrics visualization through a web interface and CUDA support for GPU acceleration.

## Features

- 4-layer CNN architecture optimized for MNIST digit recognition
- Real-time training visualization with live-updating graphs
- CUDA support for GPU acceleration
- Interactive web interface showing:
  - Live loss and accuracy curves
  - Current training statistics
  - Final results on random test samples
- Automatic model evaluation on completion

## Architecture

The CNN consists of:
1. Conv2d(1→16) + ReLU + MaxPool2d
2. Conv2d(16→32) + ReLU + MaxPool2d
3. Conv2d(32→64) + ReLU
4. Fully Connected Layer(7*7*64→10)

## Project Structure

```
├── model.py          # CNN model architecture
├── train.py         # Training script with CUDA support
├── server.py        # Visualization server
├── index.html       # Web interface for visualization
├── requirements.txt # Project dependencies
├── README.md        # Project documentation
├── HowTo.md        # Step-by-step instructions
└── logs/           # Directory for training logs and results
    ├── training_log.json  # Real-time training metrics
    └── results.png        # Final evaluation results
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional but recommended)
- Required packages listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mnist-cnn-visualizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Training and Visualization

### Step 1: Start the Visualization Server
```bash
python server.py
```
The server will:
- Create necessary directories
- Start local server on port 8000
- Open your default browser to http://localhost:8000
- Keep running throughout the training process

### Step 2: Start the Training
In a new terminal:
```bash
python train.py
```
This will:
- Download MNIST dataset (first run only)
- Initialize CNN model
- Use GPU if available (CUDA)
- Show progress in terminal
- Update web visualization
- Save results after completion

## Training Configuration

Key parameters in `train.py`:
```python
epochs = 10              # Total training epochs
batch_size = 64         # Samples per batch
learning_rate = 0.001   # Adam optimizer learning rate
```

## Monitoring Training

### Web Interface
Navigate to http://localhost:8000 to see:
- Real-time loss curve
- Accuracy progression
- Current training stats
- Final test results (after completion)

### Terminal Output
```
Epoch: 0, Batch: 0, Loss: 2.3024, Accuracy: 0.1250
Epoch: 0, Batch: 10, Loss: 2.1234, Accuracy: 0.2812
...
```

## Output Files

After training:
- `mnist_cnn.pth`: Trained model weights
- `logs/results.png`: Test results visualization
- `logs/training_log.json`: Training history

## Troubleshooting

1. Server Issues:
   - Check port 8000 availability
   - Verify Python environment
   - Check file permissions

2. Training Issues:
   - Confirm CUDA installation (for GPU)
   - Verify MNIST download
   - Check disk space for logs

3. Visualization Issues:
   - Refresh browser
   - Check both processes running
   - Clear browser cache

## Performance Tips

1. GPU Training:
   - Use CUDA-capable GPU
   - Close other GPU applications
   - Monitor GPU memory usage

2. CPU Training:
   - Reduce batch size
   - Close resource-heavy applications
   - Expect longer training time

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MNIST Dataset: [LeCun et al.](http://yann.lecun.com/exdb/mnist/)
- PyTorch Framework
- Plotly.js for visualization