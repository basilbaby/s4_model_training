from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
from pathlib import Path
import webbrowser

html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Basil's MNIST Model Comparison</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 0;
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .banner {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 10px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            flex-shrink: 0;
        }
        .banner h1 {
            margin: 0;
            font-size: 1.8em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .content {
            padding: 15px;
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 15px;
            flex-grow: 1;
            overflow: hidden;
        }
        .left-panel {
            display: flex;
            flex-direction: column;
            gap: 15px;
            overflow-y: auto;
        }
        .right-panel {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        #chart { 
            width: 100%;
            height: 500px;
            background: white;
        }
        .model-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .model-card {
            background: #f5f5f5;
            padding: 12px;
            border-radius: 8px;
        }
        .time { color: #2196F3; font-weight: bold; }
        #results {
            background: #f5f5f5;
            padding: 12px;
            border-radius: 8px;
            overflow-y: auto;
        }
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
            padding: 20px;
            box-sizing: border-box;
        }
        
        .modal-content {
            max-width: 90%;
            max-height: 90%;
            margin: auto;
            display: block;
            position: relative;
            top: 50%;
            transform: translateY(-50%);
        }
        
        .modal-close {
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }
        
        .result-image {
            cursor: pointer;
            transition: opacity 0.3s;
        }
        
        .result-image:hover {
            opacity: 0.8;
        }
        
        /* Add zoom animation */
        .zoom {
            animation: zoom 0.3s ease-in-out;
        }
        
        @keyframes zoom {
            from {transform: scale(0)}
            to {transform: scale(1)}
        }
        .results-container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .current-run-section {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #2196F3;
        }
        .previous-runs-section {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .previous-run-item {
            margin-top: 15px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            border-left: 4px solid #9e9e9e;
        }
    </style>
</head>
<body>
    <div class="banner">
        <h1>Basil's MNIST Model Comparison</h1>
    </div>
    <div class="content">
        <div class="left-panel">
            <div class="model-stats">
                <div class="model-card">
                    <h3>Model 1</h3>
                    <p>Kernels: <span id="model1_kernels">-</span></p>
                    <p>Loss: <span id="model1_loss">-</span></p>
                    <p>Accuracy: <span id="model1_accuracy">-</span></p>
                </div>
                <div class="model-card">
                    <h3>Model 2</h3>
                    <p>Kernels: <span id="model2_kernels">-</span></p>
                    <p>Loss: <span id="model2_loss">-</span></p>
                    <p>Accuracy: <span id="model2_accuracy">-</span></p>
                </div>
            </div>
            <div class="model-card">
                <p>Run ID: <span id="current-run-id">-</span></p>
                <p>Device: <span id="device">-</span></p>
                <p>Time: <span id="elapsed_time" class="time">-</span></p>
                <p>Epoch: <span id="epoch">-</span></p>
                <p>Batch: <span id="batch">-</span></p>
            </div>
            <div class="results-container">
                <div class="current-run-section">
                    <h3 id="current-results-heading">Current Run</h3>
                    <p>Run ID: <span id="current-run-id">-</span></p>
                    <img id="current-results-img" class="result-image" style="display: none; max-width: 100%;" onclick="openModal(this)" />
                </div>
                <div class="previous-runs-section">
                    <h3>Previous Runs</h3>
                    <div id="previous-runs"></div>
                </div>
            </div>
        </div>
        <div class="right-panel">
            <div id="chart"></div>
        </div>
    </div>

    <div id="imageModal" class="modal">
        <span class="modal-close">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>

    <script>
        let model1_losses = [];
        let model2_losses = [];
        let model1_accuracies = [];
        let model2_accuracies = [];
        let iterations = [];
        let currentRunId = null;

        // Initialize the plot
        function initPlot() {
            const data = [
                {
                    y: model1_losses,
                    name: 'Model 1 Loss',
                    type: 'scatter',
                    line: {color: '#1f77b4'}
                },
                {
                    y: model2_losses,
                    name: 'Model 2 Loss',
                    type: 'scatter',
                    line: {color: '#ff7f0e'}
                },
                {
                    y: model1_accuracies,
                    name: 'Model 1 Accuracy',
                    type: 'scatter',
                    yaxis: 'y2',
                    line: {color: '#2ca02c'}
                },
                {
                    y: model2_accuracies,
                    name: 'Model 2 Accuracy',
                    type: 'scatter',
                    yaxis: 'y2',
                    line: {color: '#d62728'}
                }
            ];

            const layout = {
                title: 'Training Progress',
                showlegend: true,
                legend: {
                    orientation: 'h',
                    y: -0.2
                },
                yaxis: {title: 'Loss'},
                yaxis2: {
                    title: 'Accuracy',
                    overlaying: 'y',
                    side: 'right'
                }
            };

            Plotly.newPlot('chart', data, layout);
        }

        function updateData() {
            fetch('/logs/training_log.json')
                .then(response => response.json())
                .then(data => {
                    // Update run information
                    const runId = data.run_id;
                    if (currentRunId !== runId) {
                        // Reset arrays for new run
                        model1_losses = [];
                        model2_losses = [];
                        model1_accuracies = [];
                        model2_accuracies = [];
                        iterations = [];
                        currentRunId = runId;
                    }

                    // Update stats
                    document.getElementById('current-run-id').textContent = runId;
                    document.getElementById('device').textContent = data.device;
                    document.getElementById('epoch').textContent = data.epoch;
                    document.getElementById('batch').textContent = data.batch;
                    document.getElementById('elapsed_time').textContent = data.elapsed_time || '-';
                    
                    document.getElementById('model1_kernels').textContent = data.model1.kernels.join(', ');
                    document.getElementById('model1_loss').textContent = data.model1.loss.toFixed(4);
                    document.getElementById('model1_accuracy').textContent = data.model1.accuracy.toFixed(4);
                    
                    document.getElementById('model2_kernels').textContent = data.model2.kernels.join(', ');
                    document.getElementById('model2_loss').textContent = data.model2.loss.toFixed(4);
                    document.getElementById('model2_accuracy').textContent = data.model2.accuracy.toFixed(4);

                    // Update arrays
                    model1_losses.push(data.model1.loss);
                    model2_losses.push(data.model2.loss);
                    model1_accuracies.push(data.model1.accuracy);
                    model2_accuracies.push(data.model2.accuracy);
                    iterations.push(iterations.length);

                    // Update plot
                    Plotly.update('chart', {
                        y: [model1_losses, model2_losses, model1_accuracies, model2_accuracies],
                        x: [iterations, iterations, iterations, iterations]
                    });

                    // Handle results image
                    if (data.training_complete) {
                        document.getElementById('current-results-heading').textContent = 'Current Run Results';
                        const imgUrl = `/logs/results_${runId}.png`;
                        fetch(imgUrl, {method: 'HEAD'})
                            .then(response => {
                                if (response.ok) {
                                    document.getElementById('current-results-img').src = imgUrl;
                                    document.getElementById('current-results-img').style.display = 'block';
                                }
                            });
                    }
                })
                .catch(error => console.error('Error:', error));
        }

        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', function() {
            initPlot();
            // Update every second
            setInterval(updateData, 1000);
        });

        const modal = document.getElementById('imageModal');
        const modalImg = document.getElementById('modalImage');
        const closeBtn = document.getElementsByClassName('modal-close')[0];
        
        function openModal(img) {
            modal.style.display = 'block';
            modalImg.src = img.src;
            modalImg.classList.add('zoom');
        }
        
        closeBtn.onclick = function() {
            modal.style.display = 'none';
        }
        
        modal.onclick = function(event) {
            if (event.target === modal) {
                modal.style.display = 'none';
            }
        }
        
        function updatePreviousRuns(files) {
            const previousRunsDiv = document.getElementById('previous-runs');
            previousRunsDiv.innerHTML = '';
            
            files.forEach(runId => {
                const runDiv = document.createElement('div');
                runDiv.className = 'previous-run-item';
                runDiv.innerHTML = `
                    <p>Run ID: ${runId}</p>
                    <img src="/logs/results_${runId}.png" 
                         class="result-image" 
                         style="max-width: 100%; height: auto;" 
                         onclick="openModal(this)" />
                `;
                previousRunsDiv.appendChild(runDiv);
            });
        }
        
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape' && modal.style.display === 'block') {
                modal.style.display = 'none';
            }
        });

        function updateResults() {
            fetch('/logs/training_log.json')
                .then(response => response.json())
                .then(data => {
                    const runId = data.run_id;
                    currentRunId = runId;
                    
                    // Update current run section
                    document.getElementById('current-run-id').textContent = runId;
                    
                    if (data.training_complete) {
                        document.getElementById('current-results-heading').textContent = 'Current Run (Complete)';
                        const imgUrl = `/logs/results_${runId}.png`;
                        fetch(imgUrl, {method: 'HEAD'})
                            .then(response => {
                                if (response.ok) {
                                    document.getElementById('current-results-img').src = imgUrl;
                                    document.getElementById('current-results-img').style.display = 'block';
                                }
                            });
                    } else {
                        document.getElementById('current-results-heading').textContent = 'Current Run (In Progress)';
                        document.getElementById('current-results-img').style.display = 'none';
                    }

                    // Update previous runs
                    updatePreviousRuns();
                })
                .catch(error => console.error('Error:', error));
        }

        function updatePreviousRuns() {
            fetch('/logs')
                .then(response => response.text())
                .then(text => {
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(text, 'text/html');
                    
                    // Get all result files except current run
                    const files = Array.from(doc.querySelectorAll('a'))
                        .map(a => a.href)
                        .filter(href => href.includes('results_'))
                        .map(href => href.split('results_')[1].split('.')[0])
                        .filter(id => id !== currentRunId)
                        .sort()
                        .reverse();

                    const previousRunsDiv = document.getElementById('previous-runs');
                    
                    if (files.length === 0) {
                        previousRunsDiv.innerHTML = '<p>No previous runs available</p>';
                        return;
                    }

                    previousRunsDiv.innerHTML = '';
                    files.forEach(runId => {
                        const runDiv = document.createElement('div');
                        runDiv.className = 'previous-run-item';
                        runDiv.innerHTML = `
                            <p>Run ID: ${runId}</p>
                            <img src="/logs/results_${runId}.png" 
                                 class="result-image" 
                                 style="max-width: 100%; height: auto;" 
                                 onclick="openModal(this)" />
                        `;
                        previousRunsDiv.appendChild(runDiv);
                    });
                })
                .catch(error => console.error('Error:', error));
        }

        // Update every second
        setInterval(updateResults, 1000);
    </script>
</body>
</html>
"""

# Create logs directory and HTML file
Path('logs').mkdir(exist_ok=True)
with open('index.html', 'w') as f:
    f.write(html_content)

# Start server
class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        SimpleHTTPRequestHandler.end_headers(self)

httpd = HTTPServer(('localhost', 8000), Handler)
print("Server started at http://localhost:8000")
webbrowser.open('http://localhost:8000')
httpd.serve_forever() 