from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
from pathlib import Path
import webbrowser

# Create HTML file
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
            height: calc(100vh - 60px);
            overflow: hidden;
        }
        .left-panel {
            display: flex;
            flex-direction: column;
            gap: 15px;
            overflow-y: auto;
            padding-right: 10px;
        }
        .right-panel {
            display: flex;
            flex-direction: column;
            gap: 15px;
            overflow-y: auto;
        }
        #chart { 
            width: 100%;
            height: 60vh;
            background: white;
            flex-shrink: 0;
        }
        .model-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            flex-shrink: 0;
        }
        .model-card {
            background: #f5f5f5;
            padding: 12px;
            border-radius: 8px;
            font-size: 0.9em;
        }
        .model-card h3 {
            margin: 0 0 8px 0;
        }
        .model-card p {
            margin: 5px 0;
        }
        .time { color: #2196F3; font-weight: bold; }
        .training-info {
            background: #f5f5f5;
            padding: 12px;
            border-radius: 8px;
            font-size: 0.9em;
            flex-shrink: 0;
        }
        .training-info p {
            margin: 5px 0;
        }
        #results {
            overflow-y: auto;
            background: #f5f5f5;
            padding: 12px;
            border-radius: 8px;
        }
        .current-run, .previous-run {
            margin-top: 15px;
        }
        .current-run h4, .previous-run h4 {
            margin: 0 0 10px 0;
            color: #2196F3;
        }
        .previous-run-item {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #ddd;
        }
        #results img {
            max-width: 100%;
            height: auto;
            margin: 10px 0;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
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
            <div class="training-info">
                <p>Device: <span id="device" style="font-weight: bold;">-</span></p>
                <p>Time: <span id="elapsed_time" class="time">-</span></p>
                <p>Epoch: <span id="epoch">-</span></p>
                <p>Batch: <span id="batch">-</span></p>
            </div>
            <div id="results">
                <h3>Test Results</h3>
                <div class="current-run">
                    <h4>Current Run: <span id="current-run-id">-</span></h4>
                    <p class="timestamp">Started: <span id="current-run-time">-</span></p>
                    <img id="current-results-img" style="display: none;" />
                </div>
                <div class="previous-run">
                    <h4>Previous Run Results</h4>
                    <div id="previous-runs"></div>
                </div>
            </div>
        </div>
        <div class="right-panel">
            <div id="chart"></div>
        </div>
    </div>

    <script>
        let model1_losses = [];
        let model2_losses = [];
        let model1_accuracies = [];
        let model2_accuracies = [];
        let iterations = [];

        function updateChart() {
            const trace1 = {
                y: model1_losses,
                x: iterations,
                name: 'Model 1 Loss',
                type: 'scatter',
                line: {color: '#1f77b4'}
            };

            const trace2 = {
                y: model2_losses,
                x: iterations,
                name: 'Model 2 Loss',
                type: 'scatter',
                line: {color: '#ff7f0e'}
            };

            const trace3 = {
                y: model1_accuracies,
                x: iterations,
                name: 'Model 1 Accuracy',
                type: 'scatter',
                yaxis: 'y2',
                line: {color: '#2ca02c'}
            };

            const trace4 = {
                y: model2_accuracies,
                x: iterations,
                name: 'Model 2 Accuracy',
                type: 'scatter',
                yaxis: 'y2',
                line: {color: '#d62728'}
            };

            const layout = {
                margin: { l: 50, r: 50, t: 30, b: 30 },
                showlegend: true,
                legend: {
                    orientation: 'h',
                    y: -0.2
                },
                yaxis: {
                    title: 'Loss',
                    titlefont: {size: 11},
                    tickfont: {size: 10}
                },
                yaxis2: {
                    title: 'Accuracy',
                    titlefont: {size: 11},
                    tickfont: {size: 10},
                    overlaying: 'y',
                    side: 'right'
                },
                xaxis: {
                    tickfont: {size: 10}
                }
            };

            Plotly.newPlot('chart', [trace1, trace2, trace3, trace4], layout);
        }

        function updateStats() {
            fetch('/logs/training_log.json')
                .then(response => response.json())
                .then(data => {
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

                    model1_losses.push(data.model1.loss);
                    model2_losses.push(data.model2.loss);
                    model1_accuracies.push(data.model1.accuracy);
                    model2_accuracies.push(data.model2.accuracy);
                    iterations.push(model1_losses.length);

                    updateChart();
                })
                .catch(error => console.error('Error:', error));

            fetch('/logs/results.png', {method: 'HEAD'})
                .then(response => {
                    if (response.ok) {
                        document.getElementById('results-img').src = '/logs/results.png';
                        document.getElementById('results-img').style.display = 'block';
                    }
                });
        }

        function updateResults() {
            // Get current run results
            fetch('/logs/training_log.json')
                .then(response => response.json())
                .then(data => {
                    const runId = data.run_id;
                    document.getElementById('current-run-id').textContent = runId;
                    document.getElementById('current-run-time').textContent = data.timestamp;
                    
                    // Check for current run's results image
                    fetch(`/logs/results_${runId}.png`, {method: 'HEAD'})
                        .then(response => {
                            if (response.ok) {
                                document.getElementById('current-results-img').src = `/logs/results_${runId}.png`;
                                document.getElementById('current-results-img').style.display = 'block';
                            }
                        });
                })
                .catch(error => console.error('Error:', error));

            // List previous runs
            fetch('/logs')
                .then(response => response.text())
                .then(text => {
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(text, 'text/html');
                    const files = Array.from(doc.querySelectorAll('a'))
                        .map(a => a.href)
                        .filter(href => href.includes('results_'))
                        .map(href => href.split('results_')[1].split('.')[0]);

                    const previousRunsDiv = document.getElementById('previous-runs');
                    previousRunsDiv.innerHTML = '';

                    files.forEach(runId => {
                        const runDiv = document.createElement('div');
                        runDiv.className = 'previous-run-item';
                        runDiv.innerHTML = `
                            <p>Run ID: ${runId}</p>
                            <img src="/logs/results_${runId}.png" style="max-width: 100%; height: auto;" />
                        `;
                        previousRunsDiv.appendChild(runDiv);
                    });
                })
                .catch(error => console.error('Error:', error));
        }

        // Add updateResults to the interval
        setInterval(() => {
            updateStats();
            updateResults();
        }, 1000);
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