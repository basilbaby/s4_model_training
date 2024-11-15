from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
from pathlib import Path
import webbrowser

# Create HTML file
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Basil's MNIST Training Progress</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
        .banner {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 20px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .banner h1 {
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .content {
            padding: 20px;
        }
        #chart { width: 800px; height: 400px; }
        #stats { margin: 20px 0; }
        #results { margin-top: 20px; }
        .time { color: #2196F3; font-weight: bold; }
    </style>
</head>
<body>
    <div class="banner">
        <h1>Basil's MNIST Training Progress</h1>
    </div>
    <div class="content">
        <div id="stats">
            <p>Training Device: <span id="device" style="font-weight: bold;">-</span></p>
            <p>Elapsed Time: <span id="elapsed_time" class="time">-</span></p>
            <p>Current Epoch: <span id="epoch">-</span></p>
            <p>Current Batch: <span id="batch">-</span></p>
            <p>Current Loss: <span id="loss">-</span></p>
            <p>Current Accuracy: <span id="accuracy">-</span></p>
        </div>
        <div id="chart"></div>
        <div id="results">
            <h2>Test Results</h2>
            <img id="results-img" style="display: none;" />
        </div>
    </div>

    <script>
        let losses = [];
        let accuracies = [];
        let iterations = [];

        function updateChart() {
            const trace1 = {
                y: losses,
                x: iterations,
                name: 'Loss',
                type: 'scatter'
            };

            const trace2 = {
                y: accuracies,
                x: iterations,
                name: 'Accuracy',
                type: 'scatter',
                yaxis: 'y2'
            };

            const layout = {
                title: 'Training Progress',
                yaxis: {title: 'Loss'},
                yaxis2: {
                    title: 'Accuracy',
                    overlaying: 'y',
                    side: 'right'
                }
            };

            Plotly.newPlot('chart', [trace1, trace2], layout);
        }

        function updateStats() {
            fetch('/logs/training_log.json')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('device').textContent = data.device;
                    document.getElementById('epoch').textContent = data.epoch;
                    document.getElementById('batch').textContent = data.batch;
                    document.getElementById('loss').textContent = data.loss.toFixed(4);
                    document.getElementById('accuracy').textContent = data.accuracy.toFixed(4);
                    document.getElementById('elapsed_time').textContent = 
                        data.elapsed_time || '-';

                    losses.push(data.loss);
                    accuracies.push(data.accuracy);
                    iterations.push(losses.length);

                    updateChart();
                })
                .catch(error => console.error('Error:', error));

            // Check for results image
            fetch('/logs/results.png', {method: 'HEAD'})
                .then(response => {
                    if (response.ok) {
                        document.getElementById('results-img').src = '/logs/results.png';
                        document.getElementById('results-img').style.display = 'block';
                    }
                });
        }

        // Update every second
        setInterval(updateStats, 1000);
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