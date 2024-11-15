from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
from pathlib import Path
import webbrowser

# Create HTML file
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>MNIST Training Progress</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        #chart { width: 800px; height: 400px; }
        #stats { margin: 20px 0; }
        #results { margin-top: 20px; }
    </style>
</head>
<body>
    <h1>MNIST Training Progress</h1>
    <div id="stats">
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
                    document.getElementById('epoch').textContent = data.epoch;
                    document.getElementById('batch').textContent = data.batch;
                    document.getElementById('loss').textContent = data.loss.toFixed(4);
                    document.getElementById('accuracy').textContent = data.accuracy.toFixed(4);

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