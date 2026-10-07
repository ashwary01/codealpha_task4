import http.server
import socketserver
import json
import urllib.parse
import os
import joblib
import pandas as pd
import numpy as np

PORT = 8000

# Load model if available
MODEL_PATH = 'models/best_sales_model.pkl'
model_pipeline = None
if os.path.exists(MODEL_PATH):
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        print(f"Loaded trained model pipeline from {MODEL_PATH}")
    except Exception as e:
        print(f"Model load error: {e}")

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        
        if parsed_url.path == '/api/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            if os.path.exists('models/pipeline_results.json'):
                with open('models/pipeline_results.json', 'r') as f:
                    data = f.read()
                self.wfile.write(data.encode('utf-8'))
            else:
                self.wfile.write(json.dumps({'status': 'running'}).encode('utf-8'))
            return

        # Serve static files as default
        return super().do_GET()

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)

        if parsed_url.path == '/api/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode('utf-8'))

            tv = float(req.get('TV', 100.0))
            radio = float(req.get('Radio', 25.0))
            news = float(req.get('Newspaper', 20.0))
            segment = req.get('Target_Segment', 'Mass Market')
            platform = req.get('Platform', 'Omnichannel Digital')

            if model_pipeline is not None:
                input_df = pd.DataFrame([{
                    'TV': tv,
                    'Radio': radio,
                    'Newspaper': news,
                    'Target_Segment': segment,
                    'Platform': platform
                }])
                pred_sales = float(model_pipeline.predict(input_df)[0])
            else:
                pred_sales = 3.95 + 0.04637 * tv + 0.21418 * radio + 0.00632 * news

            res = {
                'status': 'success',
                'predicted_sales_k': round(pred_sales, 2),
                'predicted_units': int(round(pred_sales * 1000)),
                'inputs': req
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(res).encode('utf-8'))
            return

        self.send_error(404, "Endpoint not found")

if __name__ == "__main__":
    handler = CustomHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Sales Prediction Dashboard Web Application running at http://localhost:{PORT}")
        httpd.serve_forever()
