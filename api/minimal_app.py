#!/usr/bin/env python3
"""
Minimal working Flask app for Railway
"""
from flask import Flask, jsonify
import os

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        'message': 'Amazon Ads Assistant - Minimal Version',
        'status': 'working',
        'directory': os.getcwd(),
        'files': os.listdir('.')[:10]
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

@app.route('/test')
def test():
    return jsonify({
        'message': 'Test endpoint working',
        'files_exist': {
            '12.py': os.path.exists('12.py'),
            'Data/docs2.jsonl': os.path.exists('Data/docs2.jsonl'),
            'Data/faiss2.index': os.path.exists('Data/faiss2.index'),
            'public/index.html': os.path.exists('public/index.html')
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting minimal app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)