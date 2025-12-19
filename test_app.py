#!/usr/bin/env python3
"""
Minimal test app for Railway
"""
from flask import Flask, jsonify
import os

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        'message': 'Railway test app is working!',
        'status': 'ok',
        'working_directory': os.getcwd(),
        'files': os.listdir('.')[:10]  # First 10 files
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)