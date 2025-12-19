#!/usr/bin/env python3
"""
Simplified version of the main app for Railway deployment
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import json

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    try:
        return send_from_directory('public', 'index.html')
    except Exception as e:
        return jsonify({
            'message': 'Amazon Ads Assistant API is running',
            'status': 'ok',
            'error': str(e),
            'endpoints': {
                'health': '/health',
                'test': '/test'
            }
        })

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'message': 'Simple Flask app is running'})

@app.route('/test')
def test():
    return jsonify({
        'message': 'Test endpoint working',
        'current_directory': os.getcwd(),
        'files_in_directory': os.listdir('.'),
        'public_exists': os.path.exists('public'),
        'public_files': os.listdir('public') if os.path.exists('public') else [],
        'data_files_exist': {
            'docs2.jsonl': os.path.exists('Data/docs2.jsonl'),
            'faiss2.index': os.path.exists('Data/faiss2.index'),
            'japan.json': os.path.exists('Data/japan.json'),
            '12.py': os.path.exists('12.py')
        }
    })

@app.route('/api/test')
def api_test():
    return jsonify({
        'message': 'API test endpoint',
        'status': 'working'
    })

# Production initialization
print("🚀 Simple Flask app initialized for Railway")
print(f"📁 Working directory: {os.getcwd()}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)