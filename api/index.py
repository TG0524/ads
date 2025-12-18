from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({
        'message': 'Amazon Ads Automation API',
        'status': 'running',
        'version': '2.0'
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok', 
        'message': 'Serverless function is running'
    })

@app.route('/api/test', methods=['GET', 'POST'])
def test_endpoint():
    return jsonify({
        'message': 'Test endpoint working',
        'method': request.method
    })