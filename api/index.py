#!/usr/bin/env python3
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import json
import subprocess
import re

# Add the parent directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Create Flask app
app = Flask(__name__)
CORS(app)

# Try to import functions from the main app
try:
    from app import (
        parse_retrieval_output, 
        parse_full_output, 
        get_japanese_name,
        translate_keywords_to_japanese
    )
    IMPORT_SUCCESS = True
except Exception as e:
    print(f"Warning: Could not import from main app: {e}")
    IMPORT_SUCCESS = False
    
    # Fallback functions
    def parse_retrieval_output(output):
        return []
    
    def parse_full_output(output, brief=""):
        return [], []
    
    def get_japanese_name(name):
        return name
    
    def translate_keywords_to_japanese(keywords):
        return keywords

@app.route('/')
def index():
    return jsonify({
        'message': 'Amazon Ads Automation API',
        'status': 'running',
        'version': '2.0',
        'import_status': 'success' if IMPORT_SUCCESS else 'fallback'
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok', 
        'message': 'Serverless function is running',
        'import_status': 'success' if IMPORT_SUCCESS else 'fallback'
    })

@app.route('/api/retrieve', methods=['POST'])
def retrieve_segments():
    try:
        data = request.json
        campaign_brief = data.get('campaign_brief', '').strip()
        
        if not campaign_brief:
            return jsonify({'error': 'Campaign brief is required'}), 400
        
        if len(campaign_brief) < 10:
            return jsonify({
                "error": f"キャンペーン説明が短すぎます。最低 10 文字必要です。"
            }), 400

        # Build command for 12.py
        cmd = [
            sys.executable, '../12.py',
            '--brief', campaign_brief,
            '--retrieval-only'
        ]

        # Run the script
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))

        if result.returncode != 0:
            return jsonify({'error': f'Script error: {result.stderr}'}), 500

        # Parse the output
        segments = parse_retrieval_output(result.stdout)

        return jsonify({
            'segments': segments,
            'total_found': len(segments)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate_segments():
    try:
        data = request.json
        campaign_brief = data.get('campaign_brief', '').strip()
        
        if not campaign_brief:
            return jsonify({'error': 'Campaign brief is required'}), 400

        # Build command for 12.py
        cmd = [
            sys.executable, '../12.py',
            '--brief', campaign_brief
        ]

        # Run the script
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))

        if result.returncode != 0:
            return jsonify({'error': f'Script error: {result.stderr}'}), 500

        # Parse both retrieval results and generated segments
        segments, generated_segments = parse_full_output(result.stdout, campaign_brief)

        return jsonify({
            'segments': segments,
            'generated_segments': generated_segments,
            'total_found': len(segments)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# This is what Vercel will use as the entry point