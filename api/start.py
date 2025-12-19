#!/usr/bin/env python3
"""
Railway startup script - handles working directory and runs the Flask app
"""
import os
import sys

# Print debug info
print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")

# Check if app.py exists in current directory
if os.path.exists('app.py'):
    print("✅ Found app.py in current directory")
    print("🚀 Starting Flask app...")
    exec(open('app.py').read())
else:
    print("❌ Cannot find app.py in current directory")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files: {os.listdir('.')}")
    sys.exit(1)