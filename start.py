#!/usr/bin/env python3
"""
Railway startup script - handles working directory and runs the Flask app
"""
import os
import sys
import subprocess

# Print debug info
print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")

# Check if we're in the root directory or api directory
if os.path.exists('api/app.py'):
    # We're in the root directory, run the api/app.py
    print("Running from root directory")
    os.chdir('api')
    print(f"Changed to: {os.getcwd()}")
    exec(open('app.py').read())
elif os.path.exists('app.py'):
    # We're already in the api directory
    print("Running from api directory")
    exec(open('app.py').read())
else:
    print("❌ Cannot find app.py in current directory or api/ subdirectory")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files: {os.listdir('.')}")
    sys.exit(1)