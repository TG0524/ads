#!/usr/bin/env python3
"""
Environment Setup Script for Amazon Ads Automation Platform
"""

import os
import shutil
from pathlib import Path

def setup_environment():
    """Set up the environment configuration for the project."""
    
    print("🚀 Amazon Ads Automation Platform - Environment Setup")
    print("=" * 60)
    
    # Check if .env already exists
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        overwrite = input("Do you want to overwrite it? (y/N): ").lower().strip()
        if overwrite != 'y':
            print("Setup cancelled.")
            return
    
    # Copy .env.example to .env
    if env_example.exists():
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from template")
    else:
        print("❌ .env.example not found")
        return
    
    print("\n📝 Please configure the following required variables in .env:")
    print("-" * 60)
    
    # Get OpenAI API key
    api_key = input("Enter your OpenAI API key (sk-...): ").strip()
    if api_key:
        # Update .env file
        with open(env_file, 'r') as f:
            content = f.read()
        
        content = content.replace('sk-your_openai_api_key_here', api_key)
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ OpenAI API key configured")
    
    # Model selection
    print("\n🤖 Model Configuration:")
    print("1. text-embedding-3-large (Best accuracy, higher cost)")
    print("2. text-embedding-3-small (Good accuracy, lower cost)")
    
    model_choice = input("Choose embedding model (1/2) [1]: ").strip() or "1"
    
    embedding_model = "text-embedding-3-large" if model_choice == "1" else "text-embedding-3-small"
    
    # Update embedding model in .env
    with open(env_file, 'r') as f:
        content = f.read()
    
    content = content.replace('EMBEDDING_MODEL=text-embedding-3-large', f'EMBEDDING_MODEL={embedding_model}')
    
    with open(env_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Embedding model set to: {embedding_model}")
    
    print("\n🎉 Environment setup complete!")
    print("\n📋 Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run the application: python app.py")
    print("3. Open http://localhost:5000 in your browser")
    
    print("\n⚠️  Important:")
    print("- Never commit your .env file to version control")
    print("- Keep your OpenAI API key secure")
    print("- Monitor your OpenAI usage and costs")

def check_requirements():
    """Check if all required files exist."""
    required_files = [
        "requirements.txt",
        "app.py",
        "12.py",
        "Data/docs2.jsonl",
        "Data/faiss2.index",
        "Data/japan.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files present")
    return True

if __name__ == "__main__":
    if check_requirements():
        setup_environment()
    else:
        print("\n❌ Setup failed: Missing required files")
        print("Please ensure you have all project files before running setup.")