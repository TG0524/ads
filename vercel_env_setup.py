#!/usr/bin/env python3
"""
Vercel Environment Variables Setup Helper
Generates commands to set environment variables in Vercel
"""

import os
from pathlib import Path

def generate_vercel_commands():
    """Generate Vercel CLI commands to set environment variables."""
    
    print("🚀 Vercel Environment Variables Setup")
    print("=" * 50)
    
    # Required environment variables for production
    env_vars = {
        'OPENAI_API_KEY': 'Your OpenAI API key (sk-...)',
        'EMBEDDING_BACKEND': 'openai',
        'EMBEDDING_MODEL': 'text-embedding-3-large',
        'OPENAI_GEN_MODEL': 'gpt-4o-mini',
        'FLASK_ENV': 'production',
        'TOKENIZERS_PARALLELISM': 'false',
        'OPENAI_HTTP_TIMEOUT': '90'
    }
    
    print("\n📋 Copy and run these commands in your terminal:")
    print("-" * 50)
    
    for var_name, description in env_vars.items():
        if var_name == 'OPENAI_API_KEY':
            print(f"# {description}")
            print(f"vercel env add {var_name}")
            print("# Enter your actual API key when prompted\n")
        else:
            print(f"# {description}")
            print(f"echo '{env_vars[var_name]}' | vercel env add {var_name}")
            print()
    
    print("# Redeploy to apply changes")
    print("vercel --prod")
    
    print("\n" + "=" * 50)
    print("📝 Alternative: Set via Vercel Dashboard")
    print("1. Go to https://vercel.com/dashboard")
    print("2. Select your project")
    print("3. Go to Settings → Environment Variables")
    print("4. Add each variable manually")
    
    print("\n🔧 Environment Variables to Add:")
    print("-" * 30)
    for var_name, value in env_vars.items():
        if var_name == 'OPENAI_API_KEY':
            print(f"{var_name} = sk-your_actual_api_key")
        else:
            print(f"{var_name} = {value}")

def check_local_env():
    """Check if local .env file exists and show current values."""
    env_file = Path(".env")
    
    if not env_file.exists():
        print("⚠️  No .env file found locally")
        return
    
    print("\n📄 Current local .env values:")
    print("-" * 30)
    
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if 'API_KEY' in key and value != 'your_openai_api_key_here':
                        print(f"{key} = {'*' * 20} (hidden)")
                    else:
                        print(f"{key} = {value}")
    except Exception as e:
        print(f"Error reading .env file: {e}")

if __name__ == "__main__":
    generate_vercel_commands()
    check_local_env()
    
    print("\n✅ Next Steps:")
    print("1. Run the vercel env commands above")
    print("2. Deploy: vercel --prod")
    print("3. Test your deployment endpoints")
    print("4. Monitor function performance in Vercel dashboard")