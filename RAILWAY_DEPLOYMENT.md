# Railway Deployment Guide

## Overview
This guide explains how to deploy the Amazon Ads Automation Platform to Railway.

## Why Railway?
- Vercel has a 250MB size limit which is exceeded by the FAISS dependency
- Railway supports larger deployments and Python applications
- Railway provides persistent storage and better support for data-heavy applications

## Prerequisites
1. Railway account (https://railway.app)
2. GitHub repository with your code
3. OpenAI API key

## Deployment Steps

### 1. Connect Repository to Railway
1. Go to https://railway.app
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your repository (e.g., `TG0524/ads`)
5. Railway will automatically detect it's a Python project

### 2. Configure Environment Variables
In the Railway dashboard, go to your project's Variables tab and add:

```
OPENAI_API_KEY=sk-your_actual_key_here
EMBEDDING_BACKEND=openai
EMBEDDING_MODEL=text-embedding-3-large
OPENAI_GEN_MODEL=gpt-4o-mini
OPENAI_HTTP_TIMEOUT=90
TOKENIZERS_PARALLELISM=false
PORT=8080
```

**Important:** Replace `sk-your_actual_key_here` with your real OpenAI API key.

### 3. Verify Deployment Configuration
The following files are already configured for Railway:

- **Procfile**: Specifies the start command
  ```
  web: cd api && python app.py
  ```

- **railway.json**: Railway-specific configuration
  ```json
  {
    "build": {
      "builder": "NIXPACKS"
    },
    "deploy": {
      "startCommand": "cd api && python app.py",
      "restartPolicyType": "ON_FAILURE"
    }
  }
  ```

- **requirements.txt**: Python dependencies (used by Railway)

### 4. Deploy
1. Railway will automatically deploy when you push to your repository
2. Wait for the build to complete (usually 2-3 minutes)
3. Railway will provide a public URL (e.g., `https://ads-production-5797.up.railway.app`)

### 5. Test the Deployment

#### Test the Homepage
Visit your Railway URL - you should see the Japanese interface.

#### Test the Debug Endpoint
Visit `https://your-app.up.railway.app/debug-script` to verify:
- Script execution works
- Data files are accessible
- Environment variables are set correctly

#### Test the Generate Endpoint
Use the web interface to submit a campaign brief and verify:
- Segments are retrieved correctly
- AI generation works
- Results are displayed in Japanese

## Troubleshooting

### Issue: "Application failed to respond"
**Solution:** Check that:
- The PORT environment variable is set
- The app is binding to `0.0.0.0` (not `localhost`)
- The Procfile has the correct start command

### Issue: "500 Error on /api/generate"
**Solution:** Check the Railway logs for detailed error messages:
1. Go to Railway dashboard
2. Click on your deployment
3. View the "Logs" tab
4. Look for Python errors or missing dependencies

Common causes:
- Missing OPENAI_API_KEY
- Data files not found (check file paths)
- Import errors (missing dependencies)

### Issue: "Data files not found"
**Solution:** Verify that the following files exist in your repository:
- `api/Data/docs2.jsonl`
- `api/Data/faiss2.index`
- `api/Data/japan.json`
- `api/12.py`
- `api/utils/embedding.py`

### Issue: "Module not found" errors
**Solution:** Ensure all dependencies are in `requirements.txt`:
```
Flask==3.0.3
Flask-Cors==4.0.0
openai>=1.45.0
faiss-cpu==1.12.0
numpy>=1.26.4,<2
httpx>=0.27.0
packaging>=21.0
gunicorn>=21.0.0
```

## File Structure
```
project/
├── api/
│   ├── Data/
│   │   ├── docs2.jsonl
│   │   ├── faiss2.index
│   │   └── japan.json
│   ├── utils/
│   │   └── embedding.py
│   ├── 12.py
│   └── app.py
├── public/
│   └── index.html
├── Procfile
├── railway.json
└── requirements.txt
```

## Key Changes for Railway Deployment

### 1. Working Directory
The app automatically changes to the `api/` directory:
```python
API_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(os.getcwd()) != 'api':
    os.chdir(API_DIR)
```

### 2. Data File Paths
The `_path()` function looks for data files in multiple locations:
```python
def _path(*parts):
    p1 = os.path.join("Data", *parts)  # Local Data directory
    p2 = os.path.join("..", "data", *parts)  # Parent data directory
    p3 = os.path.join("..", "Data", *parts)  # Parent Data directory
    if os.path.exists(p1):
        return p1
    elif os.path.exists(p2):
        return p2
    else:
        return p3
```

### 3. Subprocess Execution
The subprocess calls run from the current directory (api/):
```python
result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
```

## Monitoring

### View Logs
```bash
# In Railway dashboard, go to your deployment and click "Logs"
```

### Check Application Health
```bash
curl https://your-app.up.railway.app/health
```

### Debug Script Execution
```bash
curl https://your-app.up.railway.app/debug-script
```

## Cost Considerations
- Railway offers a free tier with limited resources
- OpenAI API usage is billed separately
- Monitor your OpenAI usage in the OpenAI dashboard
- Consider using `gpt-4o-mini` instead of `gpt-4o` to reduce costs

## Security
- Never commit your `.env` file or API keys to Git
- Use Railway's environment variables for sensitive data
- Regularly rotate your API keys
- Monitor API usage for unusual activity

## Support
If you encounter issues:
1. Check Railway logs for error messages
2. Test the `/debug-script` endpoint
3. Verify environment variables are set correctly
4. Ensure all data files are present in the repository
