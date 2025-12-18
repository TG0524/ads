# 🚀 Vercel Deployment Guide - Amazon Ads Automation Platform

## 📋 Environment Variables for Vercel

### Required Environment Variables

Copy these environment variables to your Vercel project settings:

#### **OpenAI Configuration**
```
OPENAI_API_KEY=sk-your_openai_api_key_here
OPENAI_GEN_MODEL=gpt-4o-mini
OPENAI_HTTP_TIMEOUT=90
```

#### **Embedding Configuration**
```
EMBEDDING_BACKEND=openai
EMBEDDING_MODEL=text-embedding-3-large
```

#### **Application Configuration**
```
FLASK_ENV=production
TOKENIZERS_PARALLELISM=false
```

---

## 🔧 Step-by-Step Vercel Setup

### 1. **Deploy to Vercel**
```bash
# Install Vercel CLI (if not already installed)
npm i -g vercel

# Deploy from your project directory
vercel

# Follow the prompts:
# - Link to existing project or create new
# - Set up project settings
```

### 2. **Configure Environment Variables**

#### Option A: Via Vercel Dashboard (Recommended)
1. Go to [vercel.com/dashboard](https://vercel.com/dashboard)
2. Select your project
3. Go to **Settings** → **Environment Variables**
4. Add each variable:

| Name | Value | Environment |
|------|-------|-------------|
| `OPENAI_API_KEY` | `sk-your_actual_key` | Production, Preview, Development |
| `EMBEDDING_BACKEND` | `openai` | Production, Preview, Development |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | Production, Preview, Development |
| `OPENAI_GEN_MODEL` | `gpt-4o-mini` | Production, Preview, Development |
| `FLASK_ENV` | `production` | Production |
| `TOKENIZERS_PARALLELISM` | `false` | Production, Preview, Development |

#### Option B: Via Vercel CLI
```bash
# Set environment variables via CLI
vercel env add OPENAI_API_KEY
# Enter your API key when prompted

vercel env add EMBEDDING_BACKEND
# Enter: openai

vercel env add EMBEDDING_MODEL  
# Enter: text-embedding-3-large

vercel env add OPENAI_GEN_MODEL
# Enter: gpt-4o-mini

vercel env add FLASK_ENV
# Enter: production

vercel env add TOKENIZERS_PARALLELISM
# Enter: false
```

### 3. **Redeploy After Setting Variables**
```bash
# Redeploy to apply environment variables
vercel --prod
```

---

## 📁 Project Structure for Vercel

Your project should have this structure:
```
├── api/
│   └── index.py          # Vercel serverless function entry point
├── Data/                 # Data files (will be included in deployment)
│   ├── docs2.jsonl
│   ├── faiss2.index
│   └── japan.json
├── utils/
│   └── embedding.py
├── app.py               # Main Flask application
├── 12.py                # Core processing script
├── vercel.json          # Vercel configuration
├── requirements.txt     # Python dependencies
└── .env.example         # Environment template (safe to commit)
```

---

## ⚙️ Vercel Configuration (`vercel.json`)

Current configuration:
```json
{
  "functions": {
    "api/index.py": {
      "maxDuration": 60
    }
  },
  "routes": [
    {
      "src": "/(.*)",
      "dest": "/api/index.py"
    }
  ]
}
```

### Recommended Updates for Production:
```json
{
  "functions": {
    "api/index.py": {
      "maxDuration": 300,
      "memory": 1024
    }
  },
  "routes": [
    {
      "src": "/(.*)",
      "dest": "/api/index.py"
    }
  ],
  "env": {
    "TOKENIZERS_PARALLELISM": "false"
  }
}
```

---

## 🔍 Testing Your Deployment

### 1. **Health Check**
```bash
curl https://your-project.vercel.app/health
```

Expected response:
```json
{
  "status": "ok",
  "message": "Flask app is running"
}
```

### 2. **Test API Endpoints**
```bash
# Test retrieval endpoint
curl -X POST https://your-project.vercel.app/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "campaign_brief": "Target young professionals interested in tech gadgets",
    "top_k": 5
  }'
```

### 3. **Test Generation Endpoint**
```bash
# Test generation endpoint
curl -X POST https://your-project.vercel.app/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "campaign_brief": "Target pet owners in urban areas",
    "top_k": 10
  }'
```

---

## 🚨 Important Notes

### **Security**
- ✅ Never commit `.env` file to repository
- ✅ Use Vercel's environment variable system
- ✅ Set variables for all environments (Production, Preview, Development)

### **Performance**
- 🔧 Increase `maxDuration` to 300s for AI processing
- 🔧 Set `memory` to 1024MB for large embedding operations
- 🔧 Consider using Vercel's Edge Functions for faster response times

### **Monitoring**
- 📊 Monitor function execution time in Vercel dashboard
- 📊 Check OpenAI API usage and costs
- 📊 Set up alerts for function timeouts or errors

### **Data Files**
- 📁 Large files (>100MB) may need Vercel Pro plan
- 📁 Consider using external storage for very large datasets
- 📁 Current data files (~45MB) should work on free tier

## 🔧 Troubleshooting

### **"Pattern doesn't match any Serverless Functions" Error**
If you get this error, it means Vercel can't find the serverless function:

1. **Check file structure**: Ensure `api/index.py` exists
2. **Verify vercel.json**: Make sure the path matches exactly
3. **Test locally**: Run `vercel dev` to test before deploying
4. **Check Python runtime**: Ensure `requirements.txt` is in root directory

### **Import Errors**
If the function can't import dependencies:
- Ensure all dependencies are in `requirements.txt`
- Check that file paths are correct in `api/index.py`
- Verify environment variables are set in Vercel dashboard

---

## 🎯 Quick Deployment Checklist

- [ ] Environment variables set in Vercel dashboard
- [ ] `vercel.json` configured with appropriate timeouts
- [ ] All required files present in repository
- [ ] OpenAI API key is valid and has sufficient credits
- [ ] Test endpoints working after deployment
- [ ] Monitor function performance and costs

Your Amazon Ads Automation Platform is now ready for production on Vercel! 🚀