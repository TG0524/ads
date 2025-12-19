# 🚀 Deployment Options for Amazon Ads Assistant

Vercel has a 250MB limit that's exceeded by the FAISS dependency. Here are better alternatives:

## 🎯 **Recommended: Railway (Easiest)**

### Why Railway?
- ✅ **No size limits** - Supports FAISS and large dependencies
- ✅ **Free tier** - $5/month credit, enough for this app
- ✅ **GitHub integration** - Auto-deploy on push
- ✅ **Simple setup** - Just connect and deploy

### Deploy to Railway:
1. Go to https://railway.app
2. Sign up with GitHub
3. Click "Deploy from GitHub repo"
4. Select your repository
5. Railway will auto-detect Python and deploy!

**Environment Variables to Set:**
```
OPENAI_API_KEY=sk-your-key-here
EMBEDDING_BACKEND=openai
EMBEDDING_MODEL=text-embedding-3-large
OPENAI_GEN_MODEL=gpt-4o
```

---

## 🔧 **Alternative: Render**

### Deploy to Render:
1. Go to https://render.com
2. Connect GitHub repository
3. Create new "Web Service"
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `cd api && python app.py`

---

## 🐳 **Docker Deployment (Google Cloud Run, etc.)**

### Build and Deploy:
```bash
# Build Docker image
docker build -t amazon-ads-assistant .

# Test locally
docker run -p 5000:5000 -e OPENAI_API_KEY=your-key amazon-ads-assistant

# Deploy to Google Cloud Run
gcloud run deploy amazon-ads-assistant --image amazon-ads-assistant --platform managed
```

---

## 📊 **Size Comparison:**

| Platform | Size Limit | FAISS Support | Cost |
|----------|------------|---------------|------|
| Vercel | 250MB | ❌ Too large | Free |
| Railway | No limit | ✅ Works | $5/month |
| Render | No limit | ✅ Works | Free tier |
| Google Cloud Run | 32GB | ✅ Works | Pay per use |
| Heroku | No limit | ✅ Works | $7/month |

---

## 🎯 **Quick Start with Railway:**

1. **Push current code** (already done)
2. **Go to Railway.app** 
3. **Connect GitHub repo**
4. **Add environment variables**
5. **Deploy automatically!**

Your app will be live at: `https://your-app-name.railway.app`

---

## 🔧 **Files Added for Deployment:**

- `railway.json` - Railway configuration
- `Procfile` - Heroku/Railway start command  
- `runtime.txt` - Python version specification
- `Dockerfile` - Container deployment
- This guide - `DEPLOYMENT_OPTIONS.md`

**All platforms will work with the current FAISS-based code!** 🚀