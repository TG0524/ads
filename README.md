# Amazon Ads Automation Platform

A powerful automation platform for Amazon advertising campaigns, built with AI-powered segment generation and keyword optimization.

## 🚀 Features

- **AI-Powered Campaign Generation**: Automatically generate targeted advertising segments based on campaign briefs
- **Smart Keyword Translation**: Seamless English-to-Japanese keyword translation for international campaigns  
- **Intelligent Segment Retrieval**: Find the most relevant audience segments using advanced matching algorithms
- **Real-time Campaign Optimization**: Dynamic adjustment of targeting parameters and bid strategies
- **Interactive Web Interface**: User-friendly dashboard for campaign management and monitoring

## 🛠️ Built With

- **Backend**: Python Flask with OpenAI integration
- **Frontend**: Modern web interface with real-time updates
- **AI/ML**: OpenAI GPT models for content generation and translation
- **Data Processing**: Advanced embedding models for semantic matching

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- Required environment variables (see setup section)

## ⚙️ Quick Setup

### Option 1: Automated Setup (Recommended)
```bash
# Clone the repository
git clone https://github.com/TG0524/ads.git
cd ads

# Run the setup script
python setup_env.py

# Install dependencies
pip install -r requirements.txt

# Start the application
python app.py
```

### Option 2: Manual Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/TG0524/ads.git
   cd ads
   ```

2. **Configure environment variables**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your OpenAI API key
   # OPENAI_API_KEY=sk-your_actual_api_key_here
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:5000`

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | - | Your OpenAI API key |
| `EMBEDDING_BACKEND` | ✅ Yes | `openai` | Embedding service backend |
| `EMBEDDING_MODEL` | ✅ Yes | `text-embedding-3-large` | Embedding model for semantic search |
| `OPENAI_GEN_MODEL` | No | `gpt-4o-mini` | Model for content generation |
| `FLASK_ENV` | No | `development` | Flask environment |
| `PORT` | No | `5000` | Application port |

## 🎯 Usage

The platform provides both API endpoints and a web interface for:
- Generating campaign segments from briefs
- Retrieving similar segments from existing data
- Translating keywords for international markets
- Optimizing campaign performance

## 📊 API Endpoints

- `POST /api/generate` - Generate new campaign segments
- `POST /api/retrieve` - Retrieve similar segments
- `GET /health` - Health check endpoint

## 🤝 Contributing

This project is actively developed and maintained. Feel free to submit issues and enhancement requests.

## 📄 License

This project is licensed under the MIT License.

---

*Built with ❤️ for Amazon advertising automation*
