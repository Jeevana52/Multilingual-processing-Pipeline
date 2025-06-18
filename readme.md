# 🌍 Multilingual Processing Pipeline

A comprehensive text processing and translation pipeline with sentiment analysis, built with Streamlit and Python.

## ✨ Features

- **Multi-Language Translation**: Support for 24+ languages
- **Translation Methods**:
  - Google Translate (default, free)
  - Local Transformer Models (GPU-accelerated)
  - Anthropic Claude API (premium)
- **Text Analysis**:
  - NLTK-based sentiment scoring
  - AI-powered emotion classification 
  - Automated keyword extraction
- **Database Integration**:
  - SQLite storage for translation history 
  - Analytics dashboard with usage statistics
- **Real-time Processing**: Asynchronous processing pipeline

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git
- (Optional) NVIDIA GPU with CUDA support for local transformer models
- (Optional) Anthropic API key for premium translation features

### Local Development

1. Clone the repository
   ```bash
   git clone https://github.com/Jeevana52/Multilingual-processing-Pipeline.git
   cd Multilingual-processing-Pipeline
   ```

2. Create and activate a virtual environment
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables
   ```bash
   # Create a .env file in the project root
   touch .env

   # Add the following variables (replace with your values)
   ANTHROPIC_API_KEY=your_api_key_here
   GOOGLE_TRANSLATE_API_KEY=your_api_key_here  # Optional
   ```

5. Initialize the database
   ```bash
   python scripts/init_db.py
   ```

6. Run the application
   ```bash
   streamlit run app.py
   ```

The application will be available at `http://localhost:8501` by default.

### Configuration Options

- **Translation Method**: Choose between Google Translate (default), local transformer models, or Anthropic Claude API in the settings panel
- **Database Location**: Configure SQLite database path in `config.py`
- **Model Settings**: Adjust transformer model parameters in `models/config.py`

### Troubleshooting

- If you encounter CUDA errors, ensure your NVIDIA drivers are up to date
- For API-related issues, verify your API keys in the `.env` file
- Check the logs in `logs/app.log` for detailed error information

## 🎯 Roadmap

- Add more translation providers
- Implement caching for better performance
- Add API endpoint for programmatic access
- Support for document translation
- Advanced analytics dashboard
- Multi-user support with authentication

## 📚 Dependencies

- **Streamlit**: Web interface
- **Deep Translator**: Google Translate integration
- **NLTK**: Natural language processing
- **Transformers**: Hugging Face models
- **SQLAlchemy**: Database ORM
- **Torch**: Machine learning backend

## 🏆 Acknowledgments

- Google Translate for free translation services
- Hugging Face for transformer models
- NLTK for sentiment analysis tools
- Streamlit for the amazing web framework

# Made with ❤️ and Python
