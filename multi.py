import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio
import aiohttp
import streamlit as st
import pandas as pd
from deep_translator import GoogleTranslator as DeepGoogleTranslator
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import torch

# Set page config at very start
st.set_page_config(
    page_title="Multilingual Processing Pipeline",
    page_icon="🌍",
    layout="wide"
)

# Safe imports with fallbacks
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    
    # Download required NLTK data safely
    try:
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True) 
        nltk.download('stopwords', quiet=True)
    except:
        pass
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available - sentiment analysis will be limited")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available - local models disabled")

# Configuration
@dataclass
class Config:
    # API Keys (set as environment variables)
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    AZURE_TRANSLATOR_KEY: str = os.getenv("AZURE_TRANSLATOR_KEY", "")
    AZURE_TRANSLATOR_ENDPOINT: str = os.getenv("AZURE_TRANSLATOR_ENDPOINT", "https://api.cognitive.microsofttranslator.com")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    TOGETHER_API_KEY: str = os.getenv("TOGETHER_API_KEY", "")
    
    # Supported languages
    SUPPORTED_LANGUAGES: Dict[str, str] = None
    
    # Database
    DATABASE_URL: str = "sqlite:///multilingual_pipeline.db"
    
    # Model settings
    MAX_TOKENS: int = 4000
    TEMPERATURE: float = 0.3
    
    # Local model settings
    USE_LOCAL_MODELS: bool = TRANSFORMERS_AVAILABLE and torch.cuda.is_available()
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.SUPPORTED_LANGUAGES is None:
            self.SUPPORTED_LANGUAGES = {
                'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
                'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
                'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi',
                'tr': 'Turkish', 'pl': 'Polish', 'nl': 'Dutch', 'sv': 'Swedish',
                'no': 'Norwegian', 'da': 'Danish', 'fi': 'Finnish', 'cs': 'Czech',
                'hu': 'Hungarian', 'ro': 'Romanian', 'bg': 'Bulgarian', 'hr': 'Croatian'
            }

config = Config()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database Models
Base = declarative_base()

class TranslationRecord(Base):
    __tablename__ = 'translations'
    
    id = Column(Integer, primary_key=True)
    source_text = Column(Text, nullable=False)
    translated_text = Column(Text, nullable=False)
    source_language = Column(String(10), nullable=False)
    target_language = Column(String(10), nullable=False)
    confidence_score = Column(Float, default=0.0)
    method = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Float, default=0.0)

class LanguageDetectionRecord(Base):
    __tablename__ = 'language_detections'
    
    id = Column(Integer, primary_key=True)
    text = Column(Text, nullable=False)
    detected_language = Column(String(10), nullable=False)
    confidence = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Database Manager
class DatabaseManager:
    def __init__(self, database_url: str = config.DATABASE_URL):
        try:
            self.engine = create_engine(database_url, echo=False)
            Base.metadata.create_all(self.engine)
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            self.session = None
    
    def save_translation(self, source_text: str, translated_text: str, 
                        source_lang: str, target_lang: str, method: str,
                        confidence: float = 0.0, processing_time: float = 0.0):
        if not self.session:
            return None
            
        try:
            record = TranslationRecord(
                source_text=source_text,
                translated_text=translated_text,
                source_language=source_lang,
                target_language=target_lang,
                method=method,
                confidence_score=confidence,
                processing_time=processing_time
            )
            self.session.add(record)
            self.session.commit()
            return record.id
        except Exception as e:
            logger.error(f"Failed to save translation: {e}")
            self.session.rollback()
            return None
    
    def get_translation_history(self, limit: int = 100) -> List[Dict]:
        if not self.session:
            return []
            
        try:
            records = self.session.query(TranslationRecord).order_by(
                TranslationRecord.timestamp.desc()).limit(limit).all()
            return [{
                'id': r.id,
                'source_text': r.source_text,
                'translated_text': r.translated_text,
                'source_language': r.source_language,
                'target_language': r.target_language,
                'method': r.method,
                'confidence': r.confidence_score,
                'timestamp': r.timestamp,
                'processing_time': r.processing_time
            } for r in records]
        except Exception as e:
            logger.error(f"Failed to get translation history: {e}")
            return []

# Language Detection
class LanguageDetector:
    def __init__(self):
        try:
            self.translator = DeepGoogleTranslator(source='auto', target='en')
            self.available = True
        except Exception as e:
            logger.error(f"Language detector initialization failed: {e}")
            self.available = False
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language with confidence score"""
        if not self.available:
            return 'en', 0.5
            
        try:
            detected_lang = self.translator.detect_language(text)
            return detected_lang, 0.9  # Deep translator doesn't provide confidence score
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return 'en', 0.5

# Text Processing
class TextProcessor:
    def __init__(self):
        self.sentiment_analyzer = None
        self.emotion_classifier = None
        
        if NLTK_AVAILABLE:
            try:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
            except Exception as e:
                logger.error(f"Failed to initialize sentiment analyzer: {e}")
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.emotion_classifier = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=False
                )
            except Exception as e:
                logger.error(f"Failed to initialize emotion classifier: {e}")
    
    def analyze_sentiment(self, text: str) -> Dict:
        if not self.sentiment_analyzer:
            return {
                'compound': 0.0,
                'positive': 0.33,
                'negative': 0.33,
                'neutral': 0.34,
                'sentiment': 'neutral'
            }
        
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            return {
                'compound': scores['compound'],
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'sentiment': 'positive' if scores['compound'] > 0.05 else 'negative' if scores['compound'] < -0.05 else 'neutral'
            }
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {'compound': 0.0, 'positive': 0.33, 'negative': 0.33, 'neutral': 0.34, 'sentiment': 'neutral'}
    
    def extract_emotions(self, text: str) -> Dict:
        if not self.emotion_classifier:
            return {'emotion': 'neutral', 'confidence': 0.0}
        
        try:
            result = self.emotion_classifier(text)
            return {
                'emotion': result[0]['label'].lower(),
                'confidence': result[0]['score']
            }
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            return {'emotion': 'neutral', 'confidence': 0.0}
    
    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        if not NLTK_AVAILABLE:
            # Simple fallback - split by spaces and take unique words
            words = text.lower().split()
            return list(set(words))[:num_keywords]
        
        try:
            words = word_tokenize(text.lower())
            stop_words = set(stopwords.words('english'))
            keywords = [word for word in words if word.isalpha() and word not in stop_words]
            
            from collections import Counter
            word_freq = Counter(keywords)
            return [word for word, freq in word_freq.most_common(num_keywords)]
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            words = text.lower().split()
            return list(set(words))[:num_keywords]

# Translation Engines
class GoogleTranslator:
    """Google Translate using deep_translator"""
    def __init__(self):
        self.available = True
    
    async def translate(self, text: str, target_lang: str, source_lang: str = None) -> Dict:
        if not self.available:
            return None
            
        try:
            start_time = datetime.now()
            translator = DeepGoogleTranslator(
                source=source_lang if source_lang else 'auto',
                target=target_lang
            )
            translated_text = translator.translate(text)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'translated_text': translated_text,
                'source_language': source_lang or translator.detect_language(text),
                'target_language': target_lang,
                'confidence': 0.85,
                'method': 'google',
                'processing_time': processing_time
            }
        except Exception as e:
            logger.error(f"Google translation error: {e}")
            return None

class AnthropicTranslator:
    """Translation using Anthropic Claude API"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.available = bool(api_key)
    
    async def translate(self, text: str, target_lang: str, source_lang: str = None) -> Dict:
        if not self.available:
            return None
            
        try:
            target_lang_name = config.SUPPORTED_LANGUAGES.get(target_lang, target_lang)
            source_lang_name = config.SUPPORTED_LANGUAGES.get(source_lang, source_lang) if source_lang else "detected language"
            
            headers = {
                'Content-Type': 'application/json',
                'x-api-key': self.api_key,
                'anthropic-version': '2023-06-01'
            }
            
            data = {
                'model': 'claude-3-haiku-20240307',
                'max_tokens': 1000,
                'temperature': 0.1,
                'messages': [{
                    'role': 'user',
                    'content': f'Translate the following text from {source_lang_name} to {target_lang_name}. '
                              f'Provide only the translation without any additional text:\n\n{text}'
                }]
            }
            
            start_time = datetime.now()
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        processing_time = (datetime.now() - start_time).total_seconds()
                        
                        if 'content' in result and result['content']:
                            translated_text = result['content'][0]['text'].strip()
                            return {
                                'translated_text': translated_text,
                                'source_language': source_lang,
                                'target_language': target_lang,
                                'confidence': 0.90,
                                'method': 'anthropic',
                                'processing_time': processing_time
                            }
                    else:
                        logger.error(f"Anthropic API error: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Anthropic translation error: {e}")
            return None

class LocalTransformerTranslator:
    """Local transformer-based translation using Hugging Face models"""
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = config.DEVICE
        self.available = TRANSFORMERS_AVAILABLE and config.USE_LOCAL_MODELS
        
        if self.available:
            try:
                self._load_model('facebook/m2m100_418M')
            except Exception as e:
                logger.error(f"Failed to load local model: {e}")
                self.available = False
    
    def _load_model(self, model_name: str):
        """Load translation model"""
        if not self.available:
            return
            
        try:
            if model_name not in self.models:
                logger.info(f"Loading model: {model_name}")
                
                if 'm2m100' in model_name.lower():
                    self.tokenizers[model_name] = M2M100Tokenizer.from_pretrained(model_name)
                    self.models[model_name] = M2M100ForConditionalGeneration.from_pretrained(model_name)
                
                self.models[model_name].to(self.device)
                logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            self.available = False
    
    async def translate(self, text: str, target_lang: str, source_lang: str = None) -> Dict:
        if not self.available:
            return None
            
        try:
            start_time = datetime.now()
            model_name = 'facebook/m2m100_418M'
            
            if model_name not in self.models:
                return None
            
            tokenizer = self.tokenizers[model_name]
            model = self.models[model_name]
            
            # Set source language
            if source_lang:
                tokenizer.src_lang = source_lang
            
            # Encode text
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.get_lang_id(target_lang),
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )
            
            translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'translated_text': translated_text,
                'source_language': source_lang,
                'target_language': target_lang,
                'confidence': 0.85,
                'method': 'local_transformer',
                'processing_time': processing_time
            }
        except Exception as e:
            logger.error(f"Local transformer translation error: {e}")
            return None

# Main Pipeline
class MultilingualPipeline:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.language_detector = LanguageDetector()
        self.text_processor = TextProcessor()
        
        # Initialize translators with error handling
        self.translators = {}
        
        # Google Translate (free tier, most reliable)
        google_translator = GoogleTranslator()
        if google_translator.available:
            self.translators['google'] = google_translator
        
        # Local transformer (free, but requires resources)
        if config.USE_LOCAL_MODELS:
            local_translator = LocalTransformerTranslator()
            if local_translator.available:
                self.translators['local'] = local_translator
        
        # API-based translators (require API keys)
        if config.ANTHROPIC_API_KEY:
            self.translators['anthropic'] = AnthropicTranslator(config.ANTHROPIC_API_KEY)
        
        logger.info(f"Initialized translators: {list(self.translators.keys())}")
        
        if not self.translators:
            logger.warning("No translators available! Please check your configuration.")
    
    async def process_text(self, text: str, target_language: str = None, 
                          source_language: str = None, method: str = 'auto') -> Dict:
        """Complete text processing pipeline"""
        start_time = datetime.now()
        
        # Step 1: Language Detection
        if not source_language:
            detected_lang, confidence = self.language_detector.detect_language(text)
            source_language = detected_lang
            logger.info(f"Detected language: {source_language} (confidence: {confidence})")
        
        # Step 2: Text Analysis
        sentiment = self.text_processor.analyze_sentiment(text)
        emotions = self.text_processor.extract_emotions(text)
        keywords = self.text_processor.extract_keywords(text)
        
        result = {
            'original_text': text,
            'source_language': source_language,
            'sentiment': sentiment,
            'emotions': emotions,
            'keywords': keywords,
            'translations': {}
        }
        
        # Step 3: Translation
        if target_language and target_language != source_language and self.translators:
            translation_result = await self.translate_text(
                text, target_language, source_language, method
            )
            if translation_result:
                result['translations'][target_language] = translation_result
                
                # Save to database
                self.db_manager.save_translation(
                    text, translation_result['translated_text'],
                    source_language, target_language,
                    translation_result['method'],
                    translation_result['confidence'],
                    translation_result['processing_time']
                )
        
        total_time = (datetime.now() - start_time).total_seconds()
        result['total_processing_time'] = total_time
        
        return result
    
    async def translate_text(self, text: str, target_lang: str, 
                           source_lang: str = None, method: str = 'auto') -> Dict:
        """Translate text using specified or best available method"""
        
        if not self.translators:
            logger.error("No translators available")
            return None
        
        # Define method priority
        if method == 'auto':
            priority_methods = ['google', 'local', 'anthropic']
            # Filter to only available methods
            priority_methods = [m for m in priority_methods if m in self.translators]
        else:
            priority_methods = [method] if method in self.translators else ['google']
        
        # Try translation methods in order
        for method_name in priority_methods:
            if method_name in self.translators:
                try:
                    result = await self.translators[method_name].translate(
                        text, target_lang, source_lang
                    )
                    if result:
                        logger.info(f"Translation successful using {method_name}")
                        return result
                except Exception as e:
                    logger.error(f"Translation failed with {method_name}: {e}")
                    continue
        
        logger.error("All translation methods failed")
        return None
    
    def get_translation_analytics(self) -> Dict:
        """Get analytics about translation usage"""
        history = self.db_manager.get_translation_history(1000)
        if not history:
            return {'available_translators': list(self.translators.keys())}
        
        df = pd.DataFrame(history)
        
        return {
            'total_translations': len(df),
            'average_confidence': df['confidence'].mean(),
            'most_common_source_lang': df['source_language'].mode().iloc[0] if len(df) > 0 else None,
            'most_common_target_lang': df['target_language'].mode().iloc[0] if len(df) > 0 else None,
            'method_distribution': df['method'].value_counts().to_dict(),
            'average_processing_time': df['processing_time'].mean(),
            'languages_processed': {
                'source': df['source_language'].unique().tolist(),
                'target': df['target_language'].unique().tolist()
            },
            'available_translators': list(self.translators.keys())
        }

# Streamlit Interface
def create_streamlit_app():
    st.title("🌍 Multilingual Processing Pipeline")
    st.markdown("**Cross-language translation with sentiment analysis**")
    
    # Initialize pipeline
    if 'pipeline' not in st.session_state:
        with st.spinner("Initializing pipeline..."):
            st.session_state.pipeline = MultilingualPipeline()
    
    pipeline = st.session_state.pipeline
    
    # Check if any translators are available
    if not pipeline.translators:
        st.error("❌ No translators available! Please check your configuration.")
        st.info("💡 Make sure you have internet connection for Google Translate, or set up API keys.")
        return
    
    # Display available translators
    st.sidebar.header("Available Translators")
    available_translators = list(pipeline.translators.keys())
    st.sidebar.success(f"Active: {', '.join(available_translators)}")
    
    # Configuration
    st.sidebar.header("Configuration")
    
    languages = config.SUPPORTED_LANGUAGES
    source_lang = st.sidebar.selectbox(
        "Source Language",
        options=['auto'] + list(languages.keys()),
        format_func=lambda x: 'Auto-detect' if x == 'auto' else languages.get(x, x)
    )
    
    target_lang = st.sidebar.selectbox(
        "Target Language",
        options=list(languages.keys()),
        format_func=lambda x: languages.get(x, x),
        index=1  # Default to Spanish
    )
    
    translation_method = st.sidebar.selectbox(
        "Translation Method",
        options=['auto'] + available_translators,
        help="Auto will try the best available method"
    )
    
    # Main interface
    st.header("Text Processing")
    
    text_input = st.text_area(
        "Enter text to process:",
        height=150,
        placeholder="Enter your text here..."
    )
    
    if st.button("Process Text", type="primary"):
        if not text_input.strip():
            st.warning("Please enter some text to process.")
            return
            
        with st.spinner("Processing text..."):
            try:
                result = asyncio.run(pipeline.process_text(
                    text_input,
                    target_lang,
                    source_lang if source_lang != 'auto' else None,
                    translation_method
                ))
                
                if result:
                    st.success("✅ Processing completed!")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Source Language", 
                                 config.SUPPORTED_LANGUAGES.get(result['source_language'], result['source_language']))
                    with col2:
                        st.metric("Processing Time", f"{result['total_processing_time']:.2f}s")
                    with col3:
                        if result.get('translations'):
                            translation_info = list(result['translations'].values())[0]
                            st.metric("Translation Method", translation_info['method'].title())
                    
                    # Sentiment Analysis
                    st.subheader("📊 Sentiment Analysis")
                    sentiment = result['sentiment']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        sentiment_color = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
                        st.metric("Overall Sentiment", 
                                 f"{sentiment_color.get(sentiment['sentiment'], '⚪')} {sentiment['sentiment'].title()}")
                    with col2:
                        st.metric("Compound Score", f"{sentiment['compound']:.3f}")
                    with col3:
                        confidence = max(sentiment['positive'], sentiment['negative'], sentiment['neutral'])
                        st.metric("Confidence", f"{confidence:.3f}")
                    
                    # Emotion Analysis
                    if result.get('emotions') and result['emotions']['confidence'] > 0:
                        st.subheader("😊 Emotion Analysis")
                        emotion = result['emotions']
                        emotion_emojis = {
                            'joy': '😊', 'sadness': '😢', 'anger': '😠', 'fear': '😨',
                            'surprise': '😲', 'disgust': '🤢', 'neutral': '😐'
                        }
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Primary Emotion", 
                                     f"{emotion_emojis.get(emotion['emotion'], '😐')} {emotion['emotion'].title()}")
                        with col2:
                            st.metric("Confidence", f"{emotion['confidence']:.3f}")
                    
                    # Keywords
                    if result.get('keywords'):
                        st.subheader("🔑 Key Terms")
                        st.write(" • ".join(result['keywords'][:10]))
                    
                    # Translation Results
                    if result.get('translations'):
                        st.subheader("🌐 Translation Results")
                        for lang, translation in result['translations'].items():
                            with st.expander(f"Translation to {config.SUPPORTED_LANGUAGES.get(lang, lang)}", expanded=True):
                                st.write(translation['translated_text'])
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.caption(f"Method: {translation['method'].title()}")
                                with col2:
                                    st.caption(f"Confidence: {translation['confidence']:.2f}")
                                with col3:
                                    st.caption(f"Time: {translation['processing_time']:.2f}s")
                
                else:
                    st.error("❌ Processing failed. Please try again.")
                    
            except Exception as e:
                st.error(f"❌ Error processing text: {str(e)}")
                logger.error(f"Processing error: {e}")
    
    # Analytics section
    st.header("📈 Analytics")
    if st.button("Show Analytics"):
        try:
            analytics = pipeline.get_translation_analytics()
            
            if analytics and analytics.get('total_translations', 0) > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Translations", analytics.get('total_translations', 0))
                with col2:
                    st.metric("Average Confidence", f"{analytics.get('average_confidence', 0):.3f}")
                with col3:
                    st.metric("Avg Processing Time", f"{analytics.get('average_processing_time', 0):.2f}s")
                
                # Method distribution
                if analytics.get('method_distribution'):
                    st.subheader("Method Usage")
                    method_df = pd.DataFrame(
                        list(analytics['method_distribution'].items()),
                        columns=['Method', 'Count']
                    )
                    st.bar_chart(method_df.set_index('Method'))
            else:
                st.info("No translation history available yet. Start translating to see analytics!")
                
        except Exception as e:
            st.error(f"❌ Analytics error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("**🌍 Multilingual Processing Pipeline** | Built with Streamlit and Python")

# Configuration help
def show_setup_help():
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Setup Help")
    
    with st.sidebar.expander("API Configuration"):
        st.markdown("""
        **Optional API Keys (for premium features):**
        
        Set environment variables:
        ```bash
        export ANTHROPIC_API_KEY="your-key"
        ```
        
        **Free Options:**
        - 🌐 Google Translate (default)
        - 🔧 Local Models (if GPU available)
        """)

# Main execution
if __name__ == "__main__":
    try:
        show_setup_help()
        create_streamlit_app()
    except Exception as e:
        st.error(f"Application startup failed: {str(e)}")
        logger.error(f"Startup error: {e}")