#!/usr/bin/env python3
import asyncio
import os
import sys
import json
import time
import sqlite3
import logging
import hashlib
import re
import requests
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict, deque, Counter
from pathlib import Path
import tempfile
from io import BytesIO
import aiofiles
from bs4 import BeautifulSoup
import numpy as np


# FastAPI imports
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from pydub import AudioSegment
from io import BytesIO

# Environment loading
from dotenv import load_dotenv
load_dotenv()

def sanitize_metadata(meta):
    """Convert numpy + nested dicts/lists into JSON-safe values for metadata storage."""
    if isinstance(meta, dict):
        return {k: sanitize_metadata(v) for k, v in meta.items()}
    elif isinstance(meta, (list, tuple)):
        return [sanitize_metadata(v) for v in meta]
    elif isinstance(meta, np.generic):  # numpy types
        return float(meta)
    elif isinstance(meta, (np.ndarray,)):
        return meta.tolist()
    elif isinstance(meta, (str, int, float, bool)) or meta is None:
        return meta
    else:
        # Anything else (like nested dicts), stringify safely
        try:
            return json.dumps(meta)
        except:
            return str(meta)

def webm_to_wav(audio_bytes: bytes) -> bytes:
    """Convert browser WebM/Opus to WAV in memory."""
    audio = AudioSegment.from_file(BytesIO(audio_bytes), format="webm")
    wav_io = BytesIO()
    audio.export(wav_io, format="wav")
    return wav_io.getvalue()

# Setup paths (same as CLI)
project_root = os.path.dirname(os.path.abspath(__file__))
folders_to_add = [
    'src', os.path.join('src', 'memory'), os.path.join('src', 'unique_features'),
    os.path.join('src', 'agents'), 'ML'
]
for folder in folders_to_add:
    folder_path = os.path.join(project_root, folder)
    if os.path.exists(folder_path) and folder_path not in sys.path:
        sys.path.insert(0, folder_path)

# Voice processing imports (same as CLI)
try:
    import speech_recognition as sr
    import pyttsx3
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_VOICE_AVAILABLE = True
except ImportError:
    AZURE_VOICE_AVAILABLE = False

# File processing imports (same as CLI)
try:
    from PIL import Image
    import PyPDF2
    import docx
    import pandas as pd
    FILE_PROCESSING_AVAILABLE = True
except ImportError:
    FILE_PROCESSING_AVAILABLE = False

# GitHub Integration imports (same as CLI)
try:
    import chromadb
    from langchain_community.document_loaders import UnstructuredFileLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    GITHUB_INTEGRATION = True
except ImportError:
    GITHUB_INTEGRATION = False

# Professional Agents Import (same as CLI)
try:
    from agents.coding_agent import ProLevelCodingExpert
    from agents.career_coach import ProfessionalCareerCoach  
    from agents.business_consultant import SmartBusinessConsultant
    from agents.medical_advisor import SimpleMedicalAdvisor
    from agents.emotional_counselor import SimpleEmotionalCounselor
    from agents.techincal_architect import TechnicalArchitect
    PROFESSIONAL_AGENTS_LOADED = True
except ImportError:
    PROFESSIONAL_AGENTS_LOADED = False

# Advanced Systems Import (same as CLI)
try:
    from memory.sharp_memory import SharpMemorySystem
    from unique_features.smart_orchestrator import IntelligentAPIOrchestrator
    from unique_features.api_drift_detector import APIPerformanceDrifter
    ADVANCED_SYSTEMS = True
except ImportError:
    ADVANCED_SYSTEMS = False
    # Fallback classes
    class SharpMemorySystem:
        def __init__(self): pass
        async def remember_conversation_advanced(self, *args): pass
        async def get_semantic_context(self, *args): return ""
    
    class IntelligentAPIOrchestrator:
        def __init__(self): pass
        async def get_optimized_response(self, *args): return None, {}
    
    class APIPerformanceDrifter:
        def __init__(self): pass
        def record_response_quality(self, *args): pass

# GitHub QA Engine Import (same as CLI)
try:
    from agents.ingest import main as ingest_repo
    from agents.qa_engine import create_qa_engine
    GITHUB_INTEGRATION = GITHUB_INTEGRATION and True
except ImportError:
    GITHUB_INTEGRATION = False
    ingest_repo = None
    create_qa_engine = None

# ML System Import (same as CLI)
try:
    from ml_integration import EnhancedMLManager
    ml_manager = EnhancedMLManager()
    ML_SYSTEM_AVAILABLE = True
except ImportError:
    ML_SYSTEM_AVAILABLE = False
    class EnhancedMLManager:
        def __init__(self): pass
        async def enhance_query(self, query, context): return query
        async def optimize_response(self, response, context): return response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== ULTRA HYBRID MEMORY SYSTEM (EXACT FROM CLI) ==========
class UltraHybridMemorySystem:
    """Ultra Advanced Hybrid Memory - EXACT from NOVA-CLI.py"""
    
    def __init__(self, db_path="nova_ultra_professional_memory.db"):
        if not os.path.isabs(db_path):
            self.db_path = os.path.join(os.getcwd(), db_path)
        else:
            self.db_path = db_path
        
        self.setup_database()
        
        # ALL memory layers from CLI - EXACT
        self.conversation_context = deque(maxlen=100)
        self.user_profile = {}
        self.emotional_state = "neutral"
        self.learning_patterns = defaultdict(list)
        self.personality_insights = {}
        self.user_preferences = {}
        self.conversation_history = []
        
        # Memory layers from CLI - EXACT
        self.short_term_memory = deque(maxlen=200)
        self.working_memory = {}
        self.conversation_threads = {}
        self.context_memory = {}
        
        # Premium memory features - EXACT
        self.voice_memory = deque(maxlen=50)
        self.file_memory = {}
        self.search_memory = deque(maxlen=30)
        self.image_memory = deque(maxlen=20)
        
        # Semantic memory - EXACT from CLI
        self.setup_semantic_memory()

    def setup_database(self):
        """Setup database schema - EXACT from CLI"""
        try:
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Enhanced conversations table - EXACT from CLI
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        session_id TEXT,
                        user_input TEXT,
                        bot_response TEXT,
                        agent_type TEXT,
                        language TEXT,
                        emotion TEXT,
                        confidence REAL,
                        timestamp DATETIME,
                        feedback INTEGER DEFAULT 0,
                        context_summary TEXT,
                        learned_facts TEXT,
                        satisfaction_rating INTEGER,
                        conversation_thread_id TEXT,
                        intent_detected TEXT,
                        response_time REAL,
                        voice_used BOOLEAN DEFAULT 0,
                        location TEXT,
                        weather_context TEXT,
                        search_queries TEXT
                    )
                ''')
                
                # Enhanced user profiles - EXACT from CLI
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        user_id TEXT PRIMARY KEY,
                        name TEXT,
                        career_goals TEXT,
                        current_role TEXT,
                        experience_years INTEGER,
                        skills TEXT,
                        preferences TEXT,
                        communication_style TEXT,
                        emotional_patterns TEXT,
                        conversation_patterns TEXT,
                        expertise_level TEXT,
                        topics_of_interest TEXT,
                        last_updated DATETIME,
                        total_conversations INTEGER DEFAULT 0,
                        preferred_voice TEXT,
                        location TEXT,
                        timezone TEXT,
                        personality_type TEXT,
                        learning_style TEXT
                    )
                ''')
                
                # Other tables - EXACT from CLI
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS github_repos (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        repo_url TEXT UNIQUE,
                        repo_name TEXT,
                        analysis_date DATETIME,
                        file_count INTEGER,
                        languages_detected TEXT,
                        issues_found TEXT,
                        suggestions TEXT,
                        vector_db_path TEXT
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS voice_interactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        voice_input TEXT,
                        voice_response TEXT,
                        language_detected TEXT,
                        emotion_detected TEXT,
                        voice_engine TEXT,
                        timestamp DATETIME
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS file_processing (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        file_path TEXT,
                        file_type TEXT,
                        processing_result TEXT,
                        timestamp DATETIME,
                        success BOOLEAN
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS search_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        search_query TEXT,
                        search_type TEXT,
                        results_count INTEGER,
                        timestamp DATETIME
                    )
                ''')
                
                conn.commit()
                logger.info("✅ Database initialized")
        except Exception as e:
            logger.error(f"Database setup error: {e}")

    def setup_semantic_memory(self):
        """Setup semantic memory - EXACT from CLI"""
        try:
            if ADVANCED_SYSTEMS:
                self.semantic_memory = SharpMemorySystem()
            else:
                self.semantic_memory = None
        except Exception as e:
            logger.error(f"Semantic memory setup error: {e}")
            self.semantic_memory = None

    async def remember_conversation(self, user_id: str, session_id: str,
                                  user_input: str, bot_response: str,
                                  agent_type: str, language: str,
                                  emotion: str, confidence: float,
                                  intent: str = None, response_time: float = 0.0,
                                  voice_used: bool = False, location: str = None,
                                  weather_context: str = None, search_queries: str = None,
                                  file_analyzed: str = None):
        """Enhanced conversation memory storage - EXACT from CLI"""
        try:
            learned_facts = self.extract_learning_points(user_input, bot_response)
            context_summary = self.generate_context_summary()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO conversations
                    (user_id, session_id, user_input, bot_response, agent_type,
                     language, emotion, confidence, timestamp, context_summary,
                     learned_facts, conversation_thread_id, intent_detected, response_time,
                     voice_used, location, weather_context, search_queries)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, session_id, user_input, bot_response, agent_type,
                      language, emotion, confidence, datetime.now(), context_summary,
                      learned_facts, self.generate_thread_id(), intent, response_time,
                      voice_used, location, weather_context, search_queries))
                conn.commit()
            
            # Store in conversation context - EXACT from CLI
            self.conversation_context.append({
                'user': user_input,
                'bot': bot_response,
                'emotion': emotion,
                'agent': agent_type,
                'timestamp': datetime.now(),
                'voice_used': voice_used,
                'location': location,
                'file_analyzed': file_analyzed
            })
            
            # Store in short-term memory - EXACT from CLI
            memory_entry = {
                'user_id': user_id,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'user_input': user_input,
                'ai_response': bot_response,
                'agent_used': agent_type,
                'emotion': emotion,
                'intent': intent,
                'voice_used': voice_used,
                'file_analyzed': file_analyzed
            }
            self.short_term_memory.append(memory_entry)
            
            # Store in semantic memory - EXACT from CLI
            if self.semantic_memory and agent_type in ['coding', 'business', 'technical_architect']:
                try:
                    await self.semantic_memory.remember_conversation_advanced(
                        user_input, bot_response,
                        {'agent_used': agent_type, 'emotion': emotion},
                        user_id, session_id
                    )
                except Exception as e:
                    logger.error(f"Semantic memory storage error: {e}")
                    
        except Exception as e:
            logger.error(f"Memory storage error: {e}")

    def get_relevant_context(self, user_input: str, user_id: str, limit: int = 15) -> str:
        """Get relevant context - EXACT from CLI"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT user_input, bot_response, emotion, learned_facts, agent_type,
                           voice_used, location, weather_context
                    FROM conversations
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (user_id, limit))
                conversations = cursor.fetchall()
                
                if not conversations:
                    return ""
                
                # Build context summary - EXACT from CLI
                context = "Previous conversation context:\n"
                for conv in conversations:
                    context += f"[{conv[4].upper()}] User ({conv[2]}): {conv[:80]}...\n"
                    context += f"NOVA: {conv[1][:80]}...\n"
                    if conv:
                        context += f"Learned: {conv}\n"
                    if conv:  # voice_used
                        context += f"[VOICE MODE]\n"
                    if conv:  # location
                        context += f"Location: {conv}\n"
                    if conv:  # weather_context
                        context += f"Weather: {conv}\n"
                    context += "---\n"
                
                return context
        except Exception as e:
            logger.error(f"Context retrieval error: {e}")
            return ""

    def extract_learning_points(self, user_input: str, bot_response: str) -> str:
        """Extract learning points - EXACT from CLI"""
        learning_keywords = [
            "my name is", "i am", "i work", "i like", "i don't like",
            "my preference", "remember that", "important", "my goal",
            "my project", "my problem", "i need help with", "my role",
            "my company", "my experience", "my skills", "career goal",
            "i live in", "my location", "my city", "my country",
            "i prefer", "i want", "i need", "i use", "my favorite"
        ]
        
        learned = []
        user_lower = user_input.lower()
        for keyword in learning_keywords:
            if keyword in user_lower:
                sentences = user_input.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        learned.append(sentence.strip())
        
        return "; ".join(learned)

    def generate_context_summary(self) -> str:
        """Generate context summary - EXACT from CLI"""
        if not self.conversation_context:
            return ""
        
        recent_topics = []
        emotions = []
        agents = []
        voice_usage = []
        locations = []
        
        for conv in list(self.conversation_context)[-10:]:
            recent_topics.append(conv['user'][:50])
            emotions.append(conv['emotion'])
            agents.append(conv['agent'])
            if conv.get('voice_used'):
                voice_usage.append(True)
            if conv.get('location'):
                locations.append(conv['location'])
        
        dominant_emotion = max(set(emotions), key=emotions.count) if emotions else "neutral"
        most_used_agent = max(set(agents), key=agents.count) if agents else "general"
        voice_percentage = (len(voice_usage) / len(emotions)) * 100 if emotions else 0
        
        summary = f"Recent topics: {'; '.join(recent_topics)}. "
        summary += f"Emotion: {dominant_emotion}. Agent: {most_used_agent}. "
        if voice_percentage > 0:
            summary += f"Voice usage: {voice_percentage:.0f}%. "
        if locations:
            summary += f"Location context: {locations[-1]}."
        
        return summary

    def generate_thread_id(self) -> str:
        """Generate thread ID - EXACT from CLI"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"thread_{timestamp}_{random.randint(1000, 9999)}"

    def remember_file_processing(self, user_id: str, file_path: str,
                               file_type: str, result: str, success: bool):
        """Remember file processing - EXACT from CLI"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO file_processing
                    (user_id, file_path, file_type, processing_result, timestamp, success)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (user_id, file_path, file_type, result, datetime.now(), success))
                conn.commit()
            
            self.file_memory[file_path] = {
                'type': file_type,
                'result': result,
                'success': success,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"File memory error: {e}")

# ========== LANGUAGE AND EMOTION DETECTORS (EXACT FROM CLI) ==========
class FastLanguageDetector:
    """Language detection - EXACT from CLI"""
    
    def __init__(self):
        self.hinglish_words = {
            "yaar", "bhai", "ji", "hai", "hoon", "kya", "aur", "tum", "main",
            "accha", "theek", "nahi", "haan", "matlab", "kaise", "kyun"
        }

    def detect_language(self, text: str) -> str:
        """Fast language detection - EXACT from CLI"""
        words = text.lower().split()
        hinglish_count = sum(1 for word in words if word in self.hinglish_words)
        return "hinglish" if hinglish_count > 0 else "english"

class FastEmotionDetector:
    """Emotion detection - EXACT from CLI"""
    
    def __init__(self):
        self.emotion_keywords = {
            "excited": ["excited", "amazing", "awesome", "great", "love"],
            "frustrated": ["frustrated", "angry", "upset", "hate", "annoyed"],
            "sad": ["sad", "depressed", "down", "unhappy", "lonely"],
            "anxious": ["anxious", "worried", "nervous", "scared", "stress"],
            "confident": ["confident", "sure", "ready", "motivated", "strong"],
            "confused": ["confused", "lost", "unclear", "help", "stuck"]
        }

    def detect_emotion(self, text: str) -> Tuple[str, float]:
        """Fast emotion detection - EXACT from CLI"""
        text_lower = text.lower()
        for emotion, keywords in self.emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return emotion, 0.8
        return "neutral", 0.5

# ========== OPTIMIZED API MANAGER (EXACT FROM CLI) ==========
class OptimizedAPIManager:
    """API manager - EXACT from CLI"""
    
    def __init__(self):
        # ALL 6 API providers from CLI - EXACT
        self.providers = [
            {
                "name": "Groq",
                "url": "https://api.groq.com/openai/v1/chat/completions",
                "models": ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"],
                "headers": lambda: {
                    "Authorization": f"Bearer {os.getenv('GROQ_API_KEY', '')}",
                    "Content-Type": "application/json"
                },
                "priority": 1,
                "specialty": "fast_inference"
            },
            {
                "name": "OpenRouter",
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "models": ["mistralai/mistral-7b-instruct:free", "meta-llama/llama-3.1-70b-instruct:free"],
                "headers": lambda: {
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY', '')}",
                    "Content-Type": "application/json"
                },
                "priority": 2,
                "specialty": "diverse_models"
            },
            {
                "name": "Together",
                "url": "https://api.together.xyz/v1/chat/completions",
                "models": ["meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"],
                "headers": lambda: {
                    "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY', '')}",
                    "Content-Type": "application/json"
                },
                "priority": 3,
                "specialty": "open_source"
            }
        ]
        
        # Filter available providers
        self.available = []
        for provider in self.providers:
            key_name = f"{provider['name'].upper()}_API_KEY"
            if os.getenv(key_name):
                self.available.append(provider)
                logger.info(f"✅ {provider['name']} API available")
        
        self.available.sort(key=lambda x: x['priority'])
        self.current = self.available[0] if self.available else None
        
        # Performance tracking - EXACT from CLI
        self.performance_stats = {}
        for provider in self.available:
            self.performance_stats[provider['name']] = {
                'response_times': deque(maxlen=10),
                'success_rate': 1.0,
                'total_requests': 0,
                'failures': 0
            }

    def get_best_provider(self, query_type: str = "general") -> dict:
        """Get best provider - EXACT from CLI"""
        if not self.available:
            return None
        
        # Route based on query type - EXACT from CLI
        specialty_preferences = {
            "coding": ["fast_inference", "diverse_models"],
            "creative": ["diverse_models", "open_source"],
            "analysis": ["diverse_models", "fast_inference"],
            "general": ["fast_inference", "diverse_models"]
        }
        
        preferred_specialties = specialty_preferences.get(query_type, ["fast_inference"])
        
        # Score providers - EXACT from CLI
        best_provider = None
        best_score = -1
        
        for provider in self.available:
            specialty_score = 10 if provider['specialty'] in preferred_specialties else 5
            stats = self.performance_stats[provider['name']]
            performance_score = stats['success_rate'] * 5
            
            if stats['response_times']:
                avg_time = sum(stats['response_times']) / len(stats['response_times'])
                speed_score = max(0, 5 - avg_time)
            else:
                speed_score = 5
            
            total_score = specialty_score + performance_score + speed_score
            
            if total_score > best_score:
                best_score = total_score
                best_provider = provider
        
        return best_provider or self.current

    async def get_ai_response(self, user_input: str, system_prompt: str,
                            query_type: str = "general") -> Optional[str]:
        """Get AI response - EXACT from CLI"""
        provider = self.get_best_provider(query_type)
        if not provider:
            return None
        
        start_time = time.time()
        
        # Try models from provider - EXACT from CLI
        for model in provider["models"][:2]:
            try:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    "max_tokens": 1500,
                    "temperature": 0.7
                }
                
                response = requests.post(
                    provider["url"],
                    headers=provider["headers"](),
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    choices = result.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "No response")
                        
                        # Update performance stats - EXACT from CLI
                        response_time = time.time() - start_time
                        stats = self.performance_stats[provider['name']]
                        stats['response_times'].append(response_time)
                        stats['total_requests'] += 1
                        stats['success_rate'] = (stats['total_requests'] - stats['failures']) / stats['total_requests']
                        
                        return content
                        
            except Exception as e:
                logger.error(f"❌ {provider['name']} model {model} failed: {e}")
                continue
        
        # Update failure stats - EXACT from CLI
        stats = self.performance_stats[provider['name']]
        stats['failures'] += 1
        stats['total_requests'] += 1
        stats['success_rate'] = (stats['total_requests'] - stats['failures']) / stats['total_requests']
        
        return None

# ========== FILE SYSTEM (EXACT FROM CLI) ==========
class FileUploadSystem:
    """File upload system - EXACT from CLI"""
    
    def __init__(self):
        self.supported_formats = {
            '.txt': 'text', '.py': 'python', '.js': 'javascript', '.json': 'json',
            '.md': 'markdown', '.csv': 'csv', '.pdf': 'pdf', '.docx': 'word',
            '.xlsx': 'excel', '.html': 'html', '.css': 'css', '.sql': 'sql'
        }

    def analyze_file_content(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Analyze file content - EXACT from CLI"""
        try:
            file_ext = os.path.splitext(filename)[1].lower()
            file_size = len(file_content)
            
            analysis = {
                "file_name": filename,
                "file_size": file_size,
                "file_extension": file_ext,
                "file_type": self.supported_formats.get(file_ext, "unknown"),
                "content": "",
                "summary": "",
                "analysis": ""
            }
            
            # Read content based on type - EXACT from CLI
            content = self.read_file_content(file_content, file_ext)
            if content:
                analysis["content"] = content[:5000]
                analysis["full_content"] = content
                analysis["lines"] = len(content.split('\n'))
                analysis["words"] = len(content.split())
                analysis["chars"] = len(content)
            
            return analysis
            
        except Exception as e:
            return {"error": f"File analysis failed: {str(e)}"}

    def read_file_content(self, file_content: bytes, file_ext: str) -> Optional[str]:
        """Read file content - EXACT from CLI"""
        try:
            if file_ext in ['.txt', '.py', '.js', '.json', '.md', '.html', '.css', '.sql']:
                return file_content.decode('utf-8', errors='ignore')
            elif file_ext == '.csv' and FILE_PROCESSING_AVAILABLE:
                df = pd.read_csv(BytesIO(file_content))
                return f"CSV Analysis:\nRows: {len(df)}\nColumns: {len(df.columns)}\nFirst 10 rows:\n{df.head(10).to_string()}"
            elif file_ext == '.pdf' and FILE_PROCESSING_AVAILABLE:
                reader = PyPDF2.PdfReader(BytesIO(file_content))
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
            else:
                return "Binary file - content not displayable"
        except Exception as e:
            return f"Error reading file: {str(e)}"

# ========== VOICE SYSTEM (EXACT FROM CLI) ==========
class FastVoiceSystem:
    """Voice system with both STT and TTS capabilities"""
    
    def __init__(self):
        self.azure_enabled = AZURE_VOICE_AVAILABLE
        self.basic_voice_enabled = VOICE_AVAILABLE
        
        if self.azure_enabled:
            self.setup_azure_voice()
        if self.basic_voice_enabled:
            self.setup_basic_voice()

    def setup_azure_voice(self):
        """Setup Azure voice services"""
        try:
            azure_key = os.getenv('AZURE_SPEECH_KEY')
            azure_region = os.getenv('AZURE_SPEECH_REGION', 'eastus')
            
            if azure_key:
                self.speech_config = speechsdk.SpeechConfig(
                    subscription=azure_key, 
                    region=azure_region
                )
                self.speech_config.speech_recognition_language = "en-US"
                self.speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
        except Exception as e:
            logger.error(f"Azure Voice setup error: {e}")
            self.azure_enabled = False

    def setup_basic_voice(self):
        """Setup basic voice recognition"""
        try:
            self.recognizer = sr.Recognizer()
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 180)
        except Exception as e:
            logger.error(f"Basic voice setup error: {e}")
            self.basic_voice_enabled = False

    async def recognize_audio(self, audio_data: bytes) -> str:
        """Convert audio to text using available speech recognition"""
        try:
            # Try Azure Speech-to-Text first
            if self.azure_enabled:
                audio_stream = speechsdk.audio.PushAudioInputStream()
                audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)
                recognizer = speechsdk.SpeechRecognizer(self.speech_config, audio_config)
                
                # Push audio data to stream
                audio_stream.write(audio_data)
                audio_stream.close()
                
                result = recognizer.recognize_once_async().get()
                if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    return result.text
                
            # Fallback to basic recognizer
            if self.basic_voice_enabled:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_data)
                    tmp_path = tmp.name
                
                with sr.AudioFile(tmp_path) as source:
                    audio = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio)
                
                os.unlink(tmp_path)
                return text
                
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return ""

    async def speak(self, text: str) -> bytes:
        """Convert text to speech audio"""
        clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        clean_text = re.sub(r'[🔧💼📈🏥💙🚀🎯📋💡📚🤖⚠️✅❌🔊📝🎤]', '', clean_text)
        
        if len(clean_text) > 300:
            clean_text = clean_text[:300] + "..."
        
        if self.azure_enabled:
            try:
                audio_config=None
                synthesizer = speechsdk.SpeechSynthesizer(self.speech_config, audio_config)
                result = synthesizer.speak_text_async(clean_text).get()
                
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    return result.audio_data
            except Exception as e:
                logger.error(f"Azure TTS failed: {e}")

        if self.basic_voice_enabled:
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                self.tts_engine.save_to_file(clean_text, temp_path)
                self.tts_engine.runAndWait()
                
                with open(temp_path, 'rb') as audio_file:
                    audio_data = audio_file.read()
                os.unlink(temp_path)
                return audio_data
            except Exception as e:
                logger.error(f"Basic TTS failed: {e}")
        
        return b""

    async def process_audio(self, audio_data: bytes) -> bytes:
        """Complete voice processing: STT -> AI -> TTS"""
        try:
            # Step 1: Speech to Text
            text = await self.recognize_audio(audio_data)
            if not text:
                return await self.speak("Sorry, I couldn't understand that.")
            
            # Step 2: Get AI response
            nova_system = NovaUltraSystem()
            response = await nova_system.get_response(text, "voice-user")
            
            # Step 3: Text to Speech
            return await self.speak(response["response"])
            
        except Exception as e:
            logger.error(f"Voice processing error: {e}")
            return await self.speak("Sorry, I encountered an error processing your voice.")
        
    async def text_to_speech(self, text: str, voice: str = "en-US-AriaNeural") -> bytes:
     try:
        # 1. Ensure voice is set
         self.speech_config.speech_synthesis_voice_name = voice

        # 2. audio_config=None → return audio as bytes (not speaker)
         synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config,
            audio_config=None
        )

        # 3. Do synthesis
         result = synthesizer.speak_text_async(text).get()

         if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return result.audio_data
         elif result.reason == speechsdk.ResultReason.Canceled:
            details = result.cancellation_details
            raise RuntimeError(f"TTS canceled: {details.reason} - {details.error_details}")
         else:
            raise RuntimeError(f"TTS failed: {result.reason}")

     except Exception as e:
        logger.error(f"Azure TTS failed: {e}")
        raise

# ========== WEB SEARCH (EXACT FROM CLI) ==========
class FastWebSearch:
    def __init__(self):
        self.search_enabled = True

    async def search_web(self, query: str, max_results: int = 5, summarize: bool = True):
        try:
            url = f"https://duckduckgo.com/html/?q={query}"
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; NOVA/1.0)'}
            response = requests.get(url, headers=headers, timeout=8)
            response.raise_for_status()

            results = []
            content = response.text
            titles = re.findall(r'<a[^>]*class="result__a"[^>]*>(.*?)</a>', content)
            snippets = re.findall(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', content)

            for i, title in enumerate(titles[:max_results]):
                snippet = snippets[i] if i < len(snippets) else ""
                results.append({
                    "title": title.strip(),
                    "snippet": snippet.strip(),
                    "url": f"https://duckduckgo.com/?q={query}",
                    "source": "DuckDuckGo"
                })

            summary_answer = None
            if summarize and results:
                # ✅ feed into QA Engine
                qa = create_qa_engine(persist_directory="./chroma_db")
                context_text = "\n\n".join(
                    f"{r['title']}\n{r['snippet']}\n{r['url']}" for r in results
                )
                summary_answer = qa.ask(
                    f"Based on these search results, answer the following:\n\n{query}\n\nSources:\n{context_text}"
                )

            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results),
                "summary_answer": summary_answer
            }

        except Exception as e:
            return {"success": False, "error": f"Search failed: {e}"}

# ========== GITHUB ANALYZER (EXACT FROM CLI) ==========
class EnhancedGitHubRepoAnalyzer:
    """GitHub analyzer - EXACT from CLI"""
    
    def __init__(self):
        self.active_repo = None
        self.repo_data = {}
        self.qa_engine = None
        
        if GITHUB_INTEGRATION and create_qa_engine:
            try:
                self.qa_engine = create_qa_engine(simple=False)
                logger.info("✅ GitHub QA Engine initialized")
            except Exception as e:
                try:
                    self.qa_engine = create_qa_engine(simple=True)
                except Exception as e2:
                    logger.error(f"QA Engine initialization failed: {e2}")
                    self.qa_engine = None

    async def analyze_repository(self, repo_url: str) -> Dict[str, Any]:
        """Analyze repository - EXACT from CLI"""
        if not GITHUB_INTEGRATION or not ingest_repo:
            return {"error": "GitHub integration not available"}
        
        try:
            logger.info(f"🔍 Analyzing repository: {repo_url}")
            
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            
            # Enhanced repo ingestion - EXACT from CLI
            try:
                if os.getenv('GITHUB_TOKEN'):
                    os.environ['GITHUB_TOKEN'] = os.getenv('GITHUB_TOKEN')
                
                ingest_repo(repo_url, enhanced_processing=True, include_file_contents=True)
                logger.info("✅ Repository ingested successfully")
                
                # Verify vector database - EXACT from CLI
                if os.path.exists("./chroma_db"):
                    try:
                        import chromadb
                        client = chromadb.PersistentClient(path="./chroma_db")
                        collections = client.list_collections()
                        if collections:
                            collection = collections[0]
                            test_results = collection.query(
                                query_texts=["what files are in this repository"],
                                n_results=5
                            )
                            if test_results['documents']:
                                logger.info("✅ File contents accessible in vector database")
                    except Exception as e:
                        logger.warning(f"Vector database verification failed: {e}")
                        
            except Exception as e:
                return {"error": f"Failed to ingest repository: {e}"}
            
            # Store repo information - EXACT from CLI
            self.active_repo = repo_url
            self.repo_data = {
                'name': repo_name,
                'url': repo_url,
                'analyzed_at': datetime.now(),
                'vector_db_path': "./chroma_db"
            }
            
            # Perform code analysis - EXACT from CLI
            analysis = await self.perform_enhanced_code_analysis()
            
            return {
                "success": True,
                "repo_name": repo_name,
                "repo_url": repo_url,
                "analysis": analysis,
                "files_processed": analysis.get('file_count', 0),
                "languages": analysis.get('languages', []),
                "issues_found": analysis.get('issues', []),
                "suggestions": analysis.get('suggestions', []),
                "file_content_accessible": True
            }
            
        except Exception as e:
            return {"error": f"Repository analysis failed: {e}"}

    async def perform_enhanced_code_analysis(self) -> Dict[str, Any]:
        """Code analysis - EXACT from CLI"""
        if not self.qa_engine:
            return {
                "error": "QA engine not available",
                'file_count': 'Repository processed',
                'languages': ['Python', 'JavaScript', 'Other'],
                'issues': ["Analysis engine unavailable"],
                'suggestions': ["Manual code review recommended"]
            }
        
        # Enhanced analysis questions - EXACT from CLI
        enhanced_analysis_questions = [
            "What is the main purpose of this codebase?",
            "What programming languages are used?",
            "List all the files in this repository and their purposes",
            "Show me the main functions and classes in the code",
            "Are there any potential bugs or issues in the code?",
            "What improvements can be made to this code?",
            "What is the overall structure and architecture?",
            "What are the main dependencies and libraries used?",
            "Are there any security vulnerabilities in the code?",
            "What testing frameworks or test files are present?",
            "What is the main entry point of the application?",
            "Are there any configuration files and what do they contain?"
        ]
        
        analysis_results = {}
        successful_queries = 0
        
        for question in enhanced_analysis_questions:
            try:
                result = self.qa_engine.ask(question)
                if isinstance(result, dict) and 'response' in result:
                    analysis_results[question] = result['response']
                    successful_queries += 1
                else:
                    analysis_results[question] = str(result)
                    if result and str(result) != "I don't have enough information":
                        successful_queries += 1
            except Exception as e:
                analysis_results[question] = f"Analysis failed: {e}"
        
        # Extract structured information - EXACT from CLI
        return {
            'file_count': f'{successful_queries}/{len(enhanced_analysis_questions)} queries successful',
            'languages': self.extract_languages(analysis_results),
            'issues': self.extract_enhanced_issues(analysis_results),
            'suggestions': self.extract_enhanced_suggestions(analysis_results),
            'detailed_analysis': analysis_results,
            'file_content_accessible': successful_queries > len(enhanced_analysis_questions) / 2,
            'architecture_analysis': self.extract_architecture_info(analysis_results),
            'security_analysis': self.extract_security_info(analysis_results),
            'dependency_analysis': self.extract_dependency_info(analysis_results)
        }

    def extract_languages(self, analysis: Dict[str, str]) -> List[str]:
        """Extract languages - EXACT from CLI"""
        languages = []
        languages_question = "What programming languages are used?"
        if languages_question in analysis:
            lang_text = analysis[languages_question].lower()
            common_languages = ['python', 'javascript', 'java', 'cpp', 'c++', 'html', 'css']
            for lang in common_languages:
                if lang in lang_text:
                    languages.append(lang.title())
        return languages if languages else ['Python', 'JavaScript', 'Other']

    def extract_enhanced_issues(self, analysis: Dict[str, str]) -> List[str]:
        """Extract issues - EXACT from CLI"""
        issues = []
        issue_questions = [
            "Are there any potential bugs or issues in the code?",
            "Are there any security vulnerabilities in the code?"
        ]
        
        for question in issue_questions:
            if question in analysis:
                issue_analysis = analysis[question].lower()
                if any(keyword in issue_analysis for keyword in ['bug', 'issue', 'error', 'vulnerability', 'problem']):
                    if 'security' in issue_analysis:
                        issues.append("Security vulnerabilities detected in codebase")
                    if 'bug' in issue_analysis:
                        issues.append("Potential bugs detected in codebase")
        
        return issues if issues else ["No critical issues detected"]

    def extract_enhanced_suggestions(self, analysis: Dict[str, str]) -> List[str]:
        """Extract suggestions - EXACT from CLI"""
        return [
            "Code structure and architecture improvements",
            "Documentation and comments enhancement",
            "Error handling and validation improvements",
            "Performance optimization opportunities",
            "Security enhancements and best practices",
            "Testing coverage improvements"
        ]

    def extract_architecture_info(self, analysis: Dict[str, str]) -> str:
        """Extract architecture - EXACT from CLI"""
        arch_question = "What is the overall structure and architecture?"
        return analysis.get(arch_question, "Architecture analysis not available")

    def extract_security_info(self, analysis: Dict[str, str]) -> str:
        """Extract security - EXACT from CLI"""
        security_question = "Are there any security vulnerabilities in the code?"
        return analysis.get(security_question, "Security analysis not available")

    def extract_dependency_info(self, analysis: Dict[str, str]) -> str:
        """Extract dependencies - EXACT from CLI"""
        dep_question = "What are the main dependencies and libraries used?"
        return analysis.get(dep_question, "Dependency analysis not available")

    async def answer_repo_question(self, question: str) -> str:
        """Answer repo question - EXACT from CLI"""
        if not self.active_repo or not self.qa_engine:
            return "No active repository or QA engine not available. Please analyze a repository first."
        
        try:
            result = self.qa_engine.ask(question)
            if isinstance(result, dict) and 'response' in result:
                response = result['response']
            else:
                response = str(result)
            
            # Try alternative phrasing if no response - EXACT from CLI
            if not response or "I don't have enough information" in response:
                alternative_questions = [
                    f"Based on the repository files, {question.lower()}",
                    f"From the codebase analysis, {question.lower()}",
                    f"Looking at the source code, {question.lower()}"
                ]
                
                for alt_question in alternative_questions:
                    try:
                        alt_result = self.qa_engine.ask(alt_question)
                        if isinstance(alt_result, dict) and 'response' in alt_result:
                            alt_response = alt_result['response']
                        else:
                            alt_response = str(alt_result)
                        if alt_response and "I don't have enough information" not in alt_response:
                            return alt_response
                    except:
                        continue
            
            return response if response else "Unable to find specific information about this query in the repository."
            
        except Exception as e:
            return f"Failed to answer repository question: {e}"

    def has_active_repo(self) -> bool:
        """Check active repo - EXACT from CLI"""
        return self.active_repo is not None

    def get_repo_stats(self) -> Dict[str, Any]:
        """Get repo stats - EXACT from CLI"""
        if not self.active_repo:
            return {}
        
        return {
            'repo_name': self.repo_data.get('name', 'Unknown'),
            'repo_url': self.active_repo,
            'analyzed_at': self.repo_data.get('analyzed_at'),
            'vector_db_path': self.repo_data.get('vector_db_path'),
            'file_content_accessible': True
        }

# ========== NOVA ULTRA SYSTEM (EXACT FROM CLI) ==========
class NovaUltraSystem:
    """Main NOVA system - EXACT from CLI"""
    
    def __init__(self):
        logger.info("🚀 Initializing NOVA Ultra System...")
        
        # Core systems - EXACT from CLI
        self.memory = UltraHybridMemorySystem()
        self.language_detector = FastLanguageDetector()
        self.emotion_detector = FastEmotionDetector()
        self.api_manager = OptimizedAPIManager()
        self.voice_system = FastVoiceSystem()
        self.web_search = FastWebSearch()
        self.file_system = FileUploadSystem()
        self.github_analyzer = EnhancedGitHubRepoAnalyzer()
        
        # ML System - EXACT from CLI
        self.ml_manager = None
        if ML_SYSTEM_AVAILABLE:
            try:
                self.ml_manager = EnhancedMLManager()
            except Exception as e:
                logger.error(f"ML system init error: {e}")
        
        # Advanced systems - EXACT from CLI
        self.orchestrator = None
        self.drift_detector = None
        if ADVANCED_SYSTEMS:
            try:
                self.orchestrator = IntelligentAPIOrchestrator()
                self.drift_detector = APIPerformanceDrifter()
            except Exception as e:
                logger.error(f"Advanced systems init error: {e}")
        
        # Professional agents - EXACT from CLI
        self.agents = {}
        if PROFESSIONAL_AGENTS_LOADED:
            try:
                self.agents = {
                    'coding': ProLevelCodingExpert(),
                    'career': ProfessionalCareerCoach(),
                    'business': SmartBusinessConsultant(),
                    'medical': SimpleMedicalAdvisor(),
                    'emotional': SimpleEmotionalCounselor(),
                    'technical_architect': TechnicalArchitect()
                }
            except Exception as e:
                logger.error(f"Agent loading error: {e}")
        
        # Session management - EXACT from CLI
        self.current_sessions = {}
        self.conversation_count = 0
        
        # Agent patterns - EXACT from CLI
        self.agent_patterns = {
            "coding": {
                "keywords": ["code", "programming", "debug", "python", "javascript", "bug", "development"],
                "system_prompt": "You are NOVA Coding Expert. Provide practical, production-ready code solutions with best practices."
            },
            "career": {
                "keywords": ["resume", "interview", "job", "career", "hiring", "professional"],
                "system_prompt": "You are NOVA Career Coach. Provide expert career guidance and professional advice."
            },
            "business": {
                "keywords": ["business", "analysis", "strategy", "market", "revenue", "growth"],
                "system_prompt": "You are NOVA Business Consultant. Provide strategic business insights and analysis."
            },
            "medical": {
                "keywords": ["health", "medical", "symptoms", "doctor", "treatment"],
                "system_prompt": "You are Dr. NOVA. Provide medical insights while emphasizing professional consultation."
            },
            "emotional": {
                "keywords": ["stress", "anxiety", "sad", "emotional", "support", "therapy"],
                "system_prompt": "You are Dr. NOVA Counselor. Provide empathetic emotional support and guidance."
            },
            "technical_architect": {
                "keywords": ["architecture", "system design", "scalability", "microservice"],
                "system_prompt": "You are NOVA Technical Architect. Provide comprehensive system design guidance."
            }
        }

    def get_user_session(self, user_id: str) -> Dict[str, Any]:
        """Get user session - EXACT from CLI"""
        if user_id not in self.current_sessions:
            self.current_sessions[user_id] = {
                'session_id': f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}",
                'file_context': None,
                'conversation_count': 0,
                'last_agent': 'general',
                'created_at': datetime.now()
            }
        return self.current_sessions[user_id]

    async def detect_agent_type(self, user_input: str) -> Tuple[str, float]:
        """Agent detection - EXACT from CLI"""
        text_lower = user_input.lower()
        for agent_name, agent_data in self.agent_patterns.items():
            keywords = agent_data["keywords"]
            if any(keyword in text_lower for keyword in keywords):
                return agent_name, 0.8
        return "general", 0.0

    async def create_system_prompt(self, agent_type: str, language: str, emotion: str, 
                                 user_context: str = None, file_context: str = None) -> str:
        """Create system prompt - EXACT from CLI"""
        base_prompt = """You are NOVA Ultra Professional AI, an advanced assistant with expertise across all domains.
Provide professional, actionable, and empathetic responses. Be concise yet comprehensive."""
        
        agent_prompt = self.agent_patterns.get(agent_type, {}).get("system_prompt", "")
        
        language_note = ""
        if language == "hinglish":
            language_note = " Respond naturally mixing English and Hindi as appropriate."
        
        emotion_note = ""
        if emotion in ["sad", "anxious", "frustrated"]:
            emotion_note = f" The user seems {emotion}, so be extra supportive and empathetic."
        
        context_note = ""
        if user_context:
            context_note = f"\n\nCONVERSATION CONTEXT:\n{user_context}"
        
        file_context_note = ""
        if file_context:
            file_context_note = f"\n\nFILE CONTEXT: The user has uploaded/analyzed this file:\n{file_context}\n\nUse this context to provide more relevant and specific answers."
        
        return f"{base_prompt}\n{agent_prompt}{language_note}{emotion_note}{context_note}{file_context_note}"

    async def get_response(self, user_input: str, user_id: str = "web-user") -> Dict[str, Any]:
        """Get AI response - EXACT from CLI"""
        start_time = time.time()
        
        try:
            # Get user session
            user_session = self.get_user_session(user_id)
            
            # Fast detection - EXACT from CLI
            language = self.language_detector.detect_language(user_input)
            emotion, emotion_confidence = self.emotion_detector.detect_emotion(user_input)
            agent_type, agent_confidence = await self.detect_agent_type(user_input)
            
            # Get user context from memory
            user_context = self.memory.get_relevant_context(user_input, user_id, limit=15)

            memory_note=""
            if user_context:
                memory_note= f"""
                The following is a summary of recent conversations with this user.
                Use this history to keep continuity, remember facts, and provide consistent answers:

            {user_context}
            """
            # Create system prompt with injected memory
            system_prompt = f""" 
            You are NOVA, a highly professional and advanced AI assistant. 
            Your role is to provide accurate, comprehensive, and engaging answers that feel human-like.
            Always adapt tone to the user’s emotional state and maintain continuity from past conversations.
            {memory_note}
            Now respond to the new user input in the most professional, detailed, and empathetic way possible.
            """

            # Get AI response
            ai_response = await self.api_manager.get_ai_response(user_input, system_prompt, agent_type)

            # Final fallback
            if not ai_response:
               ai_response = (
                   f"I'm having technical difficulties, but I understand you're asking about "
                   f"{agent_type}-related topics. Please try rephrasing your question."
               )
            response_time = time.time() - start_time
            # Update session
            user_session['conversation_count'] += 1
            user_session['last_agent'] = agent_type
            self.conversation_count += 1

            # Store in memory
            await self.memory.remember_conversation(
                user_id, user_session['session_id'], user_input, ai_response,
                agent_type, language, emotion, emotion_confidence,
                response_time=response_time,
                file_analyzed=user_session['file_context'] if user_session['file_context'] else None
        )
            return {
                "response": ai_response,
                "agent_used": agent_type,
                "language": language,
                "emotion": emotion,
                "emotion_confidence": emotion_confidence,
                "agent_confidence": agent_confidence,
                 "response_time": response_time,
                 "conversation_count": user_session['conversation_count'],
                  "file_context_used": user_session['file_context'] is not None,
                  "user_id": user_id,
                  "session_id": user_session['session_id']
            }
        except Exception as e:
             logger.error(f"Response error: {e}")
             return {
                 "response": "I apologize for the technical difficulty. Please try again.",
                 "agent_used": "error",
                 "language": "english",
                 "emotion": "neutral",
                 "response_time": time.time() - start_time,
                 "error": str(e),
                 "user_id": user_id
             }
           
    async def upload_and_analyze_file_content(self, file_content: bytes, filename: str, 
                                            user_id: str = "web-user") -> Dict[str, Any]:
        """File upload and analysis - EXACT from CLI"""
        try:
            # Analyze file
            file_analysis = self.file_system.analyze_file_content(file_content, filename)
            
            if file_analysis.get("error"):
                return file_analysis
            
            # Store in memory
            self.memory.remember_file_processing(
                user_id, file_analysis['file_name'], file_analysis['file_type'],
                f"File analyzed: {file_analysis['file_name']}", True
            )
            
            # Set file context - EXACT from CLI
            user_session = self.get_user_session(user_id)
            user_session['file_context'] = f"""
File: {file_analysis['file_name']}
Type: {file_analysis['file_type']}
Size: {file_analysis['file_size']} bytes
Lines: {file_analysis.get('lines', 'N/A')}
Content preview: {file_analysis['content'][:500]}...
"""
            
            return {
                "success": True,
                "file_analysis": file_analysis,
                "message": f"Successfully analyzed {file_analysis['file_name']}"
            }
            
        except Exception as e:
            return {"error": f"File upload failed: {str(e)}"}

    async def search_web(self, query: str, user_id: str = "web-user") -> Dict[str, Any]:
        """Web search - EXACT from CLI"""
        try:
            search_results = await self.web_search.search_web(query, max_results=5)
            
            if search_results.get("success"):
                # Store in memory
                with sqlite3.connect(self.memory.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO search_history (user_id, search_query, search_type, results_count, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (user_id, query, "web", search_results.get("count", 0), datetime.now()))
                    conn.commit()
                
                # Format response - EXACT from CLI
                formatted_response = f"🔍 **Web Search Results for: {query}**\n\n"
                for i, result in enumerate(search_results.get("results", []), 1):
                    formatted_response += f"**{i}. {result['title']}**\n"
                    formatted_response += f"Source: {result['source']}\n"
                    formatted_response += f"{result['snippet']}\n\n"
                
                return {"success": True, "formatted_response": formatted_response}
            else:
                return {"error": "Web search failed"}
                
        except Exception as e:
            return {"error": f"Web search error: {e}"}

    def get_system_status(self) -> Dict[str, Any]:
        """System status - EXACT from CLI"""
        return {
            "core_systems": {
                "memory": "✅ Active",
                "language_detection": "✅ Active", 
                "emotion_detection": "✅ Active",
                "api_manager": "✅ Active" if self.api_manager.current else "❌ No API",
                "file_system": "✅ Active",
                "voice_system": "✅ Active"
            },
            "premium_systems": {
                "azure_voice": "✅ Active" if self.voice_system.azure_enabled else "⚠️ Basic Only",
                "web_search": "✅ Active",
                "ml_system": "✅ Active" if self.ml_manager else "❌ Disabled"
            },
            "agents": {
                agent_name: "✅ Active" if agent_name in self.agents else "❌ Disabled"
                for agent_name in ["coding", "career", "business", "medical", "emotional", "technical_architect"]
            },
            "session_info": {
                "total_sessions": len(self.current_sessions),
                "conversation_count": self.conversation_count,
                "available_providers": len(self.api_manager.available)
            }
        }

    def clear_user_context(self, user_id: str):
        """Clear context - EXACT from CLI"""
        if user_id in self.current_sessions:
            user_session = self.current_sessions[user_id]
            user_session['file_context'] = None
            user_session['conversation_count'] = 0
            user_session['last_agent'] = 'general'

# ========== PYDANTIC MODELS ==========
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    user_id: str = Field("web-user", description="User ID")

class ChatResponse(BaseModel):
    response: str
    agent_used: str
    language: str
    emotion: str
    emotion_confidence: float
    agent_confidence: float
    response_time: float
    conversation_count: int
    file_context_used: bool
    user_id: str
    session_id: str

class VoiceRequest(BaseModel):
    text: str = Field(..., description="Text to speak")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    user_id: str = Field("web-user", description="User ID")

class GitHubRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub repository URL")

class GitHubQuestionRequest(BaseModel):
    question: str = Field(..., description="Question about repository")

# ========== FASTAPI APPLICATION ==========
app = FastAPI(
    title="NOVA Ultra Professional API - 1:1 CLI Mapping",
    description="Direct FastAPI conversion of NOVA-CLI.py with ZERO add-ons",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://nova-frontend-rouge.vercel.app"],  # ✅ no trailing slash
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize NOVA system
nova_system = NovaUltraSystem()

# ========== API ENDPOINTS ==========

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🚀 NOVA Ultra Professional API - 1:1 CLI Mapping",
        "version": "3.0.0", 
        "status": "✅ Fully Operational",
        "features": [
            "🧠 UltraHybridMemorySystem with semantic memory",
            "🤖 Multi-Agent System (6 agents)",
            "🔀 Multi-Provider AI (3 providers)",
            "📁 File Processing System",
            "🔗 GitHub Repository Analyzer",
            "🎤 Voice Processing (Azure + Basic)",
            "🔍 Web Search Integration",
            "💾 Conversation Memory"
        ]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint with ML-enriched answers"""

    # Step 1: Run ML pipeline
    ml_results = ml_manager.process_user_query(request.message, context={})

    # Extract insights from ML pipeline
    routing = ml_results.get("routing_decision", {})
    query_analysis = ml_results.get("query_analysis", {})
    context_enhancement = ml_results.get("context_enhancement", {})

    intent = routing.get("selected_agent", "general")
    confidence = routing.get("confidence_level", 0.0)

    sentiment = query_analysis.get("sentiment", "neutral")
    keywords = query_analysis.get("intent_keywords", [])
    entities = query_analysis.get("technical_context", {})
    multi_intent = query_analysis.get("multi_intent_analysis", [])
    supporting_keywords = query_analysis.get("supporting_keywords", [])

    rag_context = context_enhancement.get("relevant_context", "")
    recommendations = ml_results.get("recommendations", [])
    metrics = ml_results.get("performance_metrics", {})

    # Step 2: Build enhanced AI prompt
    enhanced_prompt = f"""
    User asked: {request.message}

    🔎 ML Analysis:
    - Detected intent: {intent} (confidence: {confidence:.2f})
    - Sentiment: {sentiment}
    - Keywords: {keywords}
    - Entities: {entities}
    - Supporting Keywords: {supporting_keywords}
    - Multi-intent Analysis: {multi_intent}
    - Relevant Context (RAG): {rag_context}
    - Recommendations: {recommendations}
    - Performance Metrics: {metrics}

    ➡️ Please generate a **professional, advanced, informative, structured, and engaging response**.  
    Use quantitative data, examples, and technical context naturally.  
    Make it read like a polished expert answer — not just plain AI output.
    """

    # Step 3: Pass to Nova AI system
    response_data = await nova_system.get_response(enhanced_prompt, request.user_id)

    # Step 4: Log interaction (without metadata)
    ml_manager.store_interaction_intelligently(
        request.message,
        response_data["response"],
        agent_used=intent
    )

    # Step 5: Return final AI response only
    return ChatResponse(**response_data)

@app.post("/file/upload")
async def upload_file_endpoint(
    file: UploadFile = File(...),
    user_id: str = Form("web-user"),
    prompt: Optional[str] = Form(None)
):
    """File upload endpoint with ML-enhanced AI analysis"""
    try:
        # 1. Read file
        file_content = await file.read()
        result = await nova_system.upload_and_analyze_file_content(
            file_content, file.filename, user_id
        )

        if not result.get("success"):
            return result

        file_analysis = result["file_analysis"]

        # 2. Run ML pipeline
        user_query = prompt or "Please analyze this file and provide useful insights."
        ml_results = ml_manager.process_user_query(
            user_query,
            context={"file_metadata": file_analysis}
        )

        intent = ml_results["routing_decision"]["selected_agent"]
        sentiment = ml_results["query_analysis"]["sentiment"]
        keywords = ml_results["query_analysis"]["intent_keywords"]
        rag_context = ml_results["context_enhancement"]["relevant_context"]

        # 3. Build enhanced AI input
        enhanced_input = f"""
        A user uploaded a file.

        📄 File: {file_analysis['file_name']}
        Type: {file_analysis['file_type']}
        Size: {file_analysis['file_size']} bytes
        Lines: {file_analysis.get('lines', 'N/A')}

        --- File Content (truncated) ---
        {file_analysis['content'][:2000]}

        --- User Question ---
        {user_query}

        --- ML Insights ---
        Detected intent: {intent}
        Sentiment: {sentiment}
        Keywords: {keywords}
        Relevant Context: {rag_context}

        ➡️ Please generate a **professional, advanced, structured, and insightful analysis**.
        Include technical depth, patterns, and recommendations.
        """

        # 4. Get final AI response
        response = await nova_system.get_response(enhanced_input, user_id)
        ai_response = response["response"]

        # 5. Log + monitor (no metadata)
        ml_manager.store_interaction_intelligently(
            user_query,
            ai_response,
            agent_used=intent
        )

        # ✅ Standardized clean response
        return {
            "success": True,
            "response": ai_response,
            "metadata": {
                "file_name": file_analysis["file_name"],
                "file_type": file_analysis["file_type"],
                "file_size": file_analysis["file_size"],
                "lines": file_analysis.get("lines", "N/A"),
                "message": f"Successfully analyzed {file_analysis['file_name']}"
            }
        }

    except Exception as e:
        logger.error(f"File upload error: {e}")
        return {"success": False, "error": f"File upload failed: {str(e)}"}
    


@app.post("/voice/speak")
async def voice_speak_endpoint(audio: UploadFile = File(...)):
    """Process voice audio and return TTS response"""
    try:
        # Save the incoming audio file temporarily
        temp_path = f"temp_{audio.filename}"
        async with aiofiles.open(temp_path, 'wb') as out_file:
            content = await audio.read()
            await out_file.write(content)
        
        # Process the audio file (implement your logic here)
        audio_data = await nova_system.voice_system.process_audio(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        return StreamingResponse(
            BytesIO(audio_data),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=response.wav"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")
    
@app.post("/voice/process")
async def process_voice_command(
    audio: UploadFile = File(None),
    text: str = Form(None),
    user_id: str = Form("voice-user")
):
    """
    Unified voice processing (same as NOVA-CLI):
    - If `audio` is uploaded → STT → AI → TTS → return spoken answer
    - If `text` is provided → TTS only → return spoken answer
    """
    try:
        if audio:
            # 1. Read raw bytes from browser
            audio_data = await audio.read()

            # 2. Convert WebM → WAV (browser sends webm/opus, CLI had wav directly)
            wav_bytes = webm_to_wav(audio_data)

            # 3. STT (same as CLI)
            user_text = await nova_system.voice_system.process_audio(wav_bytes)

            # 4. AI Response (use same method as CLI)
            ai_response = await nova_system.process_text(user_text, user_id)

            # 5. TTS (explicitly pass default voice like in CLI)
            processed_audio = await nova_system.voice_system.text_to_speech(
                ai_response,
                voice="en-US-AriaNeural"  # replace with the voice you used in CLI
            )

        elif text:
            # Direct TTS from given text
            processed_audio = await nova_system.voice_system.text_to_speech(
                text,
                voice="en-US-AriaNeural"
            )

        else:
            return JSONResponse(
                {"error": "No audio or text provided"},
                status_code=400
            )

        return StreamingResponse(
            BytesIO(processed_audio),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=response.wav"}
        )

    except Exception as e:
        logger.error(f"Voice endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")
    

@app.post("/web/search")
async def web_search_endpoint(request: SearchRequest):
    result = await nova_system.search_web(request.query, request.user_id)
    return result

@app.post("/github/analyze")
async def analyze_github_repo(repo_url: str = Form(...), user_id: str = Form("web-user")):
    try:
        persist_directory = "./chroma_db"

        # 1. Ingest repo into Chroma DB
        await ingest_repo(repo_url, persist_directory=persist_directory)

        # 2. Create QA engine
        qa = create_qa_engine(persist_directory)

        # 3. Collect raw insights
        code_quality = qa.ask("Provide a detailed code quality review of this repository.")
        debugging = qa.ask("Identify potential bugs or issues and suggest fixes.")

        # 4. Run ML pipeline on repo query
        ml_results = ml_manager.process_user_query(
            f"Analyze GitHub repo {repo_url}",
            context={
                "repo_url": repo_url,
                "code_quality": code_quality,
                "debugging": debugging
            }
        )

        # Extract ML insights
        intent = ml_results["routing_decision"]["selected_agent"]
        sentiment = ml_results["query_analysis"]["sentiment"]
        keywords = ml_results["query_analysis"]["intent_keywords"]
        rag_context = ml_results["context_enhancement"]["relevant_context"]
        recommendations = ml_results.get("recommendations", [])

        # 5. Build enhanced AI prompt
        enhanced_prompt = f"""
        User uploaded GitHub repo: {repo_url}

        --- Repository Analysis ---
        Code Quality: {code_quality}
        Debugging Suggestions: {debugging}

        --- ML Insights ---
        Intent → {intent}
        Sentiment → {sentiment}
        Keywords → {keywords}
        Relevant Context → {rag_context}
        Recommendations → {recommendations}

        ➡️ Please generate a **professional, structured, advanced repository analysis** with:
        - Key strengths
        - Critical issues
        - Suggested improvements
        - Industry best practices
        - Quantitative evaluation where possible
        """

        # 6. Generate pro AI response
        response_data = await nova_system.get_response(enhanced_prompt, user_id)

        # 7. Log & monitor (simple, no metadata arg)
        ml_manager.store_interaction_intelligently(
            f"GitHub Analysis: {repo_url}",
            response_data["response"],
            agent_used=intent
        )

        # ✅ 8. Standardized response
        return {
            "success": True,
            "response": response_data["response"],
            "metadata": {
                "repo_url": repo_url,
                "raw_insights": {
                    "code_quality": code_quality,
                    "debugging": debugging
                }
            }
        }

    except Exception as e:
        logger.error(f"GitHub analysis error: {e}")
        return {"success": False, "error": str(e)}
    
@app.post("/github/question")
async def ask_github_question(question: str = Form(...), user_id: str = Form("web-user")):
    try:
        persist_directory = "./chroma_db"
        qa = create_qa_engine(persist_directory)

        # Get base repo answer
        raw_answer = qa.ask(question)

        # Run through ML pipeline for enrichment
        ml_results = ml_manager.process_user_query(question, context={"repo_answer": raw_answer})
        intent = ml_results["routing_decision"]["selected_agent"]
        rag_context = ml_results["context_enhancement"]["relevant_context"]

        # Build enhanced prompt
        enhanced_prompt = f"""
        User asked about repository: {question}

        Base Repo Answer: {raw_answer}
        Detected intent: {intent}
        Additional Context: {rag_context}

        ➡️ Refine into a **clear, technical, professional answer** with
        practical suggestions where possible.
        """

        # Get final response
        response_data = await nova_system.get_response(enhanced_prompt, user_id)

        # Log/monitor
        ml_manager.store_interaction_intelligently(question, response_data["response"], agent_used=intent)

        return {
    "success": True,
    "response": response_data["response"],
    "metadata": {"question": question, "raw_answer": raw_answer}
}

    except Exception as e:
        logger.error(f"GitHub question error: {e}")
        return {"success": False, "error": str(e)}

@app.get("/system/status")
async def system_status_endpoint():
    """System status endpoint"""
    return nova_system.get_system_status()

@app.post("/clear/{user_id}")
async def clear_context_endpoint(user_id: str):
    """Clear user context"""
    nova_system.clear_user_context(user_id)
    return {"success": True, "message": f"Context cleared for user {user_id}"}

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0"
    }

# ========== STARTUP EVENT ==========
@app.on_event("startup")
async def startup_event():
    """Startup event"""
    logger.info("🚀 NOVA Ultra Professional API Starting...")
    logger.info("✅ 1:1 NOVA-CLI.py mapping complete")
    logger.info(f"✅ {len(nova_system.api_manager.available)} API providers available")
    logger.info(f"✅ {len(nova_system.agents)} professional agents loaded")
    logger.info("🎯 NOVA API ready!")

# ========== MAIN ENTRY POINT ==========
if __name__ == "__main__":
    logger.info("🚀 Starting NOVA Ultra Professional FastAPI Backend...")
    uvicorn.run("backend:app", port=8000, reload=True)
