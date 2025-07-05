import os
import json
import base64
import asyncio
import tempfile
import redis.asyncio as redis
from typing import Dict, Optional, AsyncGenerator, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

# Whisper for ASR
import whisper
import torch
from io import BytesIO
import numpy as np

# Chatterbox for TTS
try:
    from chatterbox import TTSEngine
    CHATTERBOX_AVAILABLE = True
except ImportError:
    CHATTERBOX_AVAILABLE = False
    logging.warning("Chatterbox not available. Install with: pip install chatterbox-tts")

# Fallback TTS options
from gtts import gTTS
import pyttsx3

# For conversation
import google.generativeai as genai
from app.config import settings

# Load environment variables
load_dotenv()

APP_NAME = "Voice Assistant"
SESSION_EXPIRY = 3600  # 1 hour

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperChatterboxService:
    """
    Voice service using Whisper for ASR and Chatterbox for TTS, 
    replacing Google ADK while maintaining Redis session management.
    """
    
    def __init__(self):
        """Initialize the Whisper + Chatterbox voice service."""
        logger.info("🎤 Initializing Whisper + Chatterbox Voice Service")
        
        # Initialize Redis connection for session management
        if hasattr(settings, 'REDIS_URL') and settings.REDIS_URL:
            self.redis_client = redis.from_url(settings.REDIS_URL, decode_responses=False)
        else:
            self.redis_client = redis.Redis(
                host=getattr(settings, 'REDIS_HOST', 'localhost'),
                port=getattr(settings, 'REDIS_PORT', 6379),
                password=getattr(settings, 'REDIS_PASSWORD', None),
                decode_responses=False
            )
        
        # Initialize Whisper model
        self._initialize_whisper()
        
        # Initialize TTS engine
        self._initialize_tts()
        
        # Initialize Gemini for conversation
        self._initialize_gemini()
        
        # Active sessions
        self.active_sessions = {}
        
        logger.info("✅ Whisper + Chatterbox Voice Service initialized successfully")
    
    def _initialize_whisper(self):
        """Initialize Whisper ASR model."""
        try:
            # Choose model size based on requirements
            model_size = os.getenv('WHISPER_MODEL_SIZE', 'base')  # tiny, base, small, medium, large
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"🔧 Loading Whisper model '{model_size}' on device '{device}'")
            
            self.whisper_model = whisper.load_model(model_size, device=device)
            self.whisper_device = device
            
            logger.info(f"✅ Whisper model '{model_size}' loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Whisper: {e}")
            raise
    
    def _initialize_tts(self):
        """Initialize TTS engine (Chatterbox or fallback)."""
        try:
            if CHATTERBOX_AVAILABLE:
                # Initialize Chatterbox TTS
                self.tts_engine = TTSEngine()
                self.tts_type = "chatterbox"
                logger.info("✅ Chatterbox TTS engine initialized")
            else:
                # Fallback to pyttsx3
                self.tts_engine = pyttsx3.init()
                self.tts_type = "pyttsx3"
                
                # Configure pyttsx3 settings
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    self.tts_engine.setProperty('voice', voices[0].id)
                self.tts_engine.setProperty('rate', 150)  # Speech rate
                self.tts_engine.setProperty('volume', 0.8)  # Volume
                
                logger.info("✅ pyttsx3 TTS engine initialized (fallback)")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize TTS: {e}")
            # Final fallback to gTTS
            self.tts_type = "gtts"
            logger.info("✅ Using gTTS as final fallback")
    
    def _initialize_gemini(self):
        """Initialize Gemini for conversation."""
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel(
                model_name=settings.GEMINI_MODEL,
                system_instruction="You are a helpful voice assistant. Respond naturally and conversationally. Keep responses concise and appropriate for voice interaction."
            )
            logger.info("✅ Gemini conversation model initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini: {e}")
            raise
    
    async def create_session(self, user_id: str, is_audio: bool = True) -> Dict[str, Any]:
        """
        Create a new voice session with Redis state management.
        
        Args:
            user_id: Unique identifier for the user
            is_audio: Whether to use audio mode (default: True)
            
        Returns:
            Session information dictionary
        """
        try:
            session_id = f"whisper_voice_{user_id}_{int(datetime.now().timestamp())}"
            
            # Create session data for Redis
            session_data = {
                "session_id": session_id,
                "user_id": user_id,
                "is_audio": is_audio,
                "created_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat(),
                "status": "active",
                "conversation_history": [],
                "service_type": "whisper_chatterbox"
            }
            
            # Store in Redis with expiration
            redis_key = f"voice_session:{session_id}"
            session_json = json.dumps(session_data)
            logger.info(f"🔍 Storing session in Redis key: {redis_key}")
            await self.redis_client.setex(
                redis_key,
                SESSION_EXPIRY,
                session_json
            )
            
            # Store in active sessions
            self.active_sessions[session_id] = {
                "user_id": user_id,
                "is_audio": is_audio,
                "conversation_history": []
            }
            
            logger.info(f"✅ Created Whisper+Chatterbox session: {session_id} for user: {user_id}")
            return session_data
            
        except Exception as e:
            logger.error(f"❌ Error creating voice session: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data from Redis."""
        try:
            redis_key = f"voice_session:{session_id}"
            session_data = await self.redis_client.get(redis_key)
            if session_data:
                return json.loads(session_data.decode('utf-8'))
            return None
        except Exception as e:
            logger.error(f"❌ Error getting session from Redis: {e}")
            return None
    
    async def update_session(self, session_id: str, update_data: Dict[str, Any]) -> bool:
        """Update session data in Redis."""
        try:
            session_data = await self.get_session(session_id)
            if not session_data:
                return False
            
            # Update the data
            session_data.update(update_data)
            session_data["last_active"] = datetime.now().isoformat()
            
            # Save back to Redis
            await self.redis_client.setex(
                f"voice_session:{session_id}",
                SESSION_EXPIRY,
                json.dumps(session_data)
            )
            return True
        except Exception as e:
            logger.error(f"❌ Error updating session in Redis: {e}")
            return False
    
    async def transcribe_audio(self, audio_data: str) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper ASR.
        
        Args:
            audio_data: Base64 encoded PCM audio data
            
        Returns:
            Transcription result with text and confidence
        """
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data)
            
            # Convert PCM bytes to numpy array
            # Assuming 16-bit PCM, mono, 16kHz (adjust based on your frontend)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Normalize to [-1, 1] for Whisper
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                audio_float,
                fp16=False,
                language=None,  # Auto-detect language
                task="transcribe"
            )
            
            transcribed_text = result["text"].strip()
            confidence = result.get("confidence", 0.0)
            
            logger.info(f"🎤 Whisper transcription: '{transcribed_text}' (confidence: {confidence:.2f})")
            
            return {
                "text": transcribed_text,
                "confidence": confidence,
                "language": result.get("language", "unknown")
            }
            
        except Exception as e:
            logger.error(f"❌ Error in Whisper transcription: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def generate_speech(self, text: str) -> str:
        """
        Generate speech using Chatterbox TTS.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Base64 encoded audio data
        """
        try:
            if self.tts_type == "chatterbox":
                # Use Chatterbox TTS
                audio_data = await self._generate_chatterbox_speech(text)
            elif self.tts_type == "pyttsx3":
                # Use pyttsx3 TTS
                audio_data = await self._generate_pyttsx3_speech(text)
            else:
                # Use gTTS fallback
                audio_data = await self._generate_gtts_speech(text)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"❌ Error in speech generation: {e}")
            raise
    
    async def _generate_chatterbox_speech(self, text: str) -> str:
        """Generate speech using Chatterbox."""
        try:
            # Generate audio with Chatterbox
            audio_data = self.tts_engine.synthesize(text)
            
            # Convert to base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            logger.info(f"🔊 Chatterbox TTS generated {len(audio_data)} bytes")
            return audio_base64
            
        except Exception as e:
            logger.error(f"❌ Chatterbox TTS error: {e}")
            raise
    
    async def _generate_pyttsx3_speech(self, text: str) -> str:
        """Generate speech using pyttsx3."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Generate speech to file
            self.tts_engine.save_to_file(text, tmp_path)
            self.tts_engine.runAndWait()
            
            # Read file and encode to base64
            with open(tmp_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up
            os.unlink(tmp_path)
            
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            logger.info(f"🔊 pyttsx3 TTS generated {len(audio_data)} bytes")
            return audio_base64
            
        except Exception as e:
            logger.error(f"❌ pyttsx3 TTS error: {e}")
            raise
    
    async def _generate_gtts_speech(self, text: str) -> str:
        """Generate speech using gTTS."""
        try:
            # Generate TTS
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save to BytesIO
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            # Read and encode
            audio_data = audio_buffer.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            logger.info(f"🔊 gTTS generated {len(audio_data)} bytes")
            return audio_base64
            
        except Exception as e:
            logger.error(f"❌ gTTS error: {e}")
            raise
    
    async def process_conversation(self, text: str, conversation_history: list) -> str:
        """
        Process conversation using Gemini.
        
        Args:
            text: User input text
            conversation_history: Previous conversation history
            
        Returns:
            AI response text
        """
        try:
            # Prepare conversation context
            context = ""
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                role = msg.get("role", "user")
                content = msg.get("content", "")
                context += f"{role}: {content}\n"
            
            # Add current user message
            context += f"user: {text}\n"
            
            # Generate response
            response = self.gemini_model.generate_content(context)
            response_text = response.text.strip()
            
            logger.info(f"💬 Gemini response: '{response_text[:100]}...'")
            return response_text
            
        except Exception as e:
            logger.error(f"❌ Error in conversation processing: {e}")
            return "I'm sorry, I encountered an error while processing your request."
    
    async def send_message(self, session_id: str, content: str, mime_type: str = "text/plain") -> AsyncGenerator[Dict[str, Any], None]:
        """
        Send a message and stream the response using Whisper + Chatterbox.
        
        Args:
            session_id: Session identifier
            content: Message content (text or base64 audio)
            mime_type: Content type
            
        Yields:
            Response events
        """
        try:
            # Get session data
            session_data = await self.get_session(session_id)
            if not session_data:
                yield {"type": "error", "message": "Session not found"}
                return
            
            conversation_history = session_data.get("conversation_history", [])
            
            if mime_type == "audio/pcm":
                # Process audio input
                logger.info(f"🎤 Processing audio input for session {session_id}")
                
                # Step 1: Transcribe audio with Whisper
                transcription_result = await self.transcribe_audio(content)
                
                if transcription_result.get("error"):
                    yield {"type": "error", "message": f"Transcription failed: {transcription_result['error']}"}
                    return
                
                user_text = transcription_result["text"]
                if not user_text:
                    yield {"type": "error", "message": "No speech detected"}
                    return
                
                # Send transcription result
                yield {
                    "type": "transcription",
                    "text": user_text,
                    "confidence": transcription_result["confidence"]
                }
                
            elif mime_type == "text/plain":
                # Process text input
                user_text = content
                logger.info(f"💬 Processing text input: '{user_text}'")
            
            else:
                yield {"type": "error", "message": f"Unsupported mime type: {mime_type}"}
                return
            
            # Step 2: Generate conversation response
            logger.info(f"🤖 Generating response for: '{user_text}'")
            response_text = await self.process_conversation(user_text, conversation_history)
            
            # Send text response
            yield {
                "type": "text",
                "data": response_text,
                "partial": False
            }
            
            # Step 3: Generate speech for audio sessions
            if session_data.get("is_audio", False):
                logger.info(f"🔊 Generating speech for response")
                try:
                    audio_base64 = await self.generate_speech(response_text)
                    yield {
                        "type": "audio",
                        "data": audio_base64,
                        "mime_type": "audio/wav"  # Adjust based on TTS output
                    }
                except Exception as e:
                    logger.error(f"❌ Speech generation failed: {e}")
                    yield {"type": "error", "message": f"Speech generation failed: {str(e)}"}
            
            # Step 4: Update conversation history
            conversation_history.extend([
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": response_text}
            ])
            
            # Keep only last 10 exchanges
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]
            
            await self.update_session(session_id, {
                "conversation_history": conversation_history
            })
            
        except Exception as e:
            logger.error(f"❌ Error in send_message: {e}")
            yield {"type": "error", "message": str(e)}
    
    async def close_session(self, session_id: str) -> bool:
        """Close and clean up a session."""
        try:
            # Remove from active sessions
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # Remove from Redis
            await self.redis_client.delete(f"voice_session:{session_id}")
            
            logger.info(f"🔚 Closed session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error closing session: {e}")
            return False
    
    async def list_active_sessions(self) -> list[str]:
        """List all active session IDs."""
        try:
            keys = await self.redis_client.keys("voice_session:*")
            return [key.decode('utf-8').split(':')[1] for key in keys]
        except Exception as e:
            logger.error(f"❌ Error listing sessions: {e}")
            return []

# Global instance
whisper_chatterbox_service = WhisperChatterboxService() 