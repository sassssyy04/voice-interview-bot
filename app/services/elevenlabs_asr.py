"""ElevenLabs ASR service for speech recognition."""

import asyncio
import httpx
import time
from typing import Tuple, Dict, Any
from app.core.config import settings
from app.core.logger import logger


class ElevenLabsASRService:
    """ASR service using ElevenLabs Speech-to-Text (Scribe) API."""
    
    def __init__(self):
        """Initialize ElevenLabs ASR service."""
        self.api_key = settings.elevenlabs_api_key
        self.base_url = "https://api.elevenlabs.io/v1"
        
        if not self.api_key:
            logger.warning("ElevenLabs API key not found - ASR will not work")
        else:
            logger.info("ElevenLabs ASR service initialized successfully")
    
    async def transcribe_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """Transcribe audio and return text and confidence.
        
        Args:
            audio_data (bytes): Raw audio bytes
            
        Returns:
            Dict[str, Any]: {"text": str, "confidence": float}
        """
        text, confidence = await self.recognize_speech_from_audio(audio_data)
        return {"text": text, "confidence": confidence}
    
    async def recognize_speech_from_audio(self, audio_data: bytes) -> Tuple[str, float]:
        """Recognize speech from audio data using ElevenLabs Scribe.
        
        Args:
            audio_data (bytes): Raw audio data
            
        Returns:
            Tuple[str, float]: (recognized_text, confidence_score)
        """
        if not self.api_key:
            logger.warning("ElevenLabs API key not available, returning empty result")
            return "", 0.0
            
        start_time = time.time()
        
        try:
            # ElevenLabs Speech-to-Text API endpoint
            url = f"{self.base_url}/speech-to-text"
            
            headers = {
                "xi-api-key": self.api_key
            }
            
            # Prepare the multipart form data
            files = {
                "file": ("audio.wav", audio_data, "audio/wav")
            }
            
            # Optional parameters for better Hinglish support
            data = {
                "model_id": "scribe_v1",
                "language_code": "hi",  # Hindi for better Hinglish support
                "tag_audio_events": "false",  # We don't need laughter/applause tags
                "timestamp_granularity": "word",  # Get word-level timestamps
                "diarize": "false"  # Single speaker for now
            }
            
            logger.info(f"Sending {len(audio_data)} bytes to ElevenLabs STT")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract text and confidence from ElevenLabs response
                    text = result.get("text", "").strip()
                    language_probability = result.get("language_probability", 0.0)
                    
                    # Use language probability as confidence score
                    confidence = float(language_probability) if language_probability else 0.8
                    
                    processing_time = (time.time() - start_time) * 1000
                    
                    logger.info(f"ElevenLabs STT completed in {processing_time:.0f}ms")
                    logger.info(f"Transcribed: '{text}' (confidence: {confidence:.2f})")
                    
                    # Apply Hinglish text normalization
                    normalized_text = self.normalize_hinglish_text(text)
                    
                    return normalized_text, confidence
                
                else:
                    logger.error(f"ElevenLabs STT error: {response.status_code} - {response.text}")
                    return "", 0.0
                    
        except asyncio.TimeoutError:
            logger.error("ElevenLabs STT timeout")
            return "", 0.0
        except Exception as e:
            logger.error(f"ElevenLabs STT error: {e}")
            return "", 0.0
    
    def normalize_hinglish_text(self, text: str) -> str:
        """Normalize Hinglish text for better entity extraction.
        
        Args:
            text (str): Raw recognized text
            
        Returns:
            str: Normalized text
        """
        if not text:
            return text
        
        # Common Hinglish normalizations for better entity extraction
        normalizations = {
            # Time expressions
            "aaj": "today",
            "kal": "tomorrow", 
            "parso": "day after tomorrow",
            "abhi": "now",
            "jaldi": "soon",
            
            # Numbers in Hindi (common in Hinglish)
            "ek": "one", "do": "two", "teen": "three", "char": "four", "panch": "five",
            "che": "six", "saat": "seven", "aath": "eight", "nau": "nine", "das": "ten",
            "gyarah": "eleven", "barah": "twelve", "terah": "thirteen", "chaudah": "fourteen", "pandrah": "fifteen",
            "solah": "sixteen", "satrah": "seventeen", "athrah": "eighteen", "unnis": "nineteen", "bees": "twenty",
            "hazaar": "thousand", "hazar": "thousand", "lakh": "lakh", "crore": "crore",
            
            # Yes/No responses
            "haan": "yes", "han": "yes", "ji haan": "yes", "bilkul": "yes",
            "nahi": "no", "nahin": "no", "na": "no", "nope": "no",
            
            # Common words for entity extraction
            "paisa": "money", "rupaye": "rupees", "rupees": "rupees", "salary": "salary",
            "morning": "morning", "evening": "evening", "night": "night", "subah": "morning", "shaam": "evening", "raat": "night",
            "bike": "two wheeler", "scooter": "two wheeler", "motorcycle": "two wheeler", "cycle": "bicycle",
            "language": "language", "languages": "languages", "bolna": "speak", "bol": "speak",
            "experience": "experience", "experience": "experience", "kaam": "work", "job": "job",
            "area": "area", "jagah": "place", "ghar": "home", "rehna": "live", "rehta": "live",
            
            # Location terms
            "mein": "in", "me": "in", "se": "from", "tak": "to", "ka": "of", "ki": "of", "ke": "of",
            "delhi": "delhi", "mumbai": "mumbai", "bangalore": "bangalore", "pune": "pune", "chennai": "chennai",
            
            # Shift preferences
            "shift": "shift", "time": "time", "timing": "timing", "kab": "when", "kitna": "how much", "kaise": "how"
        }
        
        # Convert to lowercase for consistent processing
        normalized = text.lower()
        
        # Apply word-by-word normalizations
        words = normalized.split()
        normalized_words = []
        
        for word in words:
            # Remove punctuation for matching
            clean_word = word.strip('.,!?;:"()[]{}')
            if clean_word in normalizations:
                normalized_words.append(normalizations[clean_word])
            else:
                normalized_words.append(word)
        
        result = ' '.join(normalized_words)
        
        # Additional pattern-based normalizations
        # Handle pincode patterns
        import re
        result = re.sub(r'\b(\d{1,2})\s*(lakh|hazaar|hazar)\s*(\d{1,2})\b', r'\1\3', result)
        
        logger.debug(f"Text normalization: '{text}' -> '{result}'")
        
        return result 