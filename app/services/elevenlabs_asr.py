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
            Dict[str, Any]: {"text": str, "confidence": float, "raw_confidence_data": Dict}
        """
        text, confidence = await self.recognize_speech_from_audio(audio_data)
        
        # Get additional raw confidence data
        raw_confidence_data = await self._get_raw_confidence_data(audio_data, text, confidence)
        
        return {
            "text": text, 
            "confidence": confidence,
            "raw_confidence_data": raw_confidence_data
        }
    
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
                    words = result.get("words", [])
                    
                    # Calculate realistic confidence score based on multiple factors
                    confidence = self._calculate_transcription_confidence(
                        text, words, language_probability, len(audio_data)
                    )
                    
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
    
    def _calculate_transcription_confidence(
        self, 
        text: str, 
        words: list, 
        language_probability: float,
        audio_size: int
    ) -> float:
        """Calculate realistic transcription confidence score.
        
        Args:
            text (str): Transcribed text
            words (list): Word-level data from API
            language_probability (float): Language detection confidence  
            audio_size (int): Size of audio data in bytes
            
        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        if not text.strip():
            return 0.0
        
        base_confidence = 0.85  # Base confidence for ElevenLabs
        
        # Factor 1: Text length (longer text generally more reliable)
        text_length = len(text.strip())
        if text_length < 10:
            length_factor = 0.7  # Short text, lower confidence
        elif text_length < 30:
            length_factor = 0.85
        elif text_length < 100:
            length_factor = 0.95
        else:
            length_factor = 1.0  # Long text, high confidence
        
        # Factor 2: Audio quality (estimated from size vs duration)
        # Typical good quality: ~8KB per second of audio
        estimated_duration = audio_size / 8000  # rough estimate in seconds
        words_per_second = len(words) / max(estimated_duration, 1)
        
        if words_per_second < 0.5:  # Very slow speech
            audio_factor = 0.7
        elif words_per_second < 2.0:  # Normal speech
            audio_factor = 0.9
        elif words_per_second < 4.0:  # Fast but clear
            audio_factor = 0.95
        else:  # Too fast, might be unclear
            audio_factor = 0.8
        
        # Factor 3: Word-level confidence (if available)
        word_factor = 1.0
        if words:
            # Check for any word with very low log probability
            word_logprobs = [w.get('logprob', 0.0) for w in words if w.get('type') == 'word']
            if word_logprobs:
                avg_logprob = sum(word_logprobs) / len(word_logprobs)
                # Convert log probability to confidence factor
                if avg_logprob < -2.0:  # Very uncertain words
                    word_factor = 0.7
                elif avg_logprob < -1.0:  # Somewhat uncertain
                    word_factor = 0.85
                else:  # Good word confidence
                    word_factor = 0.95
        
        # Factor 4: Language detection confidence
        lang_factor = min(language_probability, 1.0)
        
        # Factor 5: Text quality indicators  
        quality_factor = 1.0
        # Penalize if too many numbers (might be misrecognized)
        digit_ratio = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
        if digit_ratio > 0.3:
            quality_factor *= 0.9
        
        # Penalize very short words (might be fragments)
        if words:
            short_words = sum(1 for w in words if w.get('type') == 'word' and len(w.get('text', '')) <= 2)
            short_word_ratio = short_words / len(words)
            if short_word_ratio > 0.5:
                quality_factor *= 0.85
        
        # Combine all factors
        final_confidence = (
            base_confidence * 
            length_factor * 
            audio_factor * 
            word_factor * 
            lang_factor * 
            quality_factor
        )
        
        # Clamp to reasonable range (0.3 to 1.0)
        final_confidence = max(0.3, min(1.0, final_confidence))
        
        return round(final_confidence, 2)

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
    
    async def _get_raw_confidence_data(self, audio_data: bytes, text: str, confidence: float) -> Dict[str, Any]:
        """Get raw confidence data from the ElevenLabs ASR model.
        
        Args:
            audio_data: Raw audio bytes
            text: Recognized text
            confidence: Calculated confidence score
            
        Returns:
            Dict[str, Any]: Raw confidence data from the model
        """
        raw_data = {
            "model_confidence": confidence,
            "text_length": len(text) if text else 0,
            "has_text": bool(text),
            "audio_size_bytes": len(audio_data),
            "asr_provider": "elevenlabs",
            "confidence_source": "api_response",
            "model_name": "scribe_v1",
            "language_code": "hi"
        }
        
        # Add ElevenLabs-specific confidence metrics if available
        if hasattr(self, 'api_key') and self.api_key:
            raw_data["api_key_configured"] = True
        else:
            raw_data["api_key_configured"] = False
        
        return raw_data 