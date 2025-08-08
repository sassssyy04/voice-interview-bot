import asyncio
import io
import time
from typing import Tuple, Optional, Dict, Any
try:
    import speech_recognition as sr
except ImportError:
    print("SpeechRecognition not installed. Install with: pip install SpeechRecognition")
    sr = None

try:
    from pydub import AudioSegment
except ImportError:
    print("pydub not installed. Install with: pip install pydub")
    AudioSegment = None

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    print("Azure Speech SDK not installed.")
    speechsdk = None

try:
    from google.cloud import speech
except ImportError:
    print("Google Cloud Speech not installed.")
    speech = None
from app.core.config import settings
from app.core.logger import logger


class ASRService:
    """Automatic Speech Recognition service supporting multiple providers."""
    
    def __init__(self):
        if sr is not None:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 1.0
        else:
            self.recognizer = None
        
        # Configure Azure Speech (primary for Hinglish)
        if speechsdk and settings.azure_speech_key and settings.azure_speech_region:
            self.azure_config = speechsdk.SpeechConfig(
                subscription=settings.azure_speech_key,
                region=settings.azure_speech_region
            )
            self.azure_config.speech_recognition_language = "hi-IN"
        else:
            self.azure_config = None
            
        # Configure Google Speech (fallback)
        if speech and settings.google_credentials_path:
            self.google_client = speech.SpeechClient()
        else:
            self.google_client = None
    
    async def transcribe_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """Transcribe audio and return text and confidence.
        
        Args:
            audio_data: Raw audio bytes
        Returns:
            Dict[str, Any]: {"text": str, "confidence": float}
        """
        text, confidence = await self.recognize_speech_from_audio(audio_data)
        return {"text": text, "confidence": confidence}
    
    async def recognize_speech_from_audio(self, audio_data: bytes) -> Tuple[str, float]:
        """Recognize speech from audio data with confidence score.
        
        Args:
            audio_data (bytes): Raw audio data
            
        Returns:
            Tuple[str, float]: (recognized_text, confidence_score)
        """
        if self.recognizer is None:
            return "Voice recognition not available in demo mode", 0.0
            
        start_time = time.time()
        
        try:
            # Try Azure first (best for Hinglish)
            text, confidence = await self._recognize_with_azure(audio_data)
            
            if confidence < settings.confidence_threshold:
                logger.warning(f"Low confidence from Azure: {confidence}")
                # Fallback to Google
                try:
                    google_text, google_confidence = await self._recognize_with_google(audio_data)
                    if google_confidence > confidence:
                        text, confidence = google_text, google_confidence
                except Exception as ge:
                    logger.error(f"Google ASR fallback failed: {ge}")
                    # Fall through to SpeechRecognition
                    text, confidence = await self._recognize_with_speechrecognition(audio_data)
                    
        except Exception as e:
            logger.error(f"ASR recognition failed with Azure: {e}")
            # Try Google next
            try:
                text, confidence = await self._recognize_with_google(audio_data)
            except Exception as ge:
                logger.error(f"ASR recognition also failed with Google: {ge}")
                # Final fallback to speech_recognition library
                text, confidence = await self._recognize_with_speechrecognition(audio_data)
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.bind(metrics=True).info({
            "event": "asr_completed",
            "text": text,
            "confidence": confidence,
            "processing_time_ms": processing_time
        })
        
        return text, confidence
    
    async def _recognize_with_azure(self, audio_data: bytes) -> Tuple[str, float]:
        """Recognize speech using Azure Speech Service."""
        if not hasattr(self, 'azure_config') or self.azure_config is None:
            raise Exception("Azure Speech not configured")
            
        try:
            # Convert audio to format expected by Azure
            audio_stream = io.BytesIO(audio_data)
            audio_config = speechsdk.audio.AudioConfig(stream=speechsdk.audio.PullAudioInputStream(audio_stream))
            
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.azure_config,
                audio_config=audio_config
            )
            
            # Perform recognition
            result = speech_recognizer.recognize_once()
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                confidence = self._extract_azure_confidence(result)
                return result.text, confidence
            elif result.reason == speechsdk.ResultReason.NoMatch:
                return "", 0.0
            else:
                raise Exception(f"Azure recognition failed: {result.reason}")
                
        except Exception as e:
            logger.error(f"Azure ASR error: {e}")
            raise
    
    async def _recognize_with_google(self, audio_data: bytes) -> Tuple[str, float]:
        """Recognize speech using Google Speech-to-Text."""
        if not hasattr(self, 'google_client') or self.google_client is None:
            raise Exception("Google Speech not configured")
            
        try:
            audio = speech.RecognitionAudio(content=audio_data)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
                sample_rate_hertz=48000,
                language_code="hi-IN",
                alternative_language_codes=["en-IN"],
                enable_automatic_punctuation=True,
                model="latest_long"
            )
            
            response = self.google_client.recognize(config=config, audio=audio)
            
            if response.results:
                result = response.results[0]
                if result.alternatives:
                    alternative = result.alternatives[0]
                    return alternative.transcript, alternative.confidence
                    
            return "", 0.0
            
        except Exception as e:
            logger.error(f"Google ASR error: {e}")
            raise
    
    async def _recognize_with_speechrecognition(self, audio_data: bytes) -> Tuple[str, float]:
        """Fallback recognition using speech_recognition library."""
        try:
            # Convert bytes to AudioData
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
            wav_data = io.BytesIO()
            audio_segment.export(wav_data, format="wav")
            wav_data.seek(0)
            
            with sr.AudioFile(wav_data) as source:
                audio = self.recognizer.record(source)
            
            # Try Google Web Speech API as last resort
            text = self.recognizer.recognize_google(audio, language="hi-IN")
            return text, 0.8  # Assume reasonable confidence
            
        except sr.UnknownValueError:
            return "", 0.0
        except Exception as e:
            logger.error(f"Fallback ASR error: {e}")
            return "", 0.0
    
    def _extract_azure_confidence(self, result) -> float:
        """Extract confidence score from Azure result."""
        try:
            # Azure provides confidence in JSON details
            import json
            details = json.loads(result.json)
            if 'NBest' in details and details['NBest']:
                return details['NBest'][0].get('Confidence', 0.0)
        except:
            pass
        return 0.8  # Default confidence if not available
    
    async def normalize_hinglish_text(self, text: str) -> str:
        """Normalize Hinglish text for better entity extraction.
        
        Args:
            text (str): Raw recognized text
            
        Returns:
            str: Normalized text
        """
        if not text:
            return text
            
        # Common Hinglish normalizations
        normalizations = {
            # Time expressions
            "aaj": "today",
            "kal": "tomorrow", 
            "parso": "day after tomorrow",
            "abhi": "now",
            "jaldi": "soon",
            
            # Numbers in Hindi
            "ek": "1", "do": "2", "teen": "3", "char": "4", "panch": "5",
            "che": "6", "saat": "7", "aath": "8", "nau": "9", "das": "10",
            "hazaar": "thousand", "lakh": "lakh",
            
            # Yes/No
            "haan": "yes", "han": "yes", "ji haan": "yes",
            "nahi": "no", "nahin": "no", "na": "no",
            
            # Common words
            "paisa": "money", "rupaye": "rupees", "salary": "salary",
            "morning": "morning", "evening": "evening", "night": "night",
            "bike": "two wheeler", "scooter": "two wheeler", "cycle": "bicycle"
        }
        
        normalized = text.lower()
        for hindi, english in normalizations.items():
            normalized = normalized.replace(hindi, english)
            
        return normalized 