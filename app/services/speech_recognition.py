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
    from pydub.utils import which
except ImportError:
    print("pydub not installed. Install with: pip install pydub")
    AudioSegment = None

# Azure removed
speechsdk = None

try:
    from google.cloud import speech
except ImportError:
    print("Google Cloud Speech not installed.")
    speech = None


import httpx
import json
import os
import base64

from app.core.config import settings
from app.core.logger import logger

# Sarvam SDK (optional)
try:
    from sarvamai import AsyncSarvamAI  # type: ignore
except Exception:  # pragma: no cover
    AsyncSarvamAI = None  # type: ignore


class ASRService:
    """Automatic Speech Recognition service supporting multiple providers."""
    
    def __init__(self):
        # Configure ffmpeg/ffprobe if provided
        if AudioSegment is not None:
            try:
                if settings.ffmpeg_path and os.path.exists(settings.ffmpeg_path):
                    AudioSegment.converter = settings.ffmpeg_path
                if settings.ffprobe_path and os.path.exists(settings.ffprobe_path):
                    AudioSegment.ffprobe = settings.ffprobe_path
                # Fallback to auto-detect if not set
                if not getattr(AudioSegment, 'converter', None):
                    AudioSegment.converter = which('ffmpeg') or which('avconv')
                if not getattr(AudioSegment, 'ffprobe', None):
                    AudioSegment.ffprobe = which('ffprobe') or which('avprobe')
            except Exception:
                pass
        
        # Still track whether ElevenLabs key exists (used elsewhere), but do not use for ASR
        self.elevenlabs_enabled = bool(settings.elevenlabs_api_key)
        
        if sr is not None:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 1.0
        else:
            self.recognizer = None
        
        # Remove Azure
        self.azure_config = None
        
        # Configure Google Speech (primary ASR if Sarvam not available)
        if speech and (settings.google_credentials_path or settings.google_application_credentials):
            try:
                # Ensure env var is set for Google SDK
                creds_path = settings.google_credentials_path or settings.google_application_credentials
                if creds_path and os.path.exists(creds_path):
                    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", creds_path)
            except Exception:
                pass
            try:
                self.google_client = speech.SpeechClient()
            except Exception as e:
                logger.error(f"Failed to initialize Google Speech client: {e}")
                self.google_client = None
        else:
            self.google_client = None
        
        # Removed HF Whisper pipeline
        self.hf_asr = None

        # Configure Sarvam client (preferred when available)
        self.sarvam_client = None
        if settings.sarvam_api_key and AsyncSarvamAI is not None:
            try:
                self.sarvam_client = AsyncSarvamAI(api_subscription_key=settings.sarvam_api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Sarvam client: {e}")
                self.sarvam_client = None
    
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
        if self.recognizer is None and self.google_client is None and self.sarvam_client is None:
            return "Voice recognition not available in demo mode", 0.0
            
        start_time = time.time()
        
        try:
            # Preferred: Sarvam streaming if configured
            if self.sarvam_client is not None:
                text, confidence = await self._recognize_with_sarvam_streaming(audio_data)
            # Next: Google if configured
            elif self.google_client is not None:
                text, confidence = await self._recognize_with_google(audio_data)
            # Final: SpeechRecognition lib
            else:
                text, confidence = await self._recognize_with_speechrecognition(audio_data)
        except Exception as e:
            logger.error(f"ASR recognition failed: {e}")
            # Final fallback to speech_recognition library
            text, confidence = await self._recognize_with_speechrecognition(audio_data)
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.bind(metrics=True).info({
            "event": "asr_completed",
            "provider": "sarvam" if self.sarvam_client is not None else ("google" if self.google_client is not None else "speech_recognition"),
            "text": text,
            "confidence": confidence,
            "processing_time_ms": processing_time
        })
        
        return text, confidence

    async def _recognize_with_sarvam_streaming(self, audio_data: bytes) -> Tuple[str, float]:
        """Recognize speech using Sarvam streaming ASR.
        
        Args:
            audio_data (bytes): Raw audio data to transcribe
        
        Returns:
            Tuple[str, float]: (recognized_text, confidence_score)
        """
        if self.sarvam_client is None:
            raise Exception("Sarvam ASR not configured")

        audio_b64: str = base64.b64encode(audio_data).decode("utf-8")
        language_code: str = settings.sarvam_language_code or "hi-IN"
        model: str = settings.sarvam_asr_model or "saarika:v2.5"
        high_vad: bool = bool(settings.sarvam_high_vad_sensitivity)
        vad_signals: bool = bool(settings.sarvam_vad_signals)

        async with self.sarvam_client.speech_to_text_streaming.connect(
            language_code=language_code,
            model=model,
            high_vad_sensitivity=high_vad,
            vad_signals=vad_signals,
        ) as ws:
            await ws.transcribe(audio=audio_b64)

            final_text: str = ""
            # Read messages until final transcript is received
            for _ in range(50):  # safety cap to avoid infinite loops
                resp = await ws.recv()
                try:
                    if isinstance(resp, (bytes, bytearray)):
                        payload = json.loads(resp.decode("utf-8"))
                    elif isinstance(resp, str):
                        payload = json.loads(resp)
                    elif isinstance(resp, dict):
                        payload = resp
                    else:
                        payload = {}
                except Exception:
                    payload = {}
                event_type = payload.get("type") or payload.get("event")
                if event_type == "transcript" or ("transcript" in payload):
                    final_text = payload.get("transcript", payload.get("text", ""))
                    break
            confidence: float = 0.9 if final_text else 0.0
            return final_text, confidence

    async def _recognize_with_elevenlabs(self, audio_data: bytes) -> Tuple[str, float]:
        """Recognize speech using ElevenLabs Scribe STT API.
        Sends audio as WAV if possible; falls back to raw bytes.
        """
        if not settings.elevenlabs_api_key:
            raise Exception("ElevenLabs API key not configured")
        # Convert to WAV for best compatibility
        payload_bytes = audio_data
        filename = "audio.wav"
        content_type = "audio/wav"
        if AudioSegment is not None:
            try:
                segment = AudioSegment.from_file(io.BytesIO(audio_data))
                wav_buf = io.BytesIO()
                segment.export(wav_buf, format="wav")
                wav_buf.seek(0)
                payload_bytes = wav_buf.read()
            except Exception as conv_e:
                logger.warning(f"ElevenLabs ASR: could not convert to WAV, using raw bytes: {conv_e}")
                filename = "audio.bin"
                content_type = "application/octet-stream"
        else:
            filename = "audio.bin"
            content_type = "application/octet-stream"

        url = "https://api.elevenlabs.io/v1/speech-to-text"
        headers = {"xi-api-key": settings.elevenlabs_api_key}
        form = {
            "model_id": (None, settings.elevenlabs_asr_model_id or "scribe_v1"),
        }
        if settings.elevenlabs_asr_language_code:
            form["language_code"] = (None, settings.elevenlabs_asr_language_code)

        files = {
            "audio": (filename, payload_bytes, content_type)
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, headers=headers, data=form, files=files)
            if resp.status_code != 200:
                raise Exception(f"ElevenLabs STT error: {resp.status_code} {resp.text[:200]}")
            data = resp.json()
            text = data.get("text", "")
            # If they provide probability, map to confidence, else default
            confidence = float(data.get("language_probability", 0.9)) if text else 0.0
            return text, confidence

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