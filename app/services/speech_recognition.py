import asyncio
import io
import time
import wave
import audioop
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
        # Track last provider actually used
        self._last_provider_used: str = "unknown"
        # Flag: can we run pydub preprocessing/export (ffmpeg present)?
        self.ffmpeg_available: bool = bool(
            AudioSegment is not None and getattr(AudioSegment, 'converter', None) and getattr(AudioSegment, 'ffprobe', None)
        )
        
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
            Dict[str, Any]: {"text": str, "confidence": float, "raw_confidence_data": Dict}
        """
        text, confidence = await self.recognize_speech_from_audio(audio_data)
        
        # Get additional raw confidence data if available
        raw_confidence_data = await self._get_raw_confidence_data(audio_data, text, confidence)
        
        return {
            "text": text, 
            "confidence": confidence,
            "raw_confidence_data": raw_confidence_data
        }

    def _preprocess_audio_for_asr(self, audio_data: bytes) -> bytes:
        """Light preprocessing to improve ASR on noisy inputs.
        - Convert to mono 16kHz
        - Apply mild band-pass (100Hz-4kHz)
        - Normalize to a target dBFS
        """
        # Path 1: Use pydub when ffmpeg is available
        if AudioSegment is not None and getattr(self, 'ffmpeg_available', False):
            try:
                segment = AudioSegment.from_file(io.BytesIO(audio_data))
                # Convert to mono 16kHz
                segment = segment.set_channels(1).set_frame_rate(16000)
                # Mild band-pass
                try:
                    segment = segment.high_pass_filter(100).low_pass_filter(4000)
                except Exception:
                    pass
                # Normalize to about -16 dBFS
                target_dbfs = -16.0
                change = target_dbfs - segment.dBFS if segment.dBFS is not None else 0
                segment = segment.apply_gain(change)
                out_buf = io.BytesIO()
                segment.export(out_buf, format="wav")
                out_buf.seek(0)
                return out_buf.read()
            except Exception as e:
                logger.warning(f"ASR preprocessing (pydub) failed, trying pure-python: {e}")
        # Path 2: Pure-Python WAV processing (no ffmpeg)
        try:
            buf = io.BytesIO(audio_data)
            with wave.open(buf, 'rb') as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                pcm = wf.readframes(n_frames)
            # Ensure 16-bit samples
            if sampwidth not in (1, 2, 3, 4):
                return audio_data
            # To mono
            if n_channels == 2:
                pcm = audioop.tomono(pcm, sampwidth, 0.5, 0.5)
            # Resample to 16kHz if needed
            if framerate != 16000:
                pcm, _ = audioop.ratecv(pcm, sampwidth, 1, framerate, 16000, None)
                framerate = 16000
            # Normalize RMS to target
            try:
                rms = audioop.rms(pcm, sampwidth)
                if rms > 0:
                    target_rms = 2000  # conservative target to avoid clipping
                    factor = min(8.0, max(0.5, target_rms / float(rms)))
                    pcm = audioop.mul(pcm, sampwidth, factor)
            except Exception:
                pass
            # Write back to WAV
            out = io.BytesIO()
            with wave.open(out, 'wb') as ww:
                ww.setnchannels(1)
                ww.setsampwidth(sampwidth)
                ww.setframerate(framerate)
                ww.writeframes(pcm)
            out.seek(0)
            return out.read()
        except Exception as e:
            logger.warning(f"ASR preprocessing (pure-python) failed, using raw audio: {e}")
            return audio_data

    async def _get_raw_confidence_data(self, audio_data: bytes, text: str, confidence: float) -> Dict[str, Any]:
        """Get raw confidence data from the ASR model if available.
        
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
            "audio_size_bytes": len(audio_data)
        }
        
        # Report the actual provider used this turn
        provider = getattr(self, "_last_provider_used", "unknown")
        raw_data["asr_provider"] = provider
        if provider == "google":
            raw_data["confidence_source"] = "model_direct"
            raw_data["confidence_type"] = "real_model_score"
        elif provider == "elevenlabs":
            raw_data["confidence_source"] = "api_response"
            raw_data["confidence_type"] = "api_score"
        elif provider == "sarvam":
            raw_data["confidence_source"] = "api_response"
            raw_data["confidence_type"] = "api_score"
        else:
            raw_data["confidence_source"] = "estimated"
            raw_data["confidence_type"] = "estimated_score"
        
        return raw_data
    
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
        
        provider_used = "unknown"
        try:
            text = ""
            confidence = 0.0
            # Preprocess audio once for all downstream recognizers
            preprocessed = self._preprocess_audio_for_asr(audio_data)
            # Primary: Google if configured
            if self.google_client is not None:
                try:
                    text, confidence = await self._recognize_with_google(preprocessed)
                    provider_used = "google"
                except Exception as ge:
                    logger.error(f"Google ASR failed: {ge}")
                    text, confidence = "", 0.0
                # If empty, try ElevenLabs if configured
                if not text and self.elevenlabs_enabled:
                    el_text, el_conf = await self._recognize_with_elevenlabs(preprocessed)
                    if el_text:
                        text, confidence = el_text, el_conf
                        provider_used = "elevenlabs"
                # If still empty, try Sarvam if available
                if not text and self.sarvam_client is not None:
                    try:
                        text, confidence = await self._recognize_with_sarvam_streaming(preprocessed)
                        provider_used = "sarvam"
                    except Exception as se:
                        logger.error(f"Sarvam ASR failed: {se}")
                # If still empty, try local SpeechRecognition
                if not text and self.recognizer is not None:
                    sr_text, sr_conf = await self._recognize_with_speechrecognition(preprocessed)
                    if sr_text:
                        text, confidence = sr_text, sr_conf
                        provider_used = "speech_recognition"
            # If Google not configured, next prefer Sarvam
            elif self.sarvam_client is not None:
                try:
                    text, confidence = await self._recognize_with_sarvam_streaming(preprocessed)
                    provider_used = "sarvam"
                except Exception as se:
                    logger.error(f"Sarvam ASR failed: {se}")
                    text, confidence = "", 0.0
                if not text and self.recognizer is not None:
                    sr_text, sr_conf = await self._recognize_with_speechrecognition(preprocessed)
                    if sr_text:
                        text, confidence = sr_text, sr_conf
                        provider_used = "speech_recognition"
                if not text and self.elevenlabs_enabled:
                    el_text, el_conf = await self._recognize_with_elevenlabs(preprocessed)
                    if el_text:
                        text, confidence = el_text, el_conf
                        provider_used = "elevenlabs"
            # Final fallback: SpeechRecognition
            else:
                text, confidence = await self._recognize_with_speechrecognition(preprocessed)
                provider_used = "speech_recognition"
        except Exception as e:
            logger.error(f"ASR recognition failed: {e}")
            text, confidence = await self._recognize_with_speechrecognition(audio_data)
            provider_used = "speech_recognition"
        
        processing_time = (time.time() - start_time) * 1000
        # Remember provider for raw confidence reporting
        self._last_provider_used = provider_used
        logger.bind(metrics=True).info({
            "event": "asr_completed",
            "provider": provider_used,
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
            # Extract raw confidence from Sarvam response if available
            raw_confidence = payload.get("confidence", payload.get("score", payload.get("probability")))
            if raw_confidence is not None:
                confidence = float(raw_confidence)
            else:
                # No confidence available from Sarvam
                confidence = 0.0
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
            # Extract raw confidence from ElevenLabs response
            raw_confidence = data.get("confidence", data.get("score", data.get("probability")))
            if raw_confidence is not None:
                confidence = float(raw_confidence)
            else:
                # Use language probability as fallback, or reasonable default
                confidence = float(data.get("language_probability", 0.85)) if text else 0.0
            return text, confidence

    async def _recognize_with_google(self, audio_data: bytes) -> Tuple[str, float]:
        """Recognize speech using Google Speech-to-Text with real confidence scores."""
        if not hasattr(self, 'google_client') or self.google_client is None:
            raise Exception("Google Speech not configured")
            
        try:
            # Helper: heuristic MP3 detector
            def _looks_like_mp3(payload: bytes) -> bool:
                if not payload or len(payload) < 4:
                    return False
                if payload[:3] == b"ID3":
                    return True
                b1, b2 = payload[0], payload[1]
                return (b1 == 0xFF) and ((b2 & 0xE0) == 0xE0)

            # Helper: heuristic WEBM and OGG detectors
            def _looks_like_webm(payload: bytes) -> bool:
                # EBML header 1A 45 DF A3
                return bool(payload and len(payload) >= 4 and payload[0:4] == b"\x1A\x45\xDF\xA3")
            def _looks_like_ogg(payload: bytes) -> bool:
                # OggS magic
                return bool(payload and len(payload) >= 4 and payload[0:4] == b"OggS")

            def _recognize_with_cfg(payload: bytes, cfg: Any) -> Tuple[str, float]:
                rec_audio = speech.RecognitionAudio(content=payload)
                resp = self.google_client.recognize(config=cfg, audio=rec_audio)
                if resp.results and resp.results[0].alternatives:
                    alt = resp.results[0].alternatives[0]
                    return (alt.transcript or ""), float(alt.confidence or 0.0)
                return "", 0.0

            original_bytes = audio_data
            try_order: list[Any] = []
            # Prefer container-specific encodings first
            if _looks_like_webm(original_bytes):
                try_order.append(speech.RecognitionConfig.AudioEncoding.WEBM_OPUS)
            if _looks_like_ogg(original_bytes):
                try_order.append(speech.RecognitionConfig.AudioEncoding.OGG_OPUS)
            if _looks_like_mp3(original_bytes):
                try_order.append(speech.RecognitionConfig.AudioEncoding.MP3)
            # Always include unspecified as a general fallback
            try_order.append(speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED)

            # Try MP3 (if likely) then unspecified
            for enc in try_order:
                cfg = speech.RecognitionConfig(
                    encoding=enc,
                    sample_rate_hertz=16000,
                    language_code="hi-IN",
                    alternative_language_codes=["en-IN"],
                    enable_automatic_punctuation=True,
                    model="latest_long",
                    enable_word_time_offsets=False,
                    max_alternatives=1,
                )
                try:
                    text, conf = _recognize_with_cfg(original_bytes, cfg)
                    if text:
                        return text, conf
                except Exception as e:
                    logger.error(f"Google ASR config {enc} error: {e}")
                    continue

            # Final attempt: re-encode to 16kHz mono LINEAR16 using pydub if available
            if AudioSegment is not None:
                try:
                    segment = AudioSegment.from_file(io.BytesIO(original_bytes))
                    segment = segment.set_frame_rate(16000).set_channels(1)
                    wav_buf = io.BytesIO()
                    segment.export(wav_buf, format="wav")
                    wav_buf.seek(0)
                    processed_bytes = wav_buf.read()

                    cfg_lin = speech.RecognitionConfig(
                        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                        sample_rate_hertz=16000,
                        language_code="hi-IN",
                        alternative_language_codes=["en-IN"],
                        enable_automatic_punctuation=True,
                        model="latest_long",
                        enable_word_time_offsets=False,
                        max_alternatives=1,
                    )
                    return _recognize_with_cfg(processed_bytes, cfg_lin)
                except Exception as norm_e:
                    logger.error(f"Google ASR LINEAR16 fallback failed: {norm_e}")

            return "", 0.0
             
        except Exception as e:
            logger.error(f"Google ASR error: {e}")
            # Re-raise to allow higher-level fallback logic
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
            # Google Web Speech doesn't provide confidence, so estimate based on text quality
            # Longer, more coherent text suggests higher confidence
            if len(text) > 10:
                confidence = 0.75  # Good length, likely accurate
            elif len(text) > 5:
                confidence = 0.65  # Medium length, moderate confidence
            else:
                confidence = 0.55  # Short text, lower confidence
            return text, confidence
            
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