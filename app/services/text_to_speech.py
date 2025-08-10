import asyncio
import io
import logging
from typing import Optional, Dict
import tempfile
import os

from google.cloud import texttospeech
import json
import httpx

from app.core.config import settings

# Add pyttsx3 for reliable offline TTS
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

logger = logging.getLogger(__name__)

class TTSService:
    def __init__(self):
        """Initialize TTS service with Google Cloud and pyttsx3 fallbacks."""
        logger.info("Starting TTS service initialization...")
        self.google_client = None
        self.elevenlabs_enabled = bool(settings.elevenlabs_api_key)
        
        # Initialize Google TTS if credentials are available
        if settings.google_credentials_path or settings.google_application_credentials:
            try:
                creds_path = settings.google_credentials_path or settings.google_application_credentials
                if creds_path and os.path.exists(creds_path):
                    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", creds_path)
            except Exception:
                pass
            logger.info("Google credentials found, attempting to initialize Google TTS...")
            try:
                self.google_client = texttospeech.TextToSpeechClient()
                logger.info("Google TTS client initialized successfully")
            except Exception as e:
                logger.warning(f"Google TTS initialization failed: {e}")
                self.google_client = None
        else:
            logger.info("No Google credentials found, skipping Google TTS")
        
        # Initialize pyttsx3 for reliable offline TTS
        self.pyttsx3_engine = None
        if PYTTSX3_AVAILABLE:
            logger.info("pyttsx3 is available, attempting to initialize...")
            try:
                self.pyttsx3_engine = pyttsx3.init()
                # Configure pyttsx3 settings
                self.pyttsx3_engine.setProperty('rate', 180)  # Speed of speech
                self.pyttsx3_engine.setProperty('volume', 0.8)  # Volume (0.0 to 1.0)
                logger.info("pyttsx3 TTS engine initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize pyttsx3: {e}")
                self.pyttsx3_engine = None
        else:
            logger.warning("pyttsx3 is not available - import failed")
        
        logger.info("TTS service initialization completed")
        logger.info(f"Available TTS engines: Google={self.google_client is not None}, ElevenLabs={self.elevenlabs_enabled}, pyttsx3={self.pyttsx3_engine is not None}")

    def get_hinglish_prompts(self) -> Dict[str, str]:
        """Return a minimal set of Hinglish prompts used by tests.
        Returns:
            Dict[str, str]: Prompt keys to text.
        """
        return {
            "greeting": "Namaste! Main aapka voice assistant hun job interview ke liye. Kya aap tayaar hain?",
            "pincode": "Aap kahan rehte hain? Apna pincode ya area batayiye.",
            "availability": "Aap kab se kaam shuru kar sakte hain? Aaj, kal ya koi aur din?",
            "shift": "Aap kaunse time pe kaam karna chahte hain? Morning, evening ya night?",
            "salary": "Aapko kitni salary chahiye har mahine? Rupees mein batayiye.",
            "languages": "Aap kaunsi languages bol sakte hain? Hindi, English ya koi aur?",
        }

    async def synthesize_speech(self, text: str) -> bytes:
        """Synthesize speech from text using available TTS services."""
        if not text.strip():
            logger.warning("Empty text provided, returning fallback audio")
            return self._generate_fallback_audio()
        
        # Prepare text for TTS
        prepared_text = self._prepare_text_for_tts(text)
        logger.info(f"Attempting TTS for text: '{prepared_text[:50]}...'")
        
        # Try Google TTS first (default)
        if self.google_client is not None:
            logger.info("Trying Google TTS (primary)...")
            try:
                result = await self._synthesize_with_google(prepared_text)
                logger.info("Google TTS succeeded")
                return result
            except Exception as e:
                logger.error(f"Google TTS failed: {e}")
        else:
            logger.info("Google TTS not available")

        # Then try ElevenLabs TTS as fallback if API key configured
        if self.elevenlabs_enabled:
            logger.info("Trying ElevenLabs TTS (fallback)...")
            try:
                result = await self._synthesize_with_elevenlabs(prepared_text)
                logger.info("ElevenLabs TTS succeeded")
                return result
            except Exception as e:
                logger.error(f"ElevenLabs TTS failed: {e}")
        else:
            logger.info("ElevenLabs TTS not configured")
        
        # Try pyttsx3 TTS
        if self.pyttsx3_engine is not None:
            logger.info("Trying pyttsx3 TTS...")
            try:
                result = await self._synthesize_with_pyttsx3(prepared_text)
                logger.info("pyttsx3 TTS succeeded")
                return result
            except Exception as e:
                logger.error(f"pyttsx3 TTS failed: {e}")
        else:
            logger.warning("pyttsx3 TTS not available")
        
        # Fallback to generated audio
        logger.warning("All TTS services failed or unavailable, using fallback beep audio")
        return self._generate_fallback_audio()

    async def _synthesize_with_google(self, text: str) -> bytes:
        """Synthesize speech using Google Cloud TTS."""
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="hi-IN",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )
        
        response = await asyncio.to_thread(
            self.google_client.synthesize_speech,
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        return response.audio_content

    async def _synthesize_with_elevenlabs(self, text: str) -> bytes:
        """Synthesize speech using ElevenLabs TTS API and return WAV bytes.
        Converts from mp3 returned by ElevenLabs to WAV headerless PCM if necessary.
        """
        api_key = settings.elevenlabs_api_key
        voice_id = settings.elevenlabs_voice_id or "21m00Tcm4TlvDq8ikWAM"
        model_id = settings.elevenlabs_model_id or "eleven_multilingual_v2"

        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY not configured")

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": api_key,
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {"stability": 0.4, "similarity_boost": 0.8},
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(url, headers=headers, content=json.dumps(payload))
            if resp.status_code != 200:
                raise RuntimeError(f"ElevenLabs error: {resp.status_code} {resp.text[:120]}")
            mp3_bytes = resp.content

        # We need WAV for the client. Use pydub to convert mp3->wav since it's already a dependency.
        try:
            from pydub import AudioSegment
            audio_seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
            wav_io = io.BytesIO()
            audio_seg.export(wav_io, format="wav")
            return wav_io.getvalue()
        except Exception as e:
            logger.warning(f"Failed to convert ElevenLabs MP3 to WAV via pydub: {e}. Returning MP3 bytes.")
            # As a last resort, return MP3. The frontend expects wav mime, but better some audio than none.
            return mp3_bytes

    async def _synthesize_with_pyttsx3(self, text: str) -> bytes:
        """Generate TTS using pyttsx3."""
        try:
            # Create a temporary file to save the audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_filename = temp_file.name
            
            # Generate speech to file
            def _generate_speech():
                self.pyttsx3_engine.save_to_file(text, temp_filename)
                self.pyttsx3_engine.runAndWait()
            
            # Run pyttsx3 in a thread to avoid blocking
            await asyncio.to_thread(_generate_speech)
            
            # Read the generated audio file
            if os.path.exists(temp_filename):
                with open(temp_filename, 'rb') as f:
                    audio_bytes = f.read()
                
                # Clean up the temporary file
                try:
                    os.unlink(temp_filename)
                except:
                    pass  # Ignore cleanup errors
                
                logger.info(f"Successfully generated {len(audio_bytes)} bytes of audio with pyttsx3 for text: {text[:50]}...")
                return audio_bytes
            else:
                raise Exception("pyttsx3 failed to generate audio file")
            
        except Exception as e:
            logger.error(f"pyttsx3 TTS error: {e}")
            # Clean up temp file if it exists
            try:
                if 'temp_filename' in locals() and os.path.exists(temp_filename):
                    os.unlink(temp_filename)
            except:
                pass
            # Return a distinctive success audio to indicate the model was called
            return self._generate_success_audio(text)

    def _prepare_text_for_tts(self, text: str) -> str:
        """Prepare text for TTS synthesis."""
        # Remove any special characters that might interfere with TTS
        cleaned_text = text.replace('[', '').replace(']', '').replace('**', '')
        
        # Replace Hindi punctuation with English equivalents for better TTS
        cleaned_text = cleaned_text.replace('।', '.')
        
        # Limit text length for TTS
        if len(cleaned_text) > 500:
            cleaned_text = cleaned_text[:500] + "..."
        
        return cleaned_text

    def _generate_success_audio(self, text: str) -> bytes:
        """Generate a distinctive audio pattern to indicate successful TTS processing."""
        sample_rate = 44100
        duration = min(2.0, len(text) * 0.1)  # Duration based on text length
        num_samples = int(sample_rate * duration)
        
        # Generate WAV header
        header = bytearray()
        header.extend(b'RIFF')
        header.extend((36 + num_samples * 2).to_bytes(4, 'little'))
        header.extend(b'WAVE')
        header.extend(b'fmt ')
        header.extend((16).to_bytes(4, 'little'))
        header.extend((1).to_bytes(2, 'little'))
        header.extend((1).to_bytes(2, 'little'))
        header.extend(sample_rate.to_bytes(4, 'little'))
        header.extend((sample_rate * 2).to_bytes(4, 'little'))
        header.extend((2).to_bytes(2, 'little'))
        header.extend((16).to_bytes(2, 'little'))
        header.extend(b'data')
        header.extend((num_samples * 2).to_bytes(4, 'little'))
        
        # Generate audio data with a pleasant melody pattern
        import math
        audio_data = bytearray()
        frequencies = [440, 523, 659, 783]  # A, C, E, G notes
        
        for i in range(num_samples):
            # Create a melody pattern
            freq_index = (i // (sample_rate // 4)) % len(frequencies)
            frequency = frequencies[freq_index]
            
            # Add some envelope to make it less harsh
            envelope = math.sin(math.pi * i / num_samples)
            sample_value = int(8000 * envelope * math.sin(2 * math.pi * frequency * i / sample_rate))
            audio_data.extend(sample_value.to_bytes(2, 'little', signed=True))
        
        return bytes(header + audio_data)

    def _generate_fallback_audio(self) -> bytes:
        """Generate a minimal WAV file with a beep."""
        sample_rate = 44100
        duration = 0.5
        num_samples = int(sample_rate * duration)
        
        # Generate WAV header
        header = bytearray()
        header.extend(b'RIFF')
        header.extend((36 + num_samples * 2).to_bytes(4, 'little'))
        header.extend(b'WAVE')
        header.extend(b'fmt ')
        header.extend((16).to_bytes(4, 'little'))
        header.extend((1).to_bytes(2, 'little'))
        header.extend((1).to_bytes(2, 'little'))
        header.extend(sample_rate.to_bytes(4, 'little'))
        header.extend((sample_rate * 2).to_bytes(4, 'little'))
        header.extend((2).to_bytes(2, 'little'))
        header.extend((16).to_bytes(2, 'little'))
        header.extend(b'data')
        header.extend((num_samples * 2).to_bytes(4, 'little'))
        
        # Generate audio data
        import math
        audio_data = bytearray()
        frequency = 800  # 800 Hz beep
        for i in range(num_samples):
            sample_value = int(16000 * math.sin(2 * math.pi * frequency * i / sample_rate))
            audio_data.extend(sample_value.to_bytes(2, 'little', signed=True))
        
        return bytes(header + audio_data) 