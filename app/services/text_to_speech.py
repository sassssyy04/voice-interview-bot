import asyncio
import io
import time
from typing import Optional
try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    speechsdk = None

try:
    from google.cloud import texttospeech
except ImportError:
    texttospeech = None
from app.core.config import settings
from app.core.logger import logger


class TTSService:
    """Text-to-Speech service supporting multiple providers for Hinglish."""
    
    def __init__(self):
        # Configure Azure Speech (primary for Hinglish)
        if speechsdk and settings.azure_speech_key and settings.azure_speech_region:
            self.azure_config = speechsdk.SpeechConfig(
                subscription=settings.azure_speech_key,
                region=settings.azure_speech_region
            )
            # Use Hindi voice with good Hinglish support
            self.azure_config.speech_synthesis_voice_name = "hi-IN-SwaraNeural"
            self.azure_config.speech_synthesis_language = "hi-IN"
        else:
            self.azure_config = None
            
        # Configure Google TTS (fallback)
        if texttospeech and settings.google_credentials_path:
            self.google_client = texttospeech.TextToSpeechClient()
        else:
            self.google_client = None
    
    async def synthesize_speech(self, text: str) -> bytes:
        """Convert text to speech audio.
        
        Args:
            text (str): Text to convert to speech
            
        Returns:
            bytes: Audio data in WAV format
        """
        if self.azure_config is None and self.google_client is None:
            return b"TTS not available in demo mode"
            
        start_time = time.time()
        
        try:
            # Prepare text for TTS (handle Hinglish)
            prepared_text = self._prepare_hinglish_text(text)
            
            # Try Azure first (better Hinglish support)
            audio_data = await self._synthesize_with_azure(prepared_text)
            
        except Exception as e:
            logger.error(f"Azure TTS failed: {e}")
            try:
                # Fallback to Google
                audio_data = await self._synthesize_with_google(prepared_text)
            except Exception as e2:
                logger.error(f"Google TTS failed: {e2}")
                raise Exception("All TTS providers failed")
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.bind(metrics=True).info({
            "event": "tts_completed",
            "text_length": len(text),
            "audio_size_bytes": len(audio_data),
            "processing_time_ms": processing_time
        })
        
        return audio_data
    
    async def _synthesize_with_azure(self, text: str) -> bytes:
        """Synthesize speech using Azure Speech Service."""
        if not hasattr(self, 'azure_config'):
            raise Exception("Azure Speech not configured")
            
        try:
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.azure_config,
                audio_config=None  # Return audio data instead of playing
            )
            
            # Create SSML for better control
            ssml = self._create_ssml(text)
            
            result = synthesizer.speak_ssml_async(ssml).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return result.audio_data
            else:
                raise Exception(f"Azure TTS failed: {result.reason}")
                
        except Exception as e:
            logger.error(f"Azure TTS error: {e}")
            raise
    
    async def _synthesize_with_google(self, text: str) -> bytes:
        """Synthesize speech using Google Text-to-Speech."""
        if not hasattr(self, 'google_client'):
            raise Exception("Google TTS not configured")
            
        try:
            # Prepare input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Voice selection - Hindi with good English support
            voice = texttospeech.VoiceSelectionParams(
                language_code="hi-IN",
                name="hi-IN-Standard-A",  # Female voice
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
            )
            
            # Audio config
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=22050
            )
            
            response = self.google_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            return response.audio_content
            
        except Exception as e:
            logger.error(f"Google TTS error: {e}")
            raise
    
    def _prepare_hinglish_text(self, text: str) -> str:
        """Prepare text for better Hinglish pronunciation.
        
        Args:
            text (str): Original text
            
        Returns:
            str: Text optimized for TTS
        """
        # Break long sentences into shorter chunks
        if len(text) > 200:
            # Split at sentence boundaries
            sentences = text.split('। ')
            if len(sentences) == 1:
                sentences = text.split('. ')
            
            # Limit to avoid too long responses
            if len(sentences) > 3:
                text = '. '.join(sentences[:3]) + '.'
        
        # Add pauses for better clarity
        text = text.replace(',', ', ')
        text = text.replace('।', '। ')
        
        # Ensure text ends with proper punctuation
        if not text.endswith(('.', '।', '?', '!')):
            text += '।'
            
        return text
    
    def _create_ssml(self, text: str) -> str:
        """Create SSML markup for better speech control.
        
        Args:
            text (str): Text to wrap in SSML
            
        Returns:
            str: SSML markup
        """
        return f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="hi-IN">
            <voice name="hi-IN-SwaraNeural">
                <prosody rate="{settings.speech_rate}" pitch="{settings.speech_pitch:+.1f}st">
                    {text}
                </prosody>
            </voice>
        </speak>
        """
    
    def get_hinglish_prompts(self) -> dict:
        """Get conversation prompts in Hinglish for different stages.
        
        Returns:
            dict: Mapping of conversation stages to Hinglish prompts
        """
        return {
            "greeting": "Namaste! Main aapka voice assistant hun job interview ke liye। Yeh call record ho rahi hai। Kya aap tayaar hain?",
            
            "pincode": "Aap kahan rehte hain? Apna pincode ya area batayiye।",
            "pincode_confirm": "Aapne {value} kaha, sahi hai na?",
            "pincode_retry": "Pincode samajh nahi aaya। 6 digit number boliye jaise 110001।",
            
            "availability": "Aap kab se kaam shuru kar sakte hain? Aaj, kal ya koi aur din?",
            "availability_confirm": "Toh aap {value} se start kar sakte hain, correct?",
            "availability_retry": "Date samajh nahi aayi। Aaj, kal, parso - aise boliye।",
            
            "shift": "Aap kaunse time pe kaam karna chahte hain? Morning, evening ya night?",
            "shift_confirm": "Aap {value} shift prefer karte hain, right?",
            "shift_retry": "Shift samajh nahi aayi। Morning, afternoon, evening ya night - koi ek choose kariye।",
            
            "salary": "Aapko kitni salary chahiye har mahine? Rupees mein batayiye।",
            "salary_confirm": "Aapki expected salary {value} rupees per month hai, sahi?",
            "salary_retry": "Salary amount clear nahi hai। Number mein boliye jaise 15 hazaar।",
            
            "languages": "Aap kaunsi languages bol sakte hain? Hindi, English ya koi aur?",
            "languages_confirm": "Aap {value} bol sakte hain, confirm hai?",
            "languages_retry": "Languages samajh nahi aayi। Hindi, English - aise batayiye।",
            
            "two_wheeler": "Kya aapke paas bike ya scooter hai?",
            "two_wheeler_confirm": "Aapke paas two wheeler {value} hai, right?",
            "two_wheeler_retry": "Haan ya nahi mein jawab dijiye। Bike hai ya nahi?",
            
            "experience": "Aapko kitna kaam ka experience hai? Kitne saal ya mahine?",
            "experience_confirm": "Aapka total experience {value} hai, correct?",
            "experience_retry": "Experience time samajh nahi aaya। Months ya years mein batayiye।",
            
            "summary": "Perfect! Main aapki details note kar li। Aapko {locality} area mein, {salary} salary range mein best jobs bhejunga। Thank you!",
            
            "error_generic": "Sorry, samajh nahi aaya। Dobara boliye।",
            "error_noise": "Background noise zyada hai। Shant jagah se baat kariye।",
            "error_timeout": "Aapki awaaz nahi aa rahi। Phone check kariye।",
            
            "goodbye": "Aapka interview complete ho gaya। Job matches SMS mein aayenge। Dhanyawad!"
        } 