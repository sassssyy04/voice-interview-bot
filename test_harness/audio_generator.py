"""Audio generation system for creating synthetic Hinglish test audio with noise."""

import os
import yaml
import asyncio
import random
import wave
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from pydub import AudioSegment
import httpx
from loguru import logger
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class AudioGenerator:
    """Generates synthetic Hinglish audio with various accents and noise levels."""
    
    def __init__(self, config_path: str = "utterances.yaml"):
        """Initialize the audio generator.
        
        Args:
            config_path (str): Path to utterances configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.output_dir = Path("test_harness/generated_audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Eleven Labs TTS
        self.eleven_labs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.eleven_labs_base_url = "https://api.elevenlabs.io/v1"
        
        if not self.eleven_labs_api_key:
            logger.warning("ELEVENLABS_API_KEY not found, will use fallback TTS")
        else:
            logger.info("ELEVENLABS_API_KEY found, will use Eleven Labs TTS")
    
    def _load_config(self) -> Dict:
        """Load utterances configuration from YAML file.
        
        Returns:
            Dict: Configuration dictionary
        """
        with open(self.config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    async def generate_speech(self, text: str, voice_id: str, output_path: str) -> bool:
        """Generate speech audio using Eleven Labs TTS.
        
        Args:
            text (str): Text to synthesize
            voice_id (str): Eleven Labs voice ID
            output_path (str): Output file path
            
        Returns:
            bool: Success status
        """
        if not self.eleven_labs_api_key:
            return self._generate_fallback_audio(text, output_path)
        
        try:
            # Eleven Labs TTS API call
            url = f"{self.eleven_labs_base_url}/text-to-speech/{voice_id}"
            
            headers = {
                "Accept": "audio/wav",
                "Content-Type": "application/json",
                "xi-api-key": self.eleven_labs_api_key
            }
            
            data = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5,
                    "style": 0.5,
                    "use_speaker_boost": True
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=data, headers=headers, timeout=30.0)
                
                if response.status_code == 200:
                    # Save the audio directly - don't try to process with pydub
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    
                    # Just check if file has reasonable content
                    file_size = os.path.getsize(output_path)
                    if file_size > 1000:  # At least 1KB for real audio
                        logger.info(f"Generated Eleven Labs audio: {output_path} ({file_size} bytes)")
                        return True
                    else:
                        logger.warning(f"Generated file too small ({file_size} bytes), using fallback")
                        return self._generate_fallback_audio(text, output_path)
                        
                else:
                    logger.error(f"Eleven Labs TTS error: {response.status_code} - {response.text}")
                    return self._generate_fallback_audio(text, output_path)
                    
        except Exception as e:
            logger.error(f"Eleven Labs TTS error: {e}")
            return self._generate_fallback_audio(text, output_path)
    
    def _generate_fallback_audio(self, text: str, output_path: str) -> bool:
        """Generate fallback audio using pydub with synthesized beeps and duration.
        
        Args:
            text (str): Text to use for duration calculation
            output_path (str): Output file path
            
        Returns:
            bool: Success status
        """
        try:
            # Estimate duration: ~150 words per minute for Hinglish
            word_count = len(text.split())
            duration_ms = max(6000, word_count * 400)  # Min 6 seconds
            
            # Generate audio with some content instead of just silence
            # Create a simple tone pattern to simulate speech rhythm
            base_audio = AudioSegment.silent(duration=duration_ms)
            
            # Add some low-volume tones to simulate speech patterns
            from pydub.generators import Sine
            
            # Create word-like segments with brief tones
            segment_duration = 300  # 300ms per "word" - longer segments
            pause_duration = 150    # 150ms pause between words
            
            current_pos = 0
            for i in range(word_count):
                if current_pos + segment_duration > duration_ms:
                    break
                    
                # Generate a brief tone (simulating a word)
                tone_freq = 300 + (i % 5) * 75  # Vary frequency (300-675Hz)
                tone = Sine(tone_freq).to_audio_segment(duration=segment_duration)
                tone = tone - 5  # Much louder (-5dB instead of -30dB)
                
                # Add some harmonic content to make it more speech-like
                if i % 2 == 0:
                    harmonic = Sine(tone_freq * 1.5).to_audio_segment(duration=segment_duration) - 15
                    tone = tone.overlay(harmonic)
                
                # Overlay the tone at current position
                base_audio = base_audio.overlay(tone, position=current_pos)
                current_pos += segment_duration + pause_duration
            
            # Boost the overall volume significantly
            base_audio = base_audio + 20  # Boost by 20dB to make it very audible
            
            # Ensure 16kHz mono WAV format
            base_audio = base_audio.set_frame_rate(16000).set_channels(1)
            base_audio.export(output_path, format="wav")
            
            logger.warning(f"Generated fallback audio with content: {output_path} ({duration_ms}ms)")
            return True
            
        except Exception as e:
            logger.error(f"Fallback audio generation failed: {e}")
            # Last resort - create basic silence
            try:
                word_count = len(text.split())
                duration_ms = max(6000, word_count * 400)
                audio = AudioSegment.silent(duration=duration_ms)
                audio = audio.set_frame_rate(16000).set_channels(1)
                audio.export(output_path, format="wav")
                logger.warning(f"Generated basic silence fallback: {output_path}")
                return True
            except Exception as e2:
                logger.error(f"Even basic fallback failed: {e2}")
                return False
    
    def add_background_noise(self, input_wav: str, output_wav: str, noise_level_db: int = -20) -> bool:
        """Add white noise to audio file using numpy approach.
        
        Args:
            input_wav (str): Path to input clean audio file
            output_wav (str): Path to output noisy audio file
            noise_level_db (int): Noise level in dB
            
        Returns:
            bool: Success status
        """
        try:
            # Load the original audio
            audio = AudioSegment.from_wav(input_wav)

            # Generate white noise with the same duration
            # pydub uses milliseconds, so convert length
            noise = AudioSegment(
                (np.random.randn(len(audio.get_array_of_samples())) * 32767).astype(np.int16).tobytes(),
                frame_rate=audio.frame_rate,
                sample_width=audio.sample_width,
                channels=audio.channels
            )

            # Adjust noise level
            noise = noise - noise.dBFS + noise_level_db

            # Overlay the noise on the original audio
            mixed = audio.overlay(noise)

            # Save to output file
            mixed.export(output_wav, format="wav")
            logger.info(f"Successfully added white noise ({noise_level_db}dB): {output_wav}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add noise to {input_wav}: {e}")
            return False

    def add_background_noise_old(self, audio_path: str, noise_config: Dict) -> str:
        """Legacy noise addition function - kept for reference."""
        # This is the old complex method - now we use the simpler numpy approach
        return audio_path
    
    def _create_synthetic_speech(self, duration_ms: int) -> AudioSegment:
        """Create synthetic speech-like audio pattern.
        
        Args:
            duration_ms (int): Duration in milliseconds
            
        Returns:
            AudioSegment: Synthetic speech audio
        """
        from pydub.generators import Sine
        
        # Create base silence
        base_audio = AudioSegment.silent(duration=duration_ms)
        
        # Add speech-like segments with much higher volume
        current_pos = 0
        segment_duration = 300  # 300ms per "word" - longer segments
        pause_duration = 150    # 150ms pause between words
        
        word_count = duration_ms // (segment_duration + pause_duration)
        
        for i in range(int(word_count)):
            if current_pos + segment_duration > duration_ms:
                break
                
            # Generate varying frequency tone to simulate speech
            tone_freq = 300 + (i % 8) * 75  # Vary between 300-825Hz (more speech-like)
            tone = Sine(tone_freq).to_audio_segment(duration=segment_duration)
            tone = tone - 10  # Much louder than before (-10dB instead of -30dB)
            
            # Add some modulation to make it more speech-like
            if i % 3 == 0:  # Every third "word" add a harmonic
                harmonic = Sine(tone_freq * 1.5).to_audio_segment(duration=segment_duration) - 20
                tone = tone.overlay(harmonic)
            
            # Overlay the tone at current position
            base_audio = base_audio.overlay(tone, position=current_pos)
            current_pos += segment_duration + pause_duration
        
        # Make the overall speech much louder
        base_audio = base_audio + 15  # Boost by 15dB
        
        return base_audio
    
    def _mix_audio_with_noise(self, audio_path: str, noise_path: str, output_path: str, noise_config: Dict) -> bool:
        """Legacy function - not used in current pydub implementation.
        
        Args:
            audio_path (str): Path to clean audio file
            noise_path (str): Path to noise sample file
            output_path (str): Path for output mixed file
            noise_config (Dict): Noise configuration with SNR
            
        Returns:
            bool: Success status
        """
        # This function is replaced by the simpler pydub approach in add_background_noise
        return False
    
    def _generate_noise(self, duration_ms: float, noise_type: str) -> Optional[AudioSegment]:
        """Legacy function kept for compatibility - not used in current implementation.
        
        Args:
            duration_ms (float): Duration in milliseconds
            noise_type (str): Type of noise
            
        Returns:
            Optional[AudioSegment]: None (not used)
        """
        # This function is not used in the current raw audio implementation
        return None
    
    async def generate_test_audio(self, utterance_id: str = None) -> List[Dict]:
        """Generate audio files for test utterances.
        
        Args:
            utterance_id (str, optional): Generate only specific utterance
            
        Returns:
            List[Dict]: Generated audio file metadata
        """
        utterances = self.config["utterances"]
        audio_config = self.config["audio_config"]
        
        if utterance_id:
            utterances = [u for u in utterances if u["id"] == utterance_id]
        
        logger.info(f"Generating audio for {len(utterances)} utterances")
        logger.info(f"Eleven Labs API key present: {bool(self.eleven_labs_api_key)}")
        
        generated_files = []
        
        for utterance in utterances:
            try:
                logger.info(f"Processing utterance: {utterance['id']}")
                
                # Get voice for accent
                accent = utterance["accent"]
                voice_id = audio_config["voices"].get(accent, "90ipbRoKi4CpHXvKVtl0")
                
                logger.info(f"Using voice ID: {voice_id} for accent: {accent}")
                
                # Generate clean audio
                clean_path = str(self.output_dir / f"{utterance['id']}_clean.wav")
                logger.info(f"Generating audio to: {clean_path}")
                
                success = await self.generate_speech(
                    utterance["text"], 
                    voice_id, 
                    clean_path
                )
                
                if not success:
                    logger.error(f"Failed to generate audio for {utterance['id']}")
                    continue
                
                # Check if file was actually created and has content
                if os.path.exists(clean_path):
                    file_size = os.path.getsize(clean_path)
                    logger.info(f"Generated file size: {file_size} bytes")
                    if file_size == 0:
                        logger.error(f"Generated file is empty: {clean_path}")
                        continue
                else:
                    logger.error(f"Generated file does not exist: {clean_path}")
                    continue
                
                # Add noise if specified and if utterance ID ends with "001" or "003"
                noise_level = utterance["noise_level"]
                noise_config = audio_config["noise"][noise_level]
                
                final_path = clean_path  # Start with clean path
                
                # Check if utterance ID ends with "001" or "003" (apply noise only to these)
                utterance_id = utterance["id"]
                if utterance_id.endswith("_001") or utterance_id.endswith("_003"):
                    # Determine noise level in dB based on noise type
                    noise_db_levels = {
                        "construction": -15,  # High noise (louder)
                        "office": -20,        # Medium noise  
                        "traffic": -25        # Low noise (quieter)
                    }
                    
                    noise_type = noise_config.get("type")
                    if noise_type:
                        noise_level_db = noise_db_levels.get(noise_type, -20)
                        noisy_path = clean_path.replace("_clean.wav", "_noisy.wav")
                        
                        success = self.add_background_noise(clean_path, noisy_path, noise_level_db)
                        if success:
                            final_path = noisy_path
                            logger.info(f"Added {noise_type} noise to {utterance_id}")
                        else:
                            logger.warning(f"Failed to add noise to {utterance_id}, using clean version")
                else:
                    logger.info(f"Skipping noise for {utterance_id} (only adding to _001 and _003)")
                
                # Estimate duration from file size (avoid pydub/FFmpeg issues)
                try:
                    file_size = os.path.getsize(final_path)
                    # Rough estimate: 16kHz mono WAV = ~32KB per second
                    duration_s = max(6.0, file_size / 32000)
                    expected_range = utterance["duration_range"]
                    
                    logger.info(f"Estimated audio duration: {duration_s:.1f}s (expected: {expected_range})")
                    
                    if not (expected_range[0] <= duration_s <= expected_range[1] + 5):
                        logger.warning(
                            f"Estimated duration {duration_s:.1f}s outside expected range "
                            f"{expected_range} for {utterance['id']}"
                        )
                    
                except Exception as duration_error:
                    logger.warning(f"Could not estimate duration for {final_path}: {duration_error}")
                    duration_s = 6.0  # Default fallback
                
                generated_files.append({
                    "id": utterance["id"],
                    "audio_path": final_path,
                    "duration_seconds": duration_s,
                    "transcript": utterance["transcript"],
                    "entities": utterance["entities"],
                    "accent": accent,
                    "noise_level": noise_level
                })
                
                logger.info(f"✓ Successfully generated test audio: {utterance['id']}")
                    
            except Exception as e:
                logger.error(f"Error generating audio for {utterance['id']}: {e}")
        
        logger.info(f"Generated {len(generated_files)} audio files successfully")
        return generated_files
    
    def create_audio_manifest(self, generated_files: List[Dict]) -> str:
        """Create manifest file for generated audio.
        
        Args:
            generated_files (List[Dict]): Generated audio metadata
            
        Returns:
            str: Path to manifest file
        """
        manifest_path = self.output_dir / "audio_manifest.yaml"
        
        manifest = {
            "generated_at": str(asyncio.get_event_loop().time()),
            "total_files": len(generated_files),
            "files": generated_files
        }
        
        with open(manifest_path, 'w', encoding='utf-8') as file:
            yaml.dump(manifest, file, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Created audio manifest: {manifest_path}")
        return str(manifest_path)


async def main():
    """Generate all test audio files."""
    generator = AudioGenerator()
    
    logger.info("Starting audio generation...")
    generated_files = await generator.generate_test_audio()
    
    if generated_files:
        manifest_path = generator.create_audio_manifest(generated_files)
        logger.info(f"Generated {len(generated_files)} audio files")
        logger.info(f"Manifest saved to: {manifest_path}")
    else:
        logger.error("No audio files generated")


if __name__ == "__main__":
    asyncio.run(main()) 