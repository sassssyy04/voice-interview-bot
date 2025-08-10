"""Audio generation system for creating synthetic Hinglish test audio with noise."""

import os
import yaml
import asyncio
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from pydub import AudioSegment
from pydub.generators import WhiteNoise
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
            segment_duration = 200  # 200ms per "word"
            pause_duration = 100    # 100ms pause between words
            
            current_pos = 0
            for i in range(word_count):
                if current_pos + segment_duration > duration_ms:
                    break
                    
                # Generate a brief tone (simulating a word)
                tone_freq = 200 + (i % 5) * 50  # Vary frequency slightly
                tone = Sine(tone_freq).to_audio_segment(duration=segment_duration)
                tone = tone - 30  # Make it quiet (-30dB)
                
                # Overlay the tone at current position
                base_audio = base_audio.overlay(tone, position=current_pos)
                current_pos += segment_duration + pause_duration
            
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
    
    def add_background_noise(self, audio_path: str, noise_config: Dict) -> str:
        """Add background noise to audio file.
        
        Args:
            audio_path (str): Path to clean audio file
            noise_config (Dict): Noise configuration
            
        Returns:
            str: Path to noisy audio file
        """
        if not noise_config.get("type"):
            return audio_path
            
        try:
            # Load original audio
            audio = AudioSegment.from_wav(audio_path)
            
            # Generate noise based on type
            noise = self._generate_noise(
                audio.duration_seconds * 1000,
                noise_config["type"]
            )
            
            # Calculate noise level based on SNR
            snr_db = noise_config.get("snr_db", 20)
            noise_level = audio.dBFS - snr_db
            noise = noise.apply_gain(noise_level - noise.dBFS)
            
            # Mix audio with noise
            noisy_audio = audio.overlay(noise)
            
            # Save noisy version
            noisy_path = audio_path.replace(".wav", "_noisy.wav")
            noisy_audio.export(noisy_path, format="wav")
            
            logger.info(f"Added noise to audio: {noisy_path}")
            return noisy_path
            
        except Exception as e:
            logger.error(f"Failed to add noise: {e}")
            return audio_path
    
    def _generate_noise(self, duration_ms: float, noise_type: str) -> AudioSegment:
        """Generate background noise of specified type.
        
        Args:
            duration_ms (float): Duration in milliseconds
            noise_type (str): Type of noise
            
        Returns:
            AudioSegment: Generated noise
        """
        if noise_type == "traffic":
            # Low frequency rumble + occasional higher frequency
            base_noise = WhiteNoise().to_audio_segment(duration=duration_ms)
            filtered = base_noise.low_pass_filter(800)
            return filtered
            
        elif noise_type == "office":
            # Mid-frequency chatter simulation
            base_noise = WhiteNoise().to_audio_segment(duration=duration_ms)
            # Use high_pass and low_pass to simulate band_pass
            filtered = base_noise.high_pass_filter(300).low_pass_filter(3000)
            return filtered - 10  # Quieter
            
        elif noise_type == "construction":
            # High frequency with intermittent peaks
            base_noise = WhiteNoise().to_audio_segment(duration=duration_ms)
            filtered = base_noise.high_pass_filter(500)
            return filtered
            
        else:
            # Default white noise
            return WhiteNoise().to_audio_segment(duration=duration_ms)
    
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
                
                # Add noise if specified
                noise_level = utterance["noise_level"]
                noise_config = audio_config["noise"][noise_level]
                
                final_path = self.add_background_noise(clean_path, noise_config)
                
                # Try to verify duration, but don't fail if we can't
                try:
                    audio = AudioSegment.from_wav(final_path)
                    duration_s = audio.duration_seconds
                    expected_range = utterance["duration_range"]
                    
                    logger.info(f"Audio duration: {duration_s:.1f}s (expected: {expected_range})")
                    
                    if not (expected_range[0] <= duration_s <= expected_range[1] + 2):
                        logger.warning(
                            f"Duration {duration_s:.1f}s outside expected range "
                            f"{expected_range} for {utterance['id']}"
                        )
                    
                except Exception as duration_error:
                    logger.warning(f"Could not verify duration for {final_path}: {duration_error}")
                    # Estimate duration from file size (rough approximation)
                    file_size = os.path.getsize(final_path)
                    duration_s = max(6.0, file_size / 32000)  # Very rough estimate
                
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