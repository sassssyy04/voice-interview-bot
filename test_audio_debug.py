#!/usr/bin/env python3
"""Debug script to test audio generation."""

import asyncio
import os
from pathlib import Path
from pydub import AudioSegment
from pydub.generators import Sine, WhiteNoise
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_basic_audio_creation():
    """Test basic audio creation with pydub."""
    print("Testing basic audio creation...")
    
    try:
        # Create output directory
        output_dir = Path("test_audio_output")
        output_dir.mkdir(exist_ok=True)
        
        # Test 1: Generate silence
        print("1. Creating silence...")
        silence = AudioSegment.silent(duration=5000)  # 5 seconds
        silence = silence.set_frame_rate(16000).set_channels(1)
        silence_path = output_dir / "test_silence.wav"
        silence.export(str(silence_path), format="wav")
        print(f"   Created: {silence_path} ({os.path.getsize(silence_path)} bytes)")
        
        # Test 2: Generate tone
        print("2. Creating tone...")
        tone = Sine(440).to_audio_segment(duration=3000)  # 3 seconds, 440Hz
        tone = tone.set_frame_rate(16000).set_channels(1)
        tone_path = output_dir / "test_tone.wav"
        tone.export(str(tone_path), format="wav")
        print(f"   Created: {tone_path} ({os.path.getsize(tone_path)} bytes)")
        
        # Test 3: Generate speech-like pattern
        print("3. Creating speech-like pattern...")
        base_audio = AudioSegment.silent(duration=8000)  # 8 seconds
        
        # Add word-like segments
        current_pos = 0
        word_count = 10
        for i in range(word_count):
            segment_duration = 200  # 200ms per "word"
            pause_duration = 150    # 150ms pause
            
            if current_pos + segment_duration > 8000:
                break
                
            # Generate varying frequency tone
            tone_freq = 200 + (i % 5) * 50
            tone = Sine(tone_freq).to_audio_segment(duration=segment_duration)
            tone = tone - 25  # Make it quieter (-25dB)
            
            base_audio = base_audio.overlay(tone, position=current_pos)
            current_pos += segment_duration + pause_duration
        
        # Add some background noise
        noise = WhiteNoise().to_audio_segment(duration=8000)
        noise = noise - 35  # Very quiet background noise
        base_audio = base_audio.overlay(noise)
        
        base_audio = base_audio.set_frame_rate(16000).set_channels(1)
        speech_path = output_dir / "test_speech_pattern.wav"
        base_audio.export(str(speech_path), format="wav")
        print(f"   Created: {speech_path} ({os.path.getsize(speech_path)} bytes)")
        
        # Test 4: Add noise to existing audio
        print("4. Adding noise to tone...")
        traffic_noise = WhiteNoise().to_audio_segment(duration=3000)
        traffic_noise = traffic_noise.low_pass_filter(800) - 20  # Traffic-like noise
        
        noisy_tone = tone.overlay(traffic_noise)
        noisy_path = output_dir / "test_noisy_tone.wav"
        noisy_tone.export(str(noisy_path), format="wav")
        print(f"   Created: {noisy_path} ({os.path.getsize(noisy_path)} bytes)")
        
        print("\n✓ All audio tests passed!")
        print(f"Check the files in: {output_dir.absolute()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Audio test failed: {e}")
        return False

async def test_eleven_labs_api():
    """Test Eleven Labs API connection (if API key available)."""
    print("\nTesting Eleven Labs API...")
    
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("No ELEVENLABS_API_KEY found - skipping API test")
        return False
    
    try:
        import httpx
        
        # Test API connection with a simple TTS request instead of voices list
        # (voices list requires special permissions)
        voice_id = "pNInz6obpgDQGcFmaJgB"  # Adam voice (free tier)
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/wav",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        
        data = {
            "text": "Test",
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data, headers=headers, timeout=10.0)
            
            if response.status_code == 200:
                audio_size = len(response.content)
                print(f"✓ API connection successful - generated {audio_size} bytes of audio")
                return True
            elif response.status_code == 401:
                print(f"✗ API authentication failed - check your API key")
                return False
            elif response.status_code == 422:
                print(f"✗ API request invalid - check voice ID or parameters")
                return False
            else:
                print(f"✗ API error: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        print(f"✗ API test failed: {e}")
        return False

async def test_simple_tts():
    """Test simple TTS generation."""
    print("\nTesting simple TTS generation...")
    
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("No ELEVENLABS_API_KEY - using fallback audio")
        
        # Test fallback audio generation
        output_dir = Path("test_audio_output")
        output_dir.mkdir(exist_ok=True)
        
        text = "Hello this is a test"
        word_count = len(text.split())
        duration_ms = max(3000, word_count * 400)
        
        # Generate speech-like audio
        base_audio = AudioSegment.silent(duration=duration_ms)
        current_pos = 0
        
        for i in range(word_count):
            segment_duration = 200
            pause_duration = 100
            
            if current_pos + segment_duration > duration_ms:
                break
                
            tone_freq = 200 + (i % 5) * 50
            tone = Sine(tone_freq).to_audio_segment(duration=segment_duration)
            tone = tone - 30
            
            base_audio = base_audio.overlay(tone, position=current_pos)
            current_pos += segment_duration + pause_duration
        
        base_audio = base_audio.set_frame_rate(16000).set_channels(1)
        fallback_path = output_dir / "fallback_tts_test.wav"
        base_audio.export(str(fallback_path), format="wav")
        
        print(f"✓ Generated fallback audio: {fallback_path} ({os.path.getsize(fallback_path)} bytes)")
        return True
    
    try:
        import httpx
        
        # Simple TTS request
        voice_id = "pNInz6obpgDQGcFmaJgB"  # Adam voice
        text = "Hello this is a test of the Eleven Labs TTS system"
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "Accept": "audio/wav",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data, headers=headers, timeout=30.0)
            
            if response.status_code == 200:
                output_dir = Path("test_audio_output")
                output_dir.mkdir(exist_ok=True)
                
                # Save WAV directly (no conversion needed)
                wav_path = output_dir / "tts_test.wav"
                with open(wav_path, "wb") as f:
                    f.write(response.content)
                
                # Verify and convert to proper format if needed
                audio = AudioSegment.from_wav(str(wav_path))
                if audio.frame_rate != 16000 or audio.channels != 1:
                    audio = audio.set_frame_rate(16000).set_channels(1)
                    audio.export(str(wav_path), format="wav")
                
                print(f"✓ Generated TTS audio: {wav_path} ({os.path.getsize(wav_path)} bytes)")
                return True
            else:
                print(f"✗ TTS error: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        print(f"✗ TTS test failed: {e}")
        return False

async def main():
    """Run all audio tests."""
    print("=== Audio Generation Debug Tests ===\n")
    
    # Test 1: Basic audio creation
    basic_test = test_basic_audio_creation()
    
    # Test 2: Eleven Labs API
    api_test = await test_eleven_labs_api()
    
    # Test 3: Simple TTS
    tts_test = await test_simple_tts()
    
    print(f"\n=== Test Results ===")
    print(f"Basic Audio Creation: {'✓ PASS' if basic_test else '✗ FAIL'}")
    print(f"Eleven Labs API:      {'✓ PASS' if api_test else '✗ FAIL'}")
    print(f"TTS Generation:       {'✓ PASS' if tts_test else '✗ FAIL'}")
    
    if basic_test:
        print("\n🎵 Audio generation is working!")
        print("Check the 'test_audio_output' folder for generated files.")
    else:
        print("\n❌ Audio generation has issues. Check dependencies:")
        print("   pip install pydub")

if __name__ == "__main__":
    asyncio.run(main()) 