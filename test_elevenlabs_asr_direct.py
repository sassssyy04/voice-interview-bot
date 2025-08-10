#!/usr/bin/env python3
"""Direct test of ElevenLabs ASR to debug transcription issues."""

import os
import asyncio
import httpx
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

async def test_elevenlabs_asr_direct():
    """Test ElevenLabs ASR directly."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    
    if not api_key:
        print("❌ ELEVENLABS_API_KEY not found")
        return
    
    print("🎙️ Testing ElevenLabs ASR directly...")
    print(f"API key: {'✅ Found' if api_key else '❌ Missing'}")
    
    # Test files to try
    test_files = [
        "test_harness/generated_audio/pin_001_clean.wav",
        "test_harness/generated_audio/sal_001_clean.wav",
        "test_harness/generated_audio/vehicle_001_clean.wav"
    ]
    
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    
    headers = {
        "xi-api-key": api_key
    }
    
    for audio_file in test_files:
        if not Path(audio_file).exists():
            print(f"❌ File not found: {audio_file}")
            continue
            
        print(f"\n🎵 Testing: {Path(audio_file).name}")
        
        with open(audio_file, 'rb') as f:
            audio_data = f.read()
        
        print(f"   File size: {len(audio_data)} bytes")
        
        try:
            files = {
                "file": ("audio.wav", audio_data, "audio/wav")
            }
            
            data = {
                "model_id": "scribe_v1",
                "language_code": "hi",  # Hindi for Hinglish
                "tag_audio_events": "false",
                "timestamp_granularity": "word",
                "diarize": "false"
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, headers=headers, files=files, data=data)
                
                print(f"   Status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    text = result.get("text", "").strip()
                    language_code = result.get("language_code", "")
                    language_probability = result.get("language_probability", 0.0)
                    
                    print(f"   ✅ SUCCESS!")
                    print(f"   Text: '{text}'")
                    print(f"   Language: {language_code} ({language_probability:.2f})")
                    
                    if text:
                        print(f"   📝 ElevenLabs ASR IS WORKING!")
                    else:
                        print(f"   ⚠️  ASR returned empty text - audio might be silent/unclear")
                        
                elif response.status_code == 401:
                    print(f"   ❌ Unauthorized - check API key")
                    break
                elif response.status_code == 422:
                    print(f"   ❌ Validation error - check audio format")
                    print(f"   Response: {response.text}")
                else:
                    print(f"   ❌ Error: {response.text}")
                    
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    print(f"\n🔍 Summary: ElevenLabs ASR test completed")

if __name__ == "__main__":
    asyncio.run(test_elevenlabs_asr_direct()) 