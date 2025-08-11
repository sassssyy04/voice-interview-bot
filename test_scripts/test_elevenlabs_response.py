#!/usr/bin/env python3
"""Test script to see what ElevenLabs API actually returns."""

import os
import asyncio
import httpx
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

async def test_elevenlabs_response():
    """Test ElevenLabs API to see the actual response structure."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    
    if not api_key:
        print("❌ ELEVENLABS_API_KEY not found")
        return
    
    print("🔍 Testing ElevenLabs API response structure...")
    
    # Test with a known audio file
    audio_file = "test_harness/generated_audio/pin_001_clean.wav"
    
    if not Path(audio_file).exists():
        print(f"❌ Audio file not found: {audio_file}")
        return
    
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    
    headers = {
        "xi-api-key": api_key
    }
    
    with open(audio_file, 'rb') as f:
        audio_data = f.read()
    
    print(f"📁 Testing with: {Path(audio_file).name} ({len(audio_data)} bytes)")
    
    files = {
        "file": ("audio.wav", audio_data, "audio/wav")
    }
    
    data = {
        "model_id": "scribe_v1",
        "language_code": "hi",
        "tag_audio_events": "false",
        "timestamp_granularity": "word",
        "diarize": "false"
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, files=files, data=data)
            
            print(f"📡 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"\n📋 COMPLETE API RESPONSE:")
                print(f"🔍 All fields: {list(result.keys())}")
                
                for key, value in result.items():
                    print(f"  • {key}: {value}")
                
                print(f"\n📊 CONFIDENCE ANALYSIS:")
                print(f"  • language_probability: {result.get('language_probability', 'NOT_FOUND')}")
                print(f"  • confidence: {result.get('confidence', 'NOT_FOUND')}")
                print(f"  • quality_score: {result.get('quality_score', 'NOT_FOUND')}")
                print(f"  • detection_confidence: {result.get('detection_confidence', 'NOT_FOUND')}")
                
                # Check if there's word-level data with confidence
                words = result.get("words", [])
                if words:
                    print(f"\n📝 WORD-LEVEL DATA:")
                    print(f"  • Found {len(words)} words")
                    if len(words) > 0:
                        first_word = words[0]
                        print(f"  • First word structure: {list(first_word.keys())}")
                        print(f"  • Sample word: {first_word}")
                
            else:
                print(f"❌ Error: {response.text}")
                
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_elevenlabs_response()) 