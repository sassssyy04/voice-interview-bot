#!/usr/bin/env python3
"""Direct TTS test - saves Eleven Labs audio without any processing."""

import os
import asyncio
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_direct_tts():
    """Test Eleven Labs TTS and save directly without processing."""
    
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("❌ No ELEVENLABS_API_KEY found")
        return False
    
    print("🎙️  Testing Eleven Labs TTS directly...")
    
    # Test cases
    test_cases = [
        {
            "text": "Main Delhi mein rehta hun, mera pincode hai 110001",
            "voice_id": "90ipbRoKi4CpHXvKVtl0",  # Custom Hinglish voice
            "filename": "test_direct_hindi.wav"
        },
        {
            "text": "Hello, this is a test of the Eleven Labs text to speech system",
            "voice_id": "90ipbRoKi4CpHXvKVtl0",  # Custom Hinglish voice
            "filename": "test_direct_english.wav"
        },
        {
            "text": "Bhai main fifteen thousand per month expect karta hun",
            "voice_id": "90ipbRoKi4CpHXvKVtl0",  # Custom Hinglish voice
            "filename": "test_direct_hinglish.wav"
        }
    ]
    
    success_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: '{test_case['text'][:50]}...'")
        
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{test_case['voice_id']}"
            
            headers = {
                "Accept": "audio/wav",
                "Content-Type": "application/json", 
                "xi-api-key": api_key
            }
            
            data = {
                "text": test_case["text"],
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5,
                    "style": 0.5,
                    "use_speaker_boost": True
                }
            }
            
            async with httpx.AsyncClient() as client:
                print("   📡 Making API request...")
                response = await client.post(url, json=data, headers=headers, timeout=30.0)
                
                if response.status_code == 200:
                    # Save directly without any processing
                    output_path = f"test_audio_output/{test_case['filename']}"
                    os.makedirs("test_audio_output", exist_ok=True)
                    
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    
                    file_size = os.path.getsize(output_path)
                    print(f"   ✅ Success! Saved {file_size:,} bytes to {output_path}")
                    
                    # Simple content check
                    if file_size > 10000:  # At least 10KB for real speech
                        print(f"   🎵 File size looks good for speech audio")
                        success_count += 1
                    else:
                        print(f"   ⚠️  File size seems small for speech ({file_size} bytes)")
                    
                else:
                    print(f"   ❌ API Error: {response.status_code}")
                    print(f"   Response: {response.text}")
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n📊 Results: {success_count}/{len(test_cases)} tests successful")
    
    if success_count > 0:
        print("\n🎉 TTS is working! Check the test_audio_output folder.")
        print("   You should hear actual speech (not just tones) when you play these files.")
        return True
    else:
        print("\n❌ TTS failed. Check your API key and connection.")
        return False

if __name__ == "__main__":
    asyncio.run(test_direct_tts()) 