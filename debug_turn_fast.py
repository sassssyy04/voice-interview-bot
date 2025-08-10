#!/usr/bin/env python3
"""Debug script to test turn-fast with known good candidate_id."""

import asyncio
import httpx
from pathlib import Path

async def debug_turn_fast():
    """Test turn-fast with verified candidate_id."""
    base_url = "http://localhost:8000"
    
    print("🔍 Debugging turn-fast endpoint...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Start a conversation
        print("1. Starting conversation...")
        response = await client.post(f"{base_url}/api/v1/conversation/start")
        
        if response.status_code != 200:
            print(f"   ❌ Failed to start: {response.status_code}")
            return
            
        data = response.json()
        candidate_id = data.get("candidate_id")
        print(f"   ✅ Started: {candidate_id}")
        
        # Verify conversation exists
        print("2. Verifying conversation exists...")
        status_response = await client.get(f"{base_url}/api/v1/conversation/{candidate_id}/status")
        if status_response.status_code == 200:
            print("   ✅ Conversation verified")
        else:
            print(f"   ❌ Verification failed: {status_response.status_code}")
            return
        
        # Try turn-fast with the same candidate_id
        print("3. Testing turn-fast...")
        audio_path = "test_harness/generated_audio/pin_001_clean.wav"
        
        if not Path(audio_path).exists():
            print(f"   ❌ Audio file not found: {audio_path}")
            return
        
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        files = {
            'audio_file': ('recording.wav', audio_data, 'audio/wav')
        }
        
        url = f"{base_url}/api/v1/conversation/{candidate_id}/turn-fast"
        print(f"   Sending to: {url}")
        print(f"   Candidate ID: {candidate_id}")
        print(f"   Audio size: {len(audio_data)} bytes")
        
        try:
            response = await client.post(url, files=files)
            
            print(f"   Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                asr_data = data.get("asr", {})
                asr_text = asr_data.get("text", "")
                asr_confidence = asr_data.get("confidence", 0.0)
                bot_text = data.get("text", "")
                
                print(f"   ✅ SUCCESS!")
                print(f"   ASR Text: '{asr_text}'")
                print(f"   ASR Confidence: {asr_confidence:.2f}")
                print(f"   Bot Response: '{bot_text[:100]}...'")
            else:
                print(f"   ❌ FAILED")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")

if __name__ == "__main__":
    asyncio.run(debug_turn_fast()) 