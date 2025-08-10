#!/usr/bin/env python3
"""Simple test to debug conversation flow."""

import asyncio
import httpx
from pathlib import Path

async def test_conversation_flow():
    """Test the basic conversation flow."""
    base_url = "http://localhost:8000"
    
    print("🎭 Testing conversation flow...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Start conversation
        print("1. Starting conversation...")
        response = await client.post(f"{base_url}/api/v1/conversation/start")
        
        if response.status_code == 200:
            data = response.json()
            candidate_id = data.get("candidate_id")
            print(f"   ✅ Conversation started: {candidate_id}")
        else:
            print(f"   ❌ Failed to start conversation: {response.status_code}")
            return False
        
        # Step 2: Send audio
        print("2. Sending audio...")
        audio_path = "test_harness/generated_audio/pin_001_clean.wav"
        
        if not Path(audio_path).exists():
            print(f"   ❌ Audio file not found: {audio_path}")
            return False
        
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        files = {
            'audio_file': ('recording.wav', audio_data, 'audio/wav')
        }
        
        url = f"{base_url}/api/v1/conversation/{candidate_id}/turn-fast"
        print(f"   Sending to: {url}")
        print(f"   Audio size: {len(audio_data)} bytes")
        
        response = await client.post(url, files=files)
        
        if response.status_code == 200:
            data = response.json()
            asr_data = data.get("asr", {})
            asr_text = asr_data.get("text", "")
            asr_confidence = asr_data.get("confidence", 0.0)
            bot_text = data.get("text", "")
            
            print(f"   ✅ Success!")
            print(f"   ASR Text: '{asr_text}'")
            print(f"   ASR Confidence: {asr_confidence:.2f}")
            print(f"   Bot Response: '{bot_text}'")
            return True
        else:
            print(f"   ❌ Failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False

if __name__ == "__main__":
    asyncio.run(test_conversation_flow()) 