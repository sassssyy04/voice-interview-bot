#!/usr/bin/env python3
"""Test complete conversation flow with confirmation step."""

import asyncio
import httpx
from pathlib import Path

async def test_complete_conversation_with_confirmation():
    """Test complete conversation flow ending with confirmation."""
    base_url = "http://localhost:8000"
    
    print("🎭 Testing Complete Conversation with Confirmation...")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Step 1: Start conversation
        print("\n1. Starting conversation...")
        response = await client.post(f"{base_url}/api/v1/conversation/start")
        
        if response.status_code != 200:
            print(f"   ❌ Failed to start: {response.status_code}")
            return False
            
        data = response.json()
        candidate_id = data.get("candidate_id")
        print(f"   ✅ Started: {candidate_id}")
        
        # Define the conversation flow with audio files
        conversation_steps = [
            ("pin_001_clean.wav", "pincode"),
            ("sal_001_clean.wav", "salary"), 
            ("vehicle_001_clean.wav", "vehicle"),
            ("lang_001_clean.wav", "languages"),
            ("avail_001_clean.wav", "availability"),
            ("shift_001_clean.wav", "shift"),
            ("exp_001_clean.wav", "experience"),
        ]
        
        # Step 2: Complete all conversation fields
        for i, (audio_file, field_name) in enumerate(conversation_steps, 1):
            print(f"\n{i+1}. Testing {field_name}...")
            audio_path = f"test_harness/generated_audio/{audio_file}"
            
            if not Path(audio_path).exists():
                print(f"   ❌ Audio file not found: {audio_path}")
                continue
            
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            files = {'audio_file': ('recording.wav', audio_data, 'audio/wav')}
            url = f"{base_url}/api/v1/conversation/{candidate_id}/turn-fast"
            
            response = await client.post(url, files=files)
            
            if response.status_code == 200:
                data = response.json()
                asr_text = data.get("asr", {}).get("text", "")
                bot_text = data.get("text", "")
                is_complete = data.get("conversation_completed", False)
                
                print(f"   ✅ {field_name}: '{asr_text[:40]}...' (conf: {data.get('asr', {}).get('confidence', 0):.2f})")
                print(f"   Bot: '{bot_text[:60]}...'")
                print(f"   Complete: {is_complete}")
                
                # Check if we've reached confirmation step
                if "dhanyawad" in bot_text.lower() or "confirm" in bot_text.lower() or "repeat" in bot_text.lower():
                    print(f"   🎯 REACHED CONFIRMATION STEP!")
                    
                    # Test confirmation with "Yes" response
                    await test_confirmation_response(client, candidate_id, base_url)
                    return True
                    
                if is_complete:
                    print(f"   🎉 Conversation completed at step {i+1}!")
                    return True
                    
            else:
                print(f"   ❌ Failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
        
        print(f"\n⚠️  Completed all steps but didn't reach confirmation")
        return False

async def test_confirmation_response(client, candidate_id, base_url):
    """Test the confirmation response."""
    print(f"\n🎯 Testing Confirmation Response...")
    
    # Test with "Yes" response
    print(f"\n--- Testing 'Yes' Response ---")
    audio_path = "test_harness/generated_audio/conf_yes_001_clean.wav"
    
    if Path(audio_path).exists():
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        files = {'audio_file': ('confirmation.wav', audio_data, 'audio/wav')}
        url = f"{base_url}/api/v1/conversation/{candidate_id}/turn-fast"
        
        response = await client.post(url, files=files)
        
        if response.status_code == 200:
            data = response.json()
            asr_text = data.get("asr", {}).get("text", "")
            bot_text = data.get("text", "")
            is_complete = data.get("conversation_completed", False)
            
            print(f"   ASR: '{asr_text}' (conf: {data.get('asr', {}).get('confidence', 0):.2f})")
            print(f"   Bot: '{bot_text[:100]}...'")
            print(f"   Conversation Complete: {is_complete}")
            
            if is_complete:
                print(f"   ✅ SUCCESS! Conversation completed with confirmation!")
                
                # Check if job matches are mentioned
                if "job" in bot_text.lower() or "match" in bot_text.lower():
                    print(f"   🎉 Bot is providing job matches!")
                
                return True
            else:
                print(f"   ⚠️  Confirmation processed but conversation not complete")
                return False
                
        else:
            print(f"   ❌ Confirmation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    else:
        print(f"   ❌ Confirmation audio file not found: {audio_path}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_complete_conversation_with_confirmation())
    if success:
        print(f"\n🎉 Complete conversation flow with confirmation SUCCESSFUL!")
    else:
        print(f"\n❌ Complete conversation flow with confirmation FAILED!") 