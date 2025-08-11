#!/usr/bin/env python3
"""Test the confirmation flow with yes/no/correction responses."""

import asyncio
import httpx
from pathlib import Path

async def test_confirmation_flow():
    """Test the confirmation step with different response types."""
    base_url = "http://localhost:8000"
    
    print("🎭 Testing Confirmation Flow...")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Step 1: Start conversation
        print("\n1. Starting conversation...")
        response = await client.post(f"{base_url}/api/v1/conversation/start")
        
        if response.status_code != 200:
            print(f"   ❌ Failed to start: {response.status_code}")
            return
            
        data = response.json()
        candidate_id = data.get("candidate_id")
        print(f"   ✅ Started: {candidate_id}")
        
        # Step 2: Complete all fields quickly to reach confirmation
        test_utterances = [
            ("pin_001_clean.wav", "pincode"),
            ("sal_001_clean.wav", "salary"), 
            ("vehicle_001_clean.wav", "vehicle"),
            ("lang_001_clean.wav", "languages"),
            ("avail_001_clean.wav", "availability"),
            ("shift_001_clean.wav", "shift"),
            ("exp_001_clean.wav", "experience"),
        ]
        
        for i, (audio_file, field) in enumerate(test_utterances, 1):
            print(f"\n{i+1}. Sending {field} data...")
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
                
                print(f"   ✅ {field}: '{asr_text[:40]}...'")
                print(f"   Bot: '{bot_text[:60]}...'")
                
                if is_complete:
                    print(f"   🎉 Conversation completed at step {i+1}!")
                    break
                
                # Check if we've reached confirmation step
                if "dhanyawad" in bot_text.lower() or "confirm" in bot_text.lower():
                    print(f"   🎯 Reached confirmation step!")
                    
                    # Test confirmation responses
                    await test_confirmation_responses(client, candidate_id, base_url)
                    break
                    
            else:
                print(f"   ❌ Failed: {response.status_code}")
                print(f"   Error: {response.text}")
                break

async def test_confirmation_responses(client, candidate_id, base_url):
    """Test different confirmation response types."""
    print(f"\n🎯 Testing Confirmation Responses...")
    
    # Test 1: "Yes" response
    print(f"\n--- Test 1: Acceptance Response ---")
    yes_text = "हां भाई, सब सही है।"  # "Yes brother, everything is correct"
    
    # Convert text to audio (simulate TTS output as audio input)
    audio_data = yes_text.encode('utf-8')  # Simple simulation
    files = {'audio_file': ('yes_response.wav', audio_data, 'audio/wav')}
    url = f"{base_url}/api/v1/conversation/{candidate_id}/turn-fast"
    
    response = await client.post(url, files=files)
    
    if response.status_code == 200:
        data = response.json()
        bot_response = data.get("text", "")
        is_complete = data.get("conversation_completed", False)
        
        print(f"   Bot Response: '{bot_response[:80]}...'")
        print(f"   Conversation Complete: {is_complete}")
        
        if is_complete:
            print(f"   ✅ SUCCESS: Conversation completed on acceptance!")
        else:
            print(f"   ⚠️  Conversation not completed yet")
    else:
        print(f"   ❌ Failed: {response.status_code}")
        print(f"   Error: {response.text}")

async def test_correction_response():
    """Test a correction response separately."""
    print(f"\n--- Test 2: Correction Response (New Session) ---")
    
    # This would need a separate session to test correction flow
    # For now, just demonstrate the concept
    correction_examples = [
        "नहीं भाई, salary गलत है, मुझे बीस हजार चाहिए।",  # "No brother, salary is wrong, I need twenty thousand"
        "area change करना है, मैं pune में हूं।",  # "Need to change area, I'm in Pune"
        "bike नहीं है मेरे पास।",  # "I don't have a bike"
    ]
    
    for example in correction_examples:
        print(f"   Example correction: '{example}'")

if __name__ == "__main__":
    asyncio.run(test_confirmation_flow()) 