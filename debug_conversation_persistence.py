#!/usr/bin/env python3
"""Debug conversation state persistence issues."""

import asyncio
import httpx
from pathlib import Path
import time

async def debug_conversation_persistence():
    """Debug why conversation state isn't persisting."""
    base_url = "http://localhost:8000"
    
    print("🔍 Debugging conversation persistence...")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Step 1: Start conversation
        print("\n1. Starting conversation...")
        start_response = await client.post(f"{base_url}/api/v1/conversation/start")
        
        if start_response.status_code != 200:
            print(f"   ❌ Start failed: {start_response.status_code}")
            print(f"   Response: {start_response.text}")
            return
        
        start_data = start_response.json()
        candidate_id = start_data.get("candidate_id")
        print(f"   ✅ Started: {candidate_id}")
        print(f"   Response keys: {list(start_data.keys())}")
        
        # Step 2: Immediate status check
        print("\n2. Immediate status check...")
        status_response = await client.get(f"{base_url}/api/v1/conversation/{candidate_id}/status")
        
        if status_response.status_code == 200:
            status_data = status_response.json()
            print(f"   ✅ Status found immediately")
            print(f"   Current field: {status_data.get('current_field')}")
            print(f"   Completion rate: {status_data.get('completion_rate', 0):.1%}")
        else:
            print(f"   ❌ Status failed: {status_response.status_code}")
            print(f"   Error: {status_response.text}")
            return
        
        # Step 3: Wait and check again
        print("\n3. Waiting 2 seconds then checking status again...")
        await asyncio.sleep(2)
        
        status_response2 = await client.get(f"{base_url}/api/v1/conversation/{candidate_id}/status")
        
        if status_response2.status_code == 200:
            print(f"   ✅ Status still exists after wait")
        else:
            print(f"   ❌ Status disappeared: {status_response2.status_code}")
            print(f"   Error: {status_response2.text}")
            return
        
        # Step 4: Try turn-fast immediately
        print("\n4. Testing turn-fast immediately...")
        audio_path = "test_harness/generated_audio/pin_001_clean.wav"
        
        if not Path(audio_path).exists():
            print(f"   ❌ Audio file not found: {audio_path}")
            return
        
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        files = {
            'audio_file': ('recording.wav', audio_data, 'audio/wav')
        }
        
        turn_url = f"{base_url}/api/v1/conversation/{candidate_id}/turn-fast"
        print(f"   Sending to: {turn_url}")
        
        turn_response = await client.post(turn_url, files=files)
        
        if turn_response.status_code == 200:
            turn_data = turn_response.json()
            asr_data = turn_data.get("asr", {})
            print(f"   ✅ Turn-fast SUCCESS!")
            print(f"   ASR: '{asr_data.get('text', '')[:60]}...'")
            print(f"   Bot: '{turn_data.get('text', '')[:60]}...'")
        else:
            print(f"   ❌ Turn-fast FAILED: {turn_response.status_code}")
            print(f"   Error: {turn_response.text}")
            
            # Step 5: Check if conversation still exists after failure
            print("\n5. Checking if conversation still exists after failure...")
            status_response3 = await client.get(f"{base_url}/api/v1/conversation/{candidate_id}/status")
            
            if status_response3.status_code == 200:
                print(f"   ✅ Conversation still exists - issue is with turn processing")
            else:
                print(f"   ❌ Conversation disappeared - issue is with state management")
        
        # Step 6: Try the older turn endpoint
        print("\n6. Testing older /turn endpoint...")
        turn_old_url = f"{base_url}/api/v1/conversation/{candidate_id}/turn"
        
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        files2 = {
            'audio_file': ('recording.wav', audio_data, 'audio/wav')
        }
        
        turn_old_response = await client.post(turn_old_url, files=files2)
        
        if turn_old_response.status_code == 200:
            print(f"   ✅ Old turn endpoint works!")
        else:
            print(f"   ❌ Old turn endpoint also fails: {turn_old_response.status_code}")
            print(f"   Error: {turn_old_response.text}")

if __name__ == "__main__":
    asyncio.run(debug_conversation_persistence()) 