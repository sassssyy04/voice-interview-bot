#!/usr/bin/env python3
"""Test multiple turns in a single conversation to verify entity extraction."""

import asyncio
import httpx
from pathlib import Path

async def test_multi_turn_conversation():
    """Test multiple conversation turns to verify entity extraction and flow."""
    base_url = "http://localhost:8000"
    
    print("🎭 Testing multi-turn conversation with enhanced NLU...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Start conversation
        print("1. Starting conversation...")
        response = await client.post(f"{base_url}/api/v1/conversation/start")
        
        if response.status_code != 200:
            print(f"   ❌ Failed to start: {response.status_code}")
            return
            
        data = response.json()
        candidate_id = data.get("candidate_id")
        print(f"   ✅ Started: {candidate_id}")
        
        # Test turns in sequence
        test_turns = [
            {
                "name": "Pincode",
                "audio": "pin_001_clean.wav",
                "expected_field": "pincode/location",
                "expected_value": "Delhi location"
            },
            {
                "name": "Salary", 
                "audio": "sal_001_clean.wav",
                "expected_field": "expected_salary",
                "expected_value": "15000"
            },
            {
                "name": "Vehicle",
                "audio": "vehicle_001_clean.wav", 
                "expected_field": "has_two_wheeler",
                "expected_value": "True"
            }
        ]
        
        for i, turn in enumerate(test_turns, 1):
            print(f"\n--- Turn {i}: {turn['name']} ---")
            
            audio_path = f"test_harness/generated_audio/{turn['audio']}"
            if not Path(audio_path).exists():
                print(f"   ❌ Audio file not found: {audio_path}")
                continue
            
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            files = {
                'audio_file': ('recording.wav', audio_data, 'audio/wav')
            }
            
            url = f"{base_url}/api/v1/conversation/{candidate_id}/turn-fast"
            response = await client.post(url, files=files)
            
            if response.status_code == 200:
                data = response.json()
                asr_data = data.get("asr", {})
                asr_text = asr_data.get("text", "")
                asr_confidence = asr_data.get("confidence", 0.0)
                bot_text = data.get("text", "")
                conversation_complete = data.get("conversation_complete", False)
                
                print(f"   ✅ Turn successful!")
                print(f"   ASR: '{asr_text[:60]}...' (conf: {asr_confidence:.2f})")
                print(f"   Bot: '{bot_text[:80]}...'")
                print(f"   Complete: {conversation_complete}")
                
                # Check if bot has moved to next question
                if "pincode" in bot_text.lower() or "area" in bot_text.lower():
                    print("   🔄 Bot still asking for pincode")
                elif "salary" in bot_text.lower() or "kitni" in bot_text.lower():
                    print("   🎯 Bot asking for salary - Good progress!")
                elif "bike" in bot_text.lower() or "scooter" in bot_text.lower():
                    print("   🎯 Bot asking for vehicle - Excellent progress!")
                elif "language" in bot_text.lower() or "bol" in bot_text.lower():
                    print("   🎯 Bot asking for languages - Great progress!")
                elif "kab" in bot_text.lower() or "start" in bot_text.lower():
                    print("   🎯 Bot asking for availability - Amazing progress!")
                elif "shift" in bot_text.lower() or "time" in bot_text.lower():
                    print("   🎯 Bot asking for shift preference - Fantastic!")
                else:
                    print(f"   ❓ Bot response pattern not recognized")
                
                if conversation_complete:
                    print("   🎉 CONVERSATION COMPLETED!")
                    if data.get("matches"):
                        print(f"   🎯 Found {len(data['matches'])} job matches!")
                    break
                    
            else:
                print(f"   ❌ Failed: {response.status_code}")
                print(f"   Error: {response.text}")
                break
            
            # Brief pause between turns
            await asyncio.sleep(2)
    
    print(f"\n🔍 Multi-turn conversation test completed")

if __name__ == "__main__":
    asyncio.run(test_multi_turn_conversation()) 