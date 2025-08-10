#!/usr/bin/env python3
"""Test script to verify confidence calculation is working."""

import asyncio
import httpx
from pathlib import Path

async def test_confidence_calculation():
    """Test confidence scores from the bot API."""
    base_url = "http://localhost:8000"
    
    print("🎯 Testing Confidence Score Calculation...")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Start conversation
        print("\n1. Starting conversation...")
        response = await client.post(f"{base_url}/api/v1/conversation/start")
        
        if response.status_code != 200:
            print(f"   ❌ Failed to start: {response.status_code}")
            return
            
        data = response.json()
        candidate_id = data.get("candidate_id")
        print(f"   ✅ Started: {candidate_id}")
        
        # Test different audio files to see confidence variation
        test_files = [
            ("pin_001_clean.wav", "Pincode - long, clear"),
            ("sal_001_clean.wav", "Salary - medium length"),
            ("conf_yes_001_clean.wav", "Confirmation - short"),
            ("conf_yes_002_clean.wav", "Confirmation - very short"),
        ]
        
        print(f"\n2. Testing confidence scores for different audio files...")
        
        for i, (audio_file, description) in enumerate(test_files, 1):
            audio_path = f"test_harness/generated_audio/{audio_file}"
            
            if not Path(audio_path).exists():
                print(f"   ⚠️  Audio file not found: {audio_file}")
                continue
            
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            files = {'audio_file': ('recording.wav', audio_data, 'audio/wav')}
            url = f"{base_url}/api/v1/conversation/{candidate_id}/turn-fast"
            
            print(f"\n   Test {i}: {description}")
            print(f"   File: {audio_file} ({len(audio_data)} bytes)")
            
            response = await client.post(url, files=files)
            
            if response.status_code == 200:
                data = response.json()
                asr_data = data.get("asr", {})
                text = asr_data.get("text", "")
                confidence = asr_data.get("confidence", 0.0)
                
                print(f"   📝 Text: '{text[:50]}...'")
                print(f"   🎯 Confidence: {confidence:.3f}")
                print(f"   📊 Confidence %: {confidence*100:.1f}%")
                
                # Analyze confidence score
                if confidence >= 0.9:
                    print(f"   ✅ HIGH confidence (very clear audio)")
                elif confidence >= 0.8:
                    print(f"   ✅ GOOD confidence (clear audio)")
                elif confidence >= 0.7:
                    print(f"   ⚠️  MEDIUM confidence (acceptable)")
                elif confidence >= 0.5:
                    print(f"   ⚠️  LOW confidence (unclear)")
                else:
                    print(f"   ❌ VERY LOW confidence (poor quality)")
                    
            else:
                print(f"   ❌ Failed: {response.status_code}")
                print(f"   Error: {response.text}")
                break
        
        print(f"\n📊 Confidence Analysis Summary:")
        print(f"   • Each audio file should have different confidence scores")
        print(f"   • Longer, clearer audio should have higher confidence")
        print(f"   • Short audio should have lower confidence")
        print(f"   • No confidence should be exactly 1.00 (that indicates hardcoded)")

if __name__ == "__main__":
    asyncio.run(test_confidence_calculation()) 