#!/usr/bin/env python3
"""Debug script to check conversation state."""

import asyncio
import httpx

async def debug_conversation_state():
    """Check conversation state after starting one."""
    base_url = "http://localhost:8000"
    
    print("🔍 Debugging conversation state...")
    
    async with httpx.AsyncClient() as client:
        # Start a conversation
        print("1. Starting conversation...")
        response = await client.post(f"{base_url}/api/v1/conversation/start")
        
        if response.status_code == 200:
            data = response.json()
            candidate_id = data.get("candidate_id")
            print(f"   ✅ Started: {candidate_id}")
            
            # Wait a moment
            await asyncio.sleep(1)
            
            # Try to get conversation status (this would check if conversation exists)
            print("2. Checking conversation status...")
            try:
                status_response = await client.get(f"{base_url}/api/v1/conversation/{candidate_id}/status")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print("   ✅ Conversation exists!")
                    print(f"   Current field: {status_data.get('current_field')}")
                    print(f"   Completion rate: {status_data.get('completion_rate', 0):.1%}")
                else:
                    print(f"   ❌ Status call failed: {status_response.status_code}")
                    print(f"   Error: {status_response.text}")
            except Exception as e:
                print(f"   ❌ Status call error: {e}")
            
            return candidate_id
        else:
            print(f"   ❌ Failed to start: {response.status_code}")
            return None

if __name__ == "__main__":
    asyncio.run(debug_conversation_state()) 