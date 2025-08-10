#!/usr/bin/env python3
"""Test connection to the voice bot before running full test suite."""

import asyncio
import json
import websockets
from loguru import logger

async def test_bot_connection(bot_url: str = "ws://localhost:8000/ws"):
    """Test basic connection to the voice bot.
    
    Args:
        bot_url (str): WebSocket URL for bot connection
        
    Returns:
        bool: True if connection successful
    """
    print(f"🔗 Testing connection to bot at: {bot_url}")
    
    try:
        # Try to connect
        print("   📡 Attempting WebSocket connection...")
        async with websockets.connect(bot_url) as websocket:
            print("   ✅ WebSocket connection successful!")
            
            # Send a simple test message
            test_message = {
                "type": "ping",
                "message": "test_connection"
            }
            
            print("   📤 Sending test message...")
            await websocket.send(json.dumps(test_message))
            
            # Try to receive a response (with timeout)
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                print(f"   📥 Received response: {response[:100]}...")
                print("   ✅ Bot is responding!")
                return True
                
            except asyncio.TimeoutError:
                print("   ⚠️  No response received (timeout), but connection works")
                return True
                
    except ConnectionRefusedError:
        print("   ❌ Connection refused - is your bot running?")
        print("   💡 Make sure to start your bot with: python app/main.py")
        return False
        
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False

async def test_different_urls():
    """Test common bot URL variations."""
    urls_to_test = [
        "ws://localhost:8000/ws",
        "ws://127.0.0.1:8000/ws", 
        "ws://localhost:8000/websocket",
        "ws://localhost:8080/ws",
    ]
    
    print("🔍 Testing common bot URL variations...\n")
    
    for url in urls_to_test:
        success = await test_bot_connection(url)
        if success:
            print(f"\n🎉 Found working bot at: {url}")
            return url
        print()
    
    print("❌ No working bot found at any common URL")
    return None

async def main():
    """Main connection test."""
    print("="*60)
    print("VOICE BOT CONNECTION TEST")
    print("="*60)
    
    # First try the default URL
    default_url = "ws://localhost:8000/ws"
    success = await test_bot_connection(default_url)
    
    if success:
        print(f"\n🎉 Bot is ready for testing at: {default_url}")
        print("\nNext steps:")
        print("1. Run: python test_harness/bot_tester.py")
        print("2. Or run specific tests: python test_harness/bot_tester.py --utterances pin_001 sal_001")
    else:
        print("\n🔍 Trying other common URLs...")
        working_url = await test_different_urls()
        
        if working_url:
            print(f"\nUpdate your bot_tester.py to use: {working_url}")
        else:
            print("\n💡 Troubleshooting tips:")
            print("1. Make sure your bot is running: python app/main.py")
            print("2. Check the WebSocket endpoint in your bot code")
            print("3. Verify the port number (8000, 8080, etc.)")
            print("4. Check if you're using a different WebSocket path")

if __name__ == "__main__":
    asyncio.run(main()) 