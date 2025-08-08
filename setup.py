#!/usr/bin/env python3
"""
Simple setup script for the Hinglish Voice Bot.
This installs the minimal dependencies needed to run the demo.
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}")
        return False

def main():
    print("🚀 Setting up Hinglish Voice Bot...")
    print("Installing essential packages for demo...\n")
    
    # Essential packages for the demo to work
    essential_packages = [
        "fastapi==0.104.1",
        "uvicorn==0.24.0", 
        "pydantic==2.5.0",
        "pydantic-settings==2.0.0",
        "python-dotenv==1.0.0",
        "loguru==0.7.2",
        "jinja2==3.1.2",
        "aiofiles==23.2.1",
        "httpx==0.25.0"
    ]
    
    # Optional packages (for full voice functionality)
    optional_packages = [
        "SpeechRecognition==3.8.1",
        "pydub==0.25.1",
        "openai==1.3.0",
        "google-cloud-speech==2.21.0", 
        "google-cloud-texttospeech==2.16.3",
        "azure-cognitiveservices-speech==1.34.0"
    ]
    
    success_count = 0
    total_count = len(essential_packages)
    
    print("📦 Installing essential packages...")
    for package in essential_packages:
        print(f"Installing {package}...")
        if install_package(package):
            success_count += 1
            print(f"✅ {package} installed successfully")
        else:
            print(f"❌ Failed to install {package}")
    
    print(f"\n📊 Installation Results: {success_count}/{total_count} essential packages installed")
    
    if success_count == total_count:
        print("✅ Essential packages installed successfully!")
        print("\n🎯 Quick Start:")
        print("1. Copy env_example.txt to .env and add your API keys")
        print("2. Run: python -m app.main")
        print("3. Open: http://localhost:8000")
        print("4. Test the job matching without voice first")
        
        print("\n🔧 Optional: Install voice packages for full functionality:")
        for package in optional_packages:
            print(f"   pip install {package}")
        
        print("\n🧪 Test the system:")
        print("   python test_demo.py")
        
    else:
        print("⚠️ Some essential packages failed to install.")
        print("You can try installing them manually:")
        for package in essential_packages:
            print(f"   pip install {package}")

if __name__ == "__main__":
    main() 