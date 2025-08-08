#!/usr/bin/env python3
"""
Setup script for Google Cloud Speech API authentication.
"""

import os
import json

def setup_google_credentials():
    """Guide user through Google Cloud setup."""
    
    print("🔧 Setting up Google Cloud Speech API")
    print("=" * 50)
    
    print("\n📋 You have two options:")
    print("1. Use Service Account JSON key file (Recommended)")
    print("2. Use Application Default Credentials")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        setup_service_account()
    elif choice == "2":
        setup_adc()
    else:
        print("Invalid choice. Please run again.")

def setup_service_account():
    """Setup using service account JSON file."""
    
    print("\n🔑 Service Account Setup:")
    print("1. Go to: https://console.cloud.google.com")
    print("2. Navigate to: IAM & Admin → Service Accounts")
    print("3. Create a new service account or use existing")
    print("4. Download JSON key file")
    print("5. Place the JSON file in this project directory")
    
    json_file = input("\nEnter the path to your JSON key file: ").strip()
    
    if os.path.exists(json_file):
        # Create .env content
        env_content = f"""# Google Cloud Speech API
GOOGLE_APPLICATION_CREDENTIALS={json_file}

# Other API Keys (add as needed)
# OPENAI_API_KEY=your_openai_key_here
# AZURE_SPEECH_KEY=your_azure_key_here
# AZURE_SPEECH_REGION=your_azure_region_here

# Application Settings
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
ENVIRONMENT=development

# Voice Settings
DEFAULT_VOICE_LANGUAGE=hi-IN
SPEECH_RATE=1.0
SPEECH_PITCH=0.0
MAX_RESPONSE_TIME=2.0
CONFIDENCE_THRESHOLD=0.7

# Job Matching Settings
MAX_DISTANCE_KM=50
SALARY_TOLERANCE_PERCENT=20
"""
        
        # Write .env file
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print(f"\n✅ Created .env file with Google credentials!")
        print(f"📁 JSON file: {json_file}")
        
        # Test the credentials
        test_google_credentials(json_file)
        
    else:
        print(f"\n❌ File not found: {json_file}")
        print("Please check the path and try again.")

def setup_adc():
    """Setup using Application Default Credentials."""
    
    print("\n🔑 Application Default Credentials Setup:")
    print("Run this command to authenticate:")
    print("   gcloud auth application-default login")
    print("\nThen set environment variable:")
    print("   export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/application_default_credentials.json")

def test_google_credentials(json_file):
    """Test if Google credentials work."""
    
    print("\n🧪 Testing Google Cloud Speech API...")
    
    try:
        # Set environment variable
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = json_file
        
        # Try to import and create client
        from google.cloud import speech
        client = speech.SpeechClient()
        
        print("✅ Google Cloud Speech client created successfully!")
        print("🎉 You're ready to use Google Speech-to-Text!")
        
        return True
        
    except ImportError:
        print("❌ Google Cloud Speech library not installed.")
        print("Install with: pip install google-cloud-speech")
        return False
        
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        print("Please check your JSON key file and try again.")
        return False

def create_sample_env():
    """Create a sample .env file for manual editing."""
    
    env_content = """# Copy your Google Cloud service account JSON file to the project directory
# Then update the path below
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json

# Optional: Other API keys for full functionality
# OPENAI_API_KEY=your_openai_key_here
# AZURE_SPEECH_KEY=your_azure_key_here
# AZURE_SPEECH_REGION=your_azure_region_here

# Application Settings
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
ENVIRONMENT=development

# Voice Settings
DEFAULT_VOICE_LANGUAGE=hi-IN
SPEECH_RATE=1.0
SPEECH_PITCH=0.0
MAX_RESPONSE_TIME=2.0
CONFIDENCE_THRESHOLD=0.7

# Job Matching Settings
MAX_DISTANCE_KM=50
SALARY_TOLERANCE_PERCENT=20
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("📝 Created .env template file")
    print("Edit the GOOGLE_APPLICATION_CREDENTIALS path with your JSON file location")

if __name__ == "__main__":
    try:
        setup_google_credentials()
    except KeyboardInterrupt:
        print("\n👋 Setup cancelled by user")
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        print("\n📝 Creating template .env file instead...")
        create_sample_env() 