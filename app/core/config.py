import os
from typing import Optional
from dotenv import load_dotenv

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    google_credentials_path: Optional[str] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    google_application_credentials: Optional[str] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    azure_speech_key: Optional[str] = os.getenv("AZURE_SPEECH_KEY")
    azure_speech_region: Optional[str] = os.getenv("AZURE_SPEECH_REGION")
    
    # Application
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    environment: str = os.getenv("ENVIRONMENT", "development")
    
    # Voice Settings
    default_voice_language: str = os.getenv("DEFAULT_VOICE_LANGUAGE", "hi-IN")
    speech_rate: float = float(os.getenv("SPEECH_RATE", "1.0"))
    speech_pitch: float = float(os.getenv("SPEECH_PITCH", "0.0"))
    max_response_time: float = float(os.getenv("MAX_RESPONSE_TIME", "2.0"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    
    # Job Matching
    max_distance_km: int = int(os.getenv("MAX_DISTANCE_KM", "50"))
    salary_tolerance_percent: int = int(os.getenv("SALARY_TOLERANCE_PERCENT", "20"))
    
    class Config:
        env_file = ".env"


settings = Settings() 