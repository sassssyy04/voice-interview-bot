import os
from typing import Optional
from dotenv import load_dotenv

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseSettings  # type: ignore
    SettingsConfigDict = dict  # type: ignore

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # pydantic-settings v2 config (ignore unknown keys like sarvam_api)
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API Keys
    openai_api_key: Optional[str] = None
    google_credentials_path: Optional[str] = None
    google_application_credentials: Optional[str] = None
    # Azure removed
    azure_speech_key: Optional[str] = None
    azure_speech_region: Optional[str] = None

    # ElevenLabs
    elevenlabs_api_key: Optional[str] = None
    elevenlabs_voice_id: Optional[str] = "zT03pEAEi0VHKciJODfn"
    elevenlabs_model_id: Optional[str] = "eleven_multilingual_v2"
    elevenlabs_asr_model_id: Optional[str] = "scribe_v1"
    elevenlabs_asr_language_code: Optional[str] = None

    # Sarvam ASR
    sarvam_api_key: Optional[str] = None  # reads from SARVAM_API_KEY
    sarvam_asr_model: Optional[str] = "saarika:v2.5"
    sarvam_language_code: Optional[str] = "hi-IN"
    sarvam_high_vad_sensitivity: Optional[bool] = False
    sarvam_vad_signals: Optional[bool] = True

    # Application
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    environment: str = "development"

    # Voice Settings
    default_voice_language: str = "hi-IN"
    speech_rate: float = 1.0
    speech_pitch: float = 0.0
    max_response_time: float = 2.0
    confidence_threshold: float = 0.7

    # Job Matching
    max_distance_km: int = 70
    salary_tolerance_percent: int = 20

    # Optional: ffmpeg/probe overrides for Windows
    ffmpeg_path: Optional[str] = None
    ffprobe_path: Optional[str] = None


settings = Settings() 