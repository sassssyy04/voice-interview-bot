import asyncio
from pathlib import Path

from app.services.speech_recognition import ASRService


async def main() -> None:
    """Transcribes the local test WAV file using Sarvam streaming if configured."""
    audio_path = Path("test_output.wav")
    if not audio_path.exists():
        raise FileNotFoundError("test_output.wav not found in project root")

    audio_bytes = audio_path.read_bytes()
    asr = ASRService()
    result = await asr.transcribe_audio(audio_bytes)
    print(result)


if __name__ == "__main__":
    asyncio.run(main()) 