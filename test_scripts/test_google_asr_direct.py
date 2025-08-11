#!/usr/bin/env python3
"""Direct test of Google Cloud Speech-to-Text ASR to debug transcription issues."""

import os
import io
import wave
import audioop
import asyncio
from pathlib import Path
from dotenv import load_dotenv

try:
    from google.cloud import speech
    from google.api_core import exceptions as gexceptions
except Exception:
    speech = None
    gexceptions = None

load_dotenv()

def looks_like_mp3(data: bytes) -> bool:
    if not data or len(data) < 4:
        return False
    if data[:3] == b"ID3":
        return True
    b1, b2 = data[0], data[1]
    return (b1 == 0xFF) and ((b2 & 0xE0) == 0xE0)

def reencode_to_linear16_pure(data: bytes) -> bytes:
    """Pure-Python WAV path: mono, 16kHz, keep sample width, normalize RMS."""
    try:
        buf = io.BytesIO(data)
        with wave.open(buf, 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            pcm = wf.readframes(n_frames)
        if sampwidth not in (1, 2, 3, 4):
            return data
        if n_channels == 2:
            pcm = audioop.tomono(pcm, sampwidth, 0.5, 0.5)
        if framerate != 16000:
            pcm, _ = audioop.ratecv(pcm, sampwidth, 1, framerate, 16000, None)
            framerate = 16000
        try:
            rms = audioop.rms(pcm, sampwidth)
            if rms > 0:
                target_rms = 2000
                factor = min(8.0, max(0.5, target_rms / float(rms)))
                pcm = audioop.mul(pcm, sampwidth, factor)
        except Exception:
            pass
        out = io.BytesIO()
        with wave.open(out, 'wb') as ww:
            ww.setnchannels(1)
            ww.setsampwidth(sampwidth)
            ww.setframerate(framerate)
            ww.writeframes(pcm)
        out.seek(0)
        return out.read()
    except Exception:
        return data

async def transcribe_with_google(audio_bytes: bytes) -> tuple[str, float]:
    if speech is None:
        raise RuntimeError("google-cloud-speech not installed")

    client = speech.SpeechClient()

    def recognize_with_cfg(payload: bytes, cfg: speech.RecognitionConfig) -> tuple[str, float]:
        audio = speech.RecognitionAudio(content=payload)
        resp = client.recognize(config=cfg, audio=audio)
        if resp.results and resp.results[0].alternatives:
            alt = resp.results[0].alternatives[0]
            return alt.transcript or "", float(alt.confidence or 0.0)
        return "", 0.0

    # Try MP3 if likely, otherwise unspecified
    try_order = []
    if looks_like_mp3(audio_bytes):
        try_order.append(speech.RecognitionConfig.AudioEncoding.MP3)
    try_order.append(speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED)

    last_err = None
    for enc in try_order:
        cfg = speech.RecognitionConfig(
            encoding=enc,
            sample_rate_hertz=16000,
            language_code="hi-IN",
            alternative_language_codes=["en-IN"],
            enable_automatic_punctuation=True,
            model="latest_long",
            max_alternatives=1,
        )
        try:
            text, conf = recognize_with_cfg(audio_bytes, cfg)
            if text:
                return text, conf
        except Exception as e:
            last_err = e
            continue

    # Pure-Python re-encode to LINEAR16 16k mono and try again
    lin16 = reencode_to_linear16_pure(audio_bytes)
    if lin16:
        cfg_lin = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="hi-IN",
            alternative_language_codes=["en-IN"],
            enable_automatic_punctuation=True,
            model="latest_long",
            max_alternatives=1,
        )
        try:
            return recognize_with_cfg(lin16, cfg_lin)
        except Exception as e:
            last_err = e

    if last_err:
        raise last_err
    return "", 0.0

async def test_google_asr_direct():
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GOOGLE_CREDENTIALS_PATH")
    print("Google creds:", "FOUND" if (creds and creds.strip() and Path(creds).exists()) else "MISSING")
    if not speech:
        print("google-cloud-speech not installed")
        return

    test_files = [
        "test_harness/generated_audio/pin_diverse_003_noisy_gpt.wav",
        "test_harness/generated_audio/pin_diverse_003_clean.wav",
    ]

    for audio_file in test_files:
        p = Path(audio_file)
        if not p.exists():
            print(f"❌ File not found: {p}")
            continue
        print(f"\nTesting: {p.name}")
        data = p.read_bytes()
        print(f"   Size: {len(data)} bytes (looks_like_mp3={looks_like_mp3(data)})")
        try:
            text, conf = await transcribe_with_google(data)
            print(f"   Text: '{text}'")
            print(f"   Confidence: {conf:.2f}")
            if text:
                print("   ✅ Google ASR is working")
            else:
                print("   ⚠️ Empty transcript")
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            if gexceptions and isinstance(e, gexceptions.InvalidArgument):
                print("   Hint: This usually means encoding mismatch. We tried MP3/unspecified/LINEAR16.")

if __name__ == "__main__":
    asyncio.run(test_google_asr_direct()) 