# /eval/synthesize.py
import pathlib, subprocess, json
BASE = pathlib.Path("data/audio")
VOICE = "hi-IN-Neutral-1"  # swap for your provider/voice key

def tts_provider_synthesize(text, out_wav):
    # Replace with your TTS call; ensure 16kHz mono PCM WAV output.
    # Example stub: call a local TTS CLI
    cmd = ["tts-cli", "--voice", VOICE, "--text", text, "--rate", "1.0", "--out", str(out_wav), "--sr", "16000"]
    subprocess.check_call(cmd)

def main():
    for clip in sorted(BASE.glob("clip*/")):
        text = (clip / "transcript.txt").read_text(encoding="utf-8")
        out = clip / "clean.wav"
        if not out.exists():
            tts_provider_synthesize(text, out)
            print("synth", out)
if __name__ == "__main__":
    main()
