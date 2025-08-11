import os
import numpy as np
import librosa
import soundfile as sf

def rms(x):
    return np.sqrt(np.mean(x**2, dtype=np.float64))

def add_white_noise_at_snr(clean, snr_db):
    if clean.size == 0:
        return clean
    noise = np.random.default_rng().standard_normal(clean.shape).astype(np.float32)
    clean_rms = rms(clean)
    if clean_rms < 1e-9:
        return clean.copy()
    target_noise_rms = clean_rms / (10.0**(snr_db / 20.0))
    noise_rms = rms(noise)
    if noise_rms > 0:
        noise *= (target_noise_rms / noise_rms)
    return clean + noise

def process_folder(base_dir="test_harness/generated_audio", endings=("001_clean.wav", "003_clean.wav"), snr_db=15):
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if not fn.endswith(endings):
                continue
            in_path = os.path.join(root, fn)
            try:
                # Load MP3 (even if extension is .wav) as float32, sr unchanged
                clean, sr = librosa.load(in_path, sr=None, mono=False)
                if clean.ndim == 1:
                    clean = clean[np.newaxis, :]  # shape (1, n_samples)

                noisy = add_white_noise_at_snr(clean, snr_db=snr_db)

                out_path = os.path.join(root, fn.replace("_clean.wav", "_noisy_gpt.wav"))
                # Write as proper WAV
                sf.write(out_path, noisy.T, sr, subtype='PCM_16')
                print(f"Created noisy file: {out_path}")
            except Exception as e:
                print(f"Failed on {in_path}: {e}")

if __name__ == "__main__":
    process_folder(snr_db=15)
