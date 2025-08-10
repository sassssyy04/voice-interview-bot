#!/usr/bin/env python3
"""Test script for adding noise to clean audio files."""

from pydub import AudioSegment
import numpy as np
import os

def add_background_noise(input_wav, output_wav, noise_level_db=-20):
    """Add white noise to audio file using numpy approach."""
    try:
        # Load the original audio
        print(f"Loading audio from: {input_wav}")
        audio = AudioSegment.from_wav(input_wav)
        print(f"Loaded audio: {len(audio)}ms, {audio.frame_rate}Hz, {audio.channels} channels")

        # Generate white noise with the same duration
        print(f"Generating noise...")
        noise = AudioSegment(
            (np.random.randn(len(audio.get_array_of_samples())) * 32767).astype(np.int16).tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels
        )

        # Adjust noise level
        print(f"Adjusting noise level to {noise_level_db}dB")
        noise = noise - noise.dBFS + noise_level_db

        # Overlay the noise on the original audio
        print(f"Mixing audio with noise...")
        mixed = audio.overlay(noise)

        # Save to output file
        print(f"Saving to: {output_wav}")
        mixed.export(output_wav, format="wav")
        print(f"Successfully saved noisy file to: {output_wav}")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    # Test with one of the generated clean files
    clean_file = "test_harness/generated_audio/pin_diverse_001_clean.wav"
    noisy_file = "test_harness/generated_audio/pin_diverse_001_noisy.wav"
    
    if os.path.exists(clean_file):
        print(f"Testing noise addition...")
        success = add_background_noise(clean_file, noisy_file, noise_level_db=-15)
        if success:
            print("✓ Noise addition successful!")
        else:
            print("✗ Noise addition failed!")
    else:
        print(f"Clean file not found: {clean_file}") 