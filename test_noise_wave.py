#!/usr/bin/env python3
"""Test script for adding noise using wave module (no FFmpeg needed)."""

import wave
import numpy as np
import os

def add_background_noise_wave(input_wav, output_wav, noise_level=0.1):
    """Add white noise to audio file using wave module."""
    try:
        # Open the input WAV file
        print(f"Loading audio from: {input_wav}")
        with wave.open(input_wav, 'rb') as wav_file:
            # Get audio properties
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            
            print(f"Audio info: {frames} frames, {sample_rate}Hz, {channels} channels, {sample_width} bytes/sample")
            
            # Read all audio data
            audio_data = wav_file.readframes(frames)
        
        # Convert to numpy array
        if sample_width == 1:
            dtype = np.uint8
            max_val = 127
        elif sample_width == 2:
            dtype = np.int16
            max_val = 32767
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=dtype)
        print(f"Loaded {len(audio_array)} samples")
        
        # Generate white noise
        print(f"Generating noise at level {noise_level}")
        noise = np.random.randn(len(audio_array)) * max_val * noise_level
        noise = noise.astype(dtype)
        
        # Mix audio with noise
        print("Mixing audio with noise...")
        # Convert to larger dtype to prevent overflow
        mixed = audio_array.astype(np.int32) + noise.astype(np.int32)
        
        # Clip to prevent overflow and convert back
        mixed = np.clip(mixed, -max_val, max_val).astype(dtype)
        
        # Save the result
        print(f"Saving to: {output_wav}")
        with wave.open(output_wav, 'wb') as output_file:
            output_file.setnchannels(channels)
            output_file.setsampwidth(sample_width)
            output_file.setframerate(sample_rate)
            output_file.writeframes(mixed.tobytes())
        
        print(f"Successfully saved noisy file to: {output_wav}")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # List available files first
    print("Looking for available clean files...")
    import glob
    files = glob.glob("test_harness/generated_audio/*clean*.wav")
    print(f"Found {len(files)} clean files:")
    for f in files[:10]:  # Show first 10
        print(f"  {f}")
    
    if files:
        # Use the first available file
        clean_file = files[0]
        noisy_file = clean_file.replace("_clean.wav", "_noisy_test.wav")
        
        print(f"\nTesting noise addition with wave module...")
        print(f"Input: {clean_file}")
        print(f"Output: {noisy_file}")
        
        success = add_background_noise_wave(clean_file, noisy_file, noise_level=0.1)
        if success:
            print("✓ Noise addition successful!")
        else:
            print("✗ Noise addition failed!")
    else:
        print("No clean files found!") 