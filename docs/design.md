# README

## Overview
This project implements a **multi-slot, speech-driven job matching pipeline** that interacts with users via voice prompts, extracts structured entities from their responses, and matches them to relevant jobs based on weighted criteria.

Demo: https://drive.google.com/file/d/1g0ux-SUoIaGy9557SBGvYuTzfvRDUnwK/view?usp=sharing
Final metrics summary for generated personas using audio files can be found at **results_for_personas.json**
Generated audio files can be found at **test_harness\generated_audio**

The system:
1. Presents spoken prompts to the user.
2. Accepts audio responses.
3. Uses automatic speech recognition (ASR) and entity extraction to fill predefined slots.
4. Confirms extracted information with the user.
5. Matches the completed profile against a job database using a scoring formula.

---
## how test files were generated
- set seed example for jobs.json and generated using aws partyrock
- for audio files, created utterances.yaml
- Created **test audio** to validate the pipeline:
    **4 personas**, each with:
    **3 golden jobs** defined in `jobs.json` (ground truth matches).
- Audio generated using **ElevenLabs**:
  - 4 different **accents** and **voice IDs**.
- **Noise simulation** for 2 personas:
  - Added white noise at controlled SNR using:


## Architecture

### 1. Slot Filling Flow
Each slot has:
- **Prompt**: Initial question to the user.
- **Confirmation Prompt**: Used when confidence is low.

Processing steps for each slot:

1. **Prompt to Speech**  
   - Convert the first prompt to audio and play to the user.

2. **User Response**  
   - User replies via voice.  
   - Response is recorded.

3. **ASR Processing**  
   - Primary ASR: **Google Speech-to-Text**  
   - Fallback: **ElevenLabs** ASR.

4. **Entity Extraction Logic**  
   - If **confidence ≥ 0.7**:
     - **If slot ≠ "language"** → Apply rule-based entity extraction.  
     - **If slot = "language"** → Send directly to LLM.
   - If **confidence < 0.7**:
     - Play confirmation prompt, re-ASR.
     - If still < 0.7 → Pass transcript to LLM for entity extraction.

5. **Proceed to Next Slot**  
   - Continue until all slots are filled.

6. **Final Confirmation**  
   - Construct a **summary prompt** with all slot values.  
   - Present to user for final verification.
   - If any changes are requested:
     - Update affected slot using LLM extraction.

---

### 2. Job Matching
Once the slots are confirmed, we score each candidate job using:

overall_score = (0.30 * location) +
(0.15 * salary) +
(0.25 * shift) +
(0.15 * language) +
(0.10 * vehicle) +
(0.05 * experience)

 
Details for calculating each component are documented in **job_match.md**.

---



```python
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
