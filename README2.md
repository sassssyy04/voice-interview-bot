# Hinglish Voice Bot - Complete Technical Documentation

This document provides comprehensive technical documentation for the Hinglish Voice Bot system, covering architecture, server-mic flow, testing framework, and audio generation processes.

## Table of Contents
- [System Overview](#system-overview)
- [Architecture](#architecture)  
- [Server Microphone Input Flow](#server-microphone-input-flow)
- [Testing Framework](#testing-framework)
- [Audio File Creation System](#audio-file-creation-system)
- [Key Components](#key-components)
- [API Reference](#api-reference)
- [Development Setup](#development-setup)
- [Configuration](#configuration)

## System Overview

The Hinglish Voice Bot is a sophisticated voice-enabled interview system designed for blue-collar job screening in India. It conducts natural conversations in Hinglish (Hindi-English mix), extracts candidate information, and provides job matching recommendations.

### Core Features
- **Voice-First Interface**: Push-to-talk web interface with real-time audio processing
- **Multilingual ASR**: ElevenLabs ASR with Hinglish support and Google Cloud Speech fallback
- **Intelligent NLU**: OpenAI GPT-4 powered entity extraction with rule-based fallbacks
- **Job Matching**: Sophisticated matching algorithm with location, salary, and skill filtering
- **Comprehensive Testing**: Automated test harness with multiple voice personas and noise conditions

## Architecture

### System Components
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │────│  FastAPI Server │────│   AI Services   │
│  (Browser UI)   │    │ (Conversation    │    │ (ASR/NLU/TTS)   │
└─────────────────┘    │  Orchestrator)  │    └─────────────────┘
                       └─────────────────┘
                               │
                    ┌─────────────────┐
                    │  Job Matching   │
                    │    Service      │
                    └─────────────────┘
```

### Technology Stack
- **Backend**: FastAPI (Python 3.8+)
- **Frontend**: Vanilla JavaScript with Web Audio API
- **ASR**: ElevenLabs Speech-to-Text API (primary), Google Cloud Speech (fallback)
- **NLU**: OpenAI GPT-4 with rule-based entity extraction
- **TTS**: Multiple engines - Google Cloud TTS, ElevenLabs, pyttsx3 fallback
- **Testing**: Custom test harness with synthetic audio generation

## Server Microphone Input Flow

### 1. Audio Capture (Client-Side)
```javascript
// Push-to-talk implementation in static/app.js:244-268
onPressStart() {
    if (!this.conversationActive) {
        this.startConversation();
        return;
    }
    // Barge-in: stop TTS when user speaks
    if (this.responseAudio && !this.responseAudio.paused) {
        this.responseAudio.pause();
    }
    this.startListening(); // Begin audio recording
}
```

**Key Features:**
- **Push-to-Talk**: User holds button to record, releases to send
- **Barge-in Support**: Automatically pauses bot speech when user starts talking
- **Audio Quality**: 16kHz mono WAV with echo cancellation and noise suppression
- **Minimum Duration**: 300ms minimum recording to avoid accidental triggers

### 2. Audio Transmission & Processing (Server-Side)
```python
# Fast-path processing in app/api/routes.py:150-212
@router.post("/conversation/{candidate_id}/turn-fast")
async def process_voice_turn_fast(candidate_id: str, background_tasks: BackgroundTasks, audio_file: UploadFile = File(...)):
    # 1. Read and validate audio
    audio_data = await audio_file.read()
    
    # 2. Process ASR + NLU (synchronous)
    response_text, conversation_complete, asr_text, asr_conf, raw_asr_data, turn_id = await orchestrator.process_turn_text_only(
        candidate_id, audio_data
    )
    
    # 3. Start background TTS synthesis
    background_tasks.add_task(orchestrator.synthesize_and_store_audio, candidate_id, turn_id, response_text)
    
    # 4. Return text response immediately
    return {"text": response_text, "turn_id": turn_id, ...}
```

### 3. Audio Processing Pipeline
```python
# Core processing in app/services/conversation.py:185-269
async def process_turn_text_only(self, candidate_id: str, audio_data: bytes):
    # 1. ASR: Convert speech to text
    asr_result = await self.asr_service.transcribe_audio(audio_data)
    
    # 2. NLU: Extract entities and generate response
    response_text, target_slot, is_completed = await self._generate_intelligent_response(
        transcribed_text, confidence, candidate, state, history_pairs
    )
    
    # 3. Log conversation turn
    turn_id = str(uuid.uuid4())
    self._log_turn(candidate_id, transcribed_text, confidence, response_text, start_time, status, turn_id)
    
    return response_text, is_completed, transcribed_text, confidence, raw_asr_data, turn_id
```

### 4. Speech Recognition (ASR)
**Primary: ElevenLabs ASR** (`app/services/elevenlabs_asr.py`)
```python
async def transcribe_audio(self, audio_data: bytes) -> Dict[str, Any]:
    # Configure for Hinglish
    data = {
        "model_id": "scribe_v1",
        "language_code": "hi",  # Hindi for better Hinglish support
        "timestamp_granularity": "word"
    }
    
    # Calculate confidence from multiple factors
    confidence = self._calculate_transcription_confidence(
        text, words, language_probability, len(audio_data)
    )
```

**Confidence Calculation:**
- Text length factor (longer = more reliable)
- Audio quality estimation 
- Word-level log probabilities
- Language detection confidence
- Text quality indicators

### 5. Natural Language Understanding (NLU)
**Entity Extraction** (`app/services/nlu.py`)
```python
async def extract_entities(self, text: str, field: str, context: Dict[str, Any] = None):
    # 1. Normalize Hinglish text
    normalized_text = self._normalize_text(text)
    
    # 2. Field-specific extraction
    if field == "pincode":
        result = await self._extract_pincode(normalized_text)
    elif field == "expected_salary":
        result = await self._extract_salary(normalized_text)
    # ... other fields
    
    # 3. Validate and enhance results
    result = self._validate_extraction_result(result, field, text)
    
    return result
```

**Multi-tier Extraction Strategy:**
1. **Rule-based patterns**: Regex and keyword matching for high-confidence cases
2. **LLM extraction**: GPT-4 for complex cases and ambiguous inputs
3. **Validation**: Field-specific validation and confidence adjustment

### 6. Response Generation & Conversation Flow
```python
async def _generate_intelligent_response(self, user_text: str, confidence: float, candidate: Candidate, state: ConversationState, history: List):
    # 1. Determine missing information
    required_slots = ["pincode", "availability_date", "preferred_shift", "expected_salary", 
                     "languages", "has_two_wheeler", "total_experience_months", "confirmation"]
    missing_slots = [s for s in required_slots if s not in state.fields_completed]
    
    # 2. Handle confirmation step
    if "confirmation" in missing_slots and len(missing_slots) == 1:
        return await self._handle_confirmation_step(candidate, state, user_text, confidence)
    
    # 3. Extract entities if user provided input
    if user_text.strip() and confidence >= settings.confidence_threshold:
        entities = await self.nlu_service.extract_entities(user_text, next_slot)
        if self._try_update_slot(candidate, state, next_slot, entities, user_text):
            extracted_something = True
    
    # 4. Generate appropriate response
    next_slot = self._choose_next_slot(missing_slots, user_text, extracted_something)
    return response_text, next_slot, is_completed
```

## Testing Framework

### Test Architecture Overview
The testing system provides comprehensive validation of the voice bot through multiple personas and acoustic conditions.

### 1. HTTP Conversation Tester (`test_harness/http_conversation_tester.py`)
```python
class HTTPPersonaConversationTester:
    """Tests complete conversation flows with isolated personas."""
    
    def __init__(self):
        self.personas = {
            "english_man": {...},      # Professional English speaker
            "calm_hindi": {...},       # Calm Hindi speaker  
            "energetic_hindi": {...},  # Energetic Hindi speaker
            "expressive_hindi": {...}  # Expressive Hindi speaker
        }
```

**Test Personas:**
- **English Man**: Professional tone, clean pronunciation, formal responses
- **Calm Hindi**: Slower pace, clear articulation, traditional vocabulary
- **Energetic Hindi**: Fast speech, colloquial expressions, high energy
- **Expressive Hindi**: Emotional responses, varied intonation, corrections

### 2. Comprehensive Test Flow
```python
async def run_persona_test(self, persona_key: str) -> PersonaTestResult:
    # 1. Start conversation session
    candidate_id = await self.start_conversation_session(client)
    
    # 2. Process each conversation step
    for step_config in persona["conversation_flow"]:
        turn_result = await self.send_audio_turn(
            client, candidate_id, 
            step_config["audio_file"], 
            step_config["step"],
            step_config.get("expected_entity", {})
        )
        turns.append(turn_result)
        
    # 3. Calculate comprehensive metrics
    return PersonaTestResult(...)
```

### 3. Test Metrics & Analysis
```python
@dataclass
class PersonaTestResult:
    # Conversation flow metrics
    completed_steps: int
    total_steps: int
    reached_job_matching: bool
    conversation_completed: bool
    
    # Performance metrics  
    total_latency_ms: float
    avg_turn_latency_ms: float
    avg_confidence: float
    confidence_range: Tuple[float, float]
    
    # Detailed turn analysis
    turns: List[ConversationTurn]
```

**Metrics Tracked:**
- **Completion Rate**: Percentage of conversation steps successfully completed
- **Latency Analysis**: P50, P95 response times for performance monitoring
- **Confidence Distribution**: ASR confidence across different voice personas
- **Entity Extraction Accuracy**: Success rate for each information type
- **Conversation Flow**: Step-by-step progression through interview

### 4. Noise Condition Testing
The test harness validates performance under various acoustic conditions:

```yaml
# Noise levels defined in utterances.yaml:314-327
noise:
  clean: {type: null, snr_db: null}
  low: {type: "traffic", snr_db: 25}
  medium: {type: "office", snr_db: 15}  
  high: {type: "construction", snr_db: 10}
```

**Test Strategy:**
- **Selective Noise Application**: Only utterances ending in "_001" and "_003" get noise added
- **Noise Types**: Traffic, office, construction sounds at different SNR levels
- **A/B Comparison**: Clean vs noisy versions for confidence degradation analysis

## Audio File Creation System

### 1. Audio Generation Pipeline (`test_harness/audio_generator.py`)
```python
class AudioGenerator:
    """Generates synthetic Hinglish audio with various accents and noise levels."""
    
    async def generate_test_audio(self, utterance_id: str = None) -> List[Dict]:
        # 1. Load utterance configurations
        utterances = self.config["utterances"]
        
        # 2. Generate clean audio using TTS
        success = await self.generate_speech(
            utterance["text"], 
            voice_id, 
            clean_path
        )
        
        # 3. Add noise for specific patterns
        if utterance_id.endswith("_001") or utterance_id.endswith("_003"):
            success = self.add_background_noise(clean_path, noisy_path, noise_level_db)
        
        return generated_files
```

### 2. Text-to-Speech Generation
**Primary: ElevenLabs TTS**
```python
async def generate_speech(self, text: str, voice_id: str, output_path: str) -> bool:
    # Configure multilingual model
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5, 
            "style": 0.5,
            "use_speaker_boost": True
        }
    }
    
    # Generate and save audio
    response = await client.post(url, json=data, headers=headers)
    with open(output_path, "wb") as f:
        f.write(response.content)
```

**Voice Configuration** (`utterances.yaml:307-311`):
```yaml
voices:
  english_man: "LruHrtVF6PSyGItzMNHS"     # English man Hindi
  calm_hindi: "1Z7Y8o9cvUeWq8oLKgMY"      # Calm Hindi  
  energetic_hindi: "IvLWq57RKibBrqZGpQrC"  # Energetic Hindi
  expressive_hindi: "ni6cdqyS9wBvic5LPA7M" # Expressive Hindi
```

### 3. Noise Addition System
```python
def add_background_noise(self, input_wav: str, output_wav: str, noise_level_db: int = -20) -> bool:
    # Load original audio
    audio = AudioSegment.from_wav(input_wav)
    
    # Generate white noise with same duration
    noise = AudioSegment(
        (np.random.randn(len(audio.get_array_of_samples())) * 32767).astype(np.int16).tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )
    
    # Adjust noise level and mix
    noise = noise - noise.dBFS + noise_level_db
    mixed = audio.overlay(noise)
    mixed.export(output_wav, format="wav")
```

### 4. Test Data Configuration (`utterances.yaml`)
**Utterance Structure:**
```yaml
- id: "pin_diverse_001"
  text: "Main Delhi mein rehta hun, mera pincode hai 110001"
  transcript: "main delhi mein rehta hun mera pincode hai ek lakh das hazar ek"
  entities: {pincode: "110001"}
  accent: "english_man"
  noise_level: "high"
  duration_range: [6, 8]
```

**Coverage Matrix:**
- **8 Information Types**: Pincode, salary, vehicle, languages, shifts, availability, experience, confirmation
- **4 Voice Personas**: Different accents and speaking styles  
- **4 Noise Levels**: Clean to construction-level background noise
- **32 Total Utterances**: Comprehensive test coverage

### 5. Audio Manifest Generation
```python
def create_audio_manifest(self, generated_files: List[Dict]) -> str:
    manifest = {
        "generated_at": str(asyncio.get_event_loop().time()),
        "total_files": len(generated_files),
        "files": generated_files
    }
    
    with open(manifest_path, 'w', encoding='utf-8') as file:
        yaml.dump(manifest, file, default_flow_style=False, allow_unicode=True)
```

The manifest tracks all generated audio files with metadata for test execution and analysis.

## Key Components

### 1. Conversation Orchestrator (`app/services/conversation.py`)
**Core Responsibilities:**
- Manages conversation state and flow progression
- Coordinates ASR, NLU, and TTS services
- Implements intelligent response generation
- Handles entity confirmation and validation
- Tracks conversation metrics and analytics

### 2. ElevenLabs ASR Service (`app/services/elevenlabs_asr.py`) 
**Features:**
- Optimized for Hinglish speech recognition
- Multi-factor confidence calculation
- Text normalization for better entity extraction
- Fallback handling for API failures

### 3. NLU Service (`app/services/nlu.py`)
**Capabilities:**
- Multi-tier entity extraction (rules + LLM)
- Comprehensive text normalization (Devanagari + Romanized)
- Entity validation and confidence adjustment
- Dynamic pincode resolution from city names
- Confirmation dialog management

### 4. Job Matching Service (`app/services/job_matching.py`)
**Algorithm:**
- Multi-dimensional scoring (location, salary, skills, experience)
- Weighted scoring with configurable parameters
- Personalized candidate summaries using LLM
- Comprehensive match explanations

### 5. Web Interface (`static/index.html`, `static/app.js`)
**Features:**
- Push-to-talk voice interface
- Real-time conversation logging
- Live metrics dashboard
- Candidate profile visualization
- Job matching results display

## API Reference

### Core Endpoints

#### Start Conversation
```
POST /api/v1/conversation/start
Response: {
  "candidate_id": "uuid",
  "audio_data": "base64_wav",
  "audio_format": "wav",
  "message": "Conversation started successfully"
}
```

#### Process Voice Turn (Fast Path)
```
POST /api/v1/conversation/{candidate_id}/turn-fast
Content-Type: multipart/form-data
Body: audio_file (WebM/WAV)

Response: {
  "candidate_id": "uuid",
  "turn_id": "uuid", 
  "text": "Bot response text",
  "conversation_complete": false,
  "asr": {
    "text": "Transcribed user speech",
    "confidence": 0.85,
    "raw_confidence_data": {...}
  },
  "metrics": {...}
}
```

#### Fetch Background TTS Audio
```
GET /api/v1/conversation/{candidate_id}/turn-audio/{turn_id}
Response: {
  "ready": true,
  "audio_data": "base64_wav",
  "audio_format": "wav"
}
```

#### Get Conversation Status  
```
GET /api/v1/conversation/{candidate_id}/status
Response: {
  "candidate_id": "uuid",
  "current_field": "expected_salary",
  "completion_rate": 0.6,
  "conversation_complete": false,
  "candidate_profile": {...}
}
```

#### Get Job Matches
```
GET /api/v1/conversation/{candidate_id}/matches
Response: {
  "candidate_id": "uuid", 
  "matches": [{
    "job": {...},
    "match_score": 0.85,
    "rationale": "Strong location and salary match...",
    "score_breakdown": {...}
  }],
  "total_jobs_considered": 150
}
```

## Development Setup

### Prerequisites
- Python 3.8+
- Node.js (for development tools)
- ElevenLabs API key
- OpenAI API key
- Google Cloud credentials (optional)

### Environment Configuration
```bash
# Required API keys
ELEVENLABS_API_KEY=your_elevenlabs_key
OPENAI_API_KEY=your_openai_key

# Optional Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json

# Server configuration
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=development
```

### Installation & Running
```bash
# Install dependencies
pip install -r requirements.txt

# Generate test audio (optional)
python test_harness/audio_generator.py

# Start development server
python -m uvicorn app.main:app --reload

# Run tests
python test_harness/http_conversation_tester.py
```

### Testing Commands
```bash
# Test single persona
python test_harness/http_conversation_tester.py --persona english_man

# Test all personas
python test_harness/http_conversation_tester.py

# Generate new test audio
python test_harness/audio_generator.py
```

## Configuration

### Conversation Flow
The system follows a structured interview flow defined in `app/services/conversation.py:65-76`:

1. **Greeting**: Welcome and introduction
2. **Pincode**: Location information
3. **Expected Salary**: Salary expectations
4. **Two Wheeler**: Vehicle ownership
5. **Languages**: Language skills
6. **Availability**: Start date preference  
7. **Preferred Shift**: Timing preferences
8. **Experience**: Work experience
9. **Confirmation**: Validate all collected information
10. **Summary**: Generate job matches and recommendations

### Audio Settings
- **Sample Rate**: 16kHz (optimized for speech recognition)
- **Format**: WAV/WebM (client) → WAV (server processing)
- **Channels**: Mono (reduces bandwidth and improves ASR accuracy)
- **Minimum Recording**: 300ms (prevents accidental triggers)

### Performance Thresholds
- **ASR Confidence**: 0.5 minimum for entity extraction
- **Response Time**: < 2000ms target for complete turns
- **Voice Confidence**: 70% minimum for reliable operation
- **Completion Rate**: 80% target for successful conversations

This comprehensive system provides robust, multilingual voice interactions for employment screening while maintaining high accuracy and user experience standards.