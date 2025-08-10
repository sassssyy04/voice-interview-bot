# Hinglish Voice Bot Architecture Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Core Architecture](#core-architecture)
3. [Data Flow](#data-flow)
4. [Service Components](#service-components)
5. [Fallback Mechanisms](#fallback-mechanisms)
6. [State Management](#state-management)
7. [Job Matching Engine](#job-matching-engine)
8. [Design Justifications](#design-justifications)
9. [Performance & Monitoring](#performance--monitoring)
10. [API Interface](#api-interface)

## System Overview

The Hinglish Voice Bot is a sophisticated multi-modal conversational AI system designed specifically for blue-collar job screening in India. It combines real-time voice processing, natural language understanding, and intelligent job matching to create an accessible, low-literacy-friendly recruitment platform.

### Key Capabilities
- **Real-time voice conversation** in Hinglish (Hindi + English code-switching)
- **Intelligent entity extraction** from conversational speech
- **Transparent job matching** with human-readable rationales
- **Multi-provider fallbacks** for reliability
- **Comprehensive telemetry** for performance monitoring

## Core Architecture

The system follows a **microservices-inspired modular architecture** with clear separation of concerns:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Client    │◄──►│   FastAPI App    │◄──►│  Static Assets  │
│  (Browser JS)   │    │   (main.py)      │    │   (HTML/CSS)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │    API Routes Layer    │
                    │     (routes.py)        │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │ Conversation Orchestrator│
                    │   (conversation.py)    │
                    └────┬─────┬─────┬───────┘
                         │     │     │
              ┌──────────▼┐   ┌▼─────▼─────┐   ┌▼────────────┐
              │    ASR    │   │    NLU     │   │     TTS     │
              │ Service   │   │  Service   │   │   Service   │
              └───────────┘   └────────────┘   └─────────────┘
                         │                            │
              ┌──────────▼─────────┐          ┌─────▼─────────┐
              │   Job Matching     │          │  State Mgmt   │
              │     Engine         │          │  & Models     │
              └────────────────────┘          └───────────────┘
```

### Technology Stack

**Backend Framework**
- **FastAPI**: Chosen for async support, automatic API documentation, and WebSocket capabilities
- **Uvicorn**: ASGI server for high-performance async operations
- **Pydantic**: Data validation and serialization with type safety

**Voice Processing**
- **Primary ASR**: Google Cloud Speech-to-Text (best Hinglish support)
- **Fallback ASR**: Local SpeechRecognition library
- **Primary TTS**: ElevenLabs (natural multilingual voices)
- **Fallback TTS**: Google Cloud Text-to-Speech, then offline pyttsx3

**AI/ML Services**
- **OpenAI GPT-4**: Contextual response generation and entity extraction fallback
- **Rule-based NLU**: Primary entity extraction for better control and speed

**Data & Persistence**
- **In-memory storage**: Demo-appropriate, easily replaceable with Redis/PostgreSQL
- **JSON data files**: Job listings and sample candidates

## Data Flow

### 1. Conversation Initiation Flow

```
User Request → FastAPI Router → ConversationOrchestrator.start_conversation()
     │
     ▼
Create Candidate & State Objects → Generate Greeting → TTS Synthesis
     │
     ▼
Return (candidate_id, audio_data) → Base64 Encode → Send to Client
```

**Key Components:**
- **UUID Generation**: Each conversation gets a unique identifier
- **State Initialization**: `ConversationState` tracks progress and metadata
- **Welcome Audio**: Pre-synthesized Hinglish greeting sets the tone

### 2. Voice Turn Processing Flow

```
Audio Upload → ConversationOrchestrator.process_turn()
     │
     ▼
┌─ ASR Service ─┐   ┌─ Confidence Check ─┐   ┌─ Entity Extraction ─┐
│ Audio → Text  │ → │ Quality Validation │ → │   NLU Processing    │
└───────────────┘   └────────────────────┘   └─────────────────────┘
     │
     ▼
┌─ Response Generation ─┐   ┌─ TTS Synthesis ─┐   ┌─ State Update ─┐
│  LLM + Rule-based    │ → │   Text → Audio   │ → │  Progress Track │
└──────────────────────┘   └──────────────────┘   └────────────────┘
     │
     ▼
Return (response_text, audio_data, completion_status)
```

### 3. Entity Extraction Pipeline

The NLU service employs a **hybrid approach** combining speed and accuracy:

```
User Input → Rule-based Extraction → Success? → Update Candidate
     │                                  │
     ▼ (if failed)                     ▼ (if success)
OpenAI GPT-4 Fallback → Validation → State Management
```

**Extraction Strategies by Field:**

- **Pincode**: Regex patterns (`\d{6}`) with validation against location database
- **Salary**: Number extraction with currency normalization ("18 hazaar" → 18000)
- **Dates**: Relative date parsing ("kal", "parso") with fuzzy matching
- **Shifts**: Keyword matching with synonyms ("sham" → "evening")
- **Languages**: Multi-language detection with regional variants
- **Boolean fields**: Intent classification for yes/no responses

### 4. Job Matching Flow

```
Candidate Profile → Load Job Database → Calculate Match Scores
     │
     ▼
┌─ Location Scoring ─┐   ┌─ Salary Matching ─┐   ┌─ Requirement Check ─┐
│  Distance & Zones  │ + │  Range Overlap    │ + │  Hard Constraints   │
└────────────────────┘   └───────────────────┘   └─────────────────────┘
     │
     ▼
Sort by Score → Top 3 Selection → Generate Rationales → Return Results
```

## Service Components

### 1. ASR Service (`speech_recognition.py`)

**Multi-Provider Fallback Strategy:**

```python
async def recognize_speech_from_audio(self, audio_data: bytes):
    try:
        # Primary: Google Cloud Speech (best for Hinglish)
        if self.google_client:
            return await self._recognize_with_google(audio_data)
    except Exception:
        # Fallback: SpeechRecognition library
        if self.recognizer:
            return await self._recognize_with_speechrecognition(audio_data)
```

**Design Justifications:**
- **Google Cloud Primary**: Superior Hinglish recognition, handles code-switching
- **Local Library Backup**: Basic functionality when cloud services fail
- **Confidence Scoring**: Quality gating to prevent poor transcriptions

### 2. TTS Service (`text_to_speech.py`)

**Voice Selection Strategy:**
- **Primary Voice**: ElevenLabs (configurable voice id), natural multilingual
- **Fallback**: Google TTS, then offline pyttsx3
- **Rate Control**: Configurable speech rate for comprehension
- **Text Preprocessing**: Hinglish normalization and pronunciation hints

```python
def _prepare_hinglish_text(self, text: str) -> str:
    # Normalize currency mentions, punctuation, and length
    text = text.replace('।', '.')
    if len(text) > 500:
        text = text[:500] + '...'
    return text
```

### 3. NLU Service (`nlu.py`)

**Hybrid Processing Architecture:**

```python
async def extract_entities(self, text: str, field: str):
    # Implementation details...
```

**Field-Specific Extraction:**

- **Pincode Extraction**:
  ```python
  pincode_patterns = [
      r'\b(\d{6})\b',  # Direct 6-digit numbers
      r'pincode\s*(\d{6})',  # "pincode 110001"
      r'pin\s*(\d{6})'  # "pin 110001"
  ]
  ```

- **Salary Processing**:
  ```python
  def _normalize_salary(self, amount_str: str) -> int:
      # Handle Hindi number words
      if 'hazaar' in amount_str.lower():
          num = self._extract_number(amount_str)
          return num * 1000
      # Handle direct numbers with currency
      return self._extract_currency_amount(amount_str)
  ```

**Design Justifications:**
- **Rule-based Primary**: Faster, more predictable for structured data
- **LLM Fallback**: Handles complex, ambiguous, or conversational input
- **Field-specific Logic**: Optimized extraction for each data type
- **Confidence Scoring**: Quality control and fallback triggering

### 4. Conversation Orchestrator (`conversation.py`)

**State-Driven Flow Management:**

The orchestrator maintains a **finite state machine** approach:

```python
conversation_flow = [
    "greeting", "pincode", "availability_date", "preferred_shift",
    "expected_salary", "languages", "has_two_wheeler", 
    "total_experience_months", "confirmation", "summary"
]
```

**Intelligent Response Generation:**

```python
async def _generate_intelligent_response(self, user_text, confidence, candidate, state, history):
    # 1. Determine missing information
    missing_slots = self._get_missing_slots(state)
    
    # 2. Try entity extraction if user provided input
    if user_text and confidence > 0.3:
        extracted = await self._try_extract_entities(user_text, missing_slots)
    
    # 3. Generate contextual response
    if extracted:
        return self._standard_next_prompt()
    else:
        return await self._contextual_llm_response(user_text, history)
```

**Design Justifications:**
- **Linear Flow**: Predictable progress, easy to track completion
- **Smart Backtracking**: Can collect any missing field at any time
- **Context Awareness**: Uses conversation history for better responses
- **Confirmation Step**: Validates collected data before job matching

## Fallback Mechanisms

### 1. ASR Fallback Chain

**Failure Modes Handled:**
- API rate limiting or quota exhaustion
- Network connectivity issues
- Service-specific outages
- Low audio quality or noise

**Fallback Sequence:**
1. **Google Cloud Speech** (primary)
   - Best Hinglish support
   - Real-time streaming capability
   - Automatic language detection

2. **Hugging Face Whisper** (fallback)
   - Offline processing capability
   - No API dependency
   - Good general accuracy

3. **SpeechRecognition Library** (last resort)
   - Basic functionality
   - Multiple backend options
   - Local processing

### 2. TTS Fallback Strategy

**Quality Degradation Gracefully:**
```python
async def synthesize_speech(self, text: str) -> bytes:
    try:
        return await self._synthesize_with_google(text)
    except Exception as e:
        logger.warning(f"Google TTS failed: {e}")
        if self.hf_tts:
            return await self._synthesize_with_hf(text)
        else:
            # Fallback to text response
            return text.encode('utf-8')
```

### 3. NLU Fallback Hierarchy

**Processing Chain:**
1. **Rule-based extraction** (primary)
   - Fast, predictable
   - Field-specific patterns
   - High confidence for structured input

2. **OpenAI GPT-4** (fallback)
   - Handles ambiguous input
   - Contextual understanding
   - Conversational entity extraction

3. **Clarification prompts** (last resort)
   - Ask for specific format
   - Provide examples
   - Guide user to expected input

### 4. Service Availability Handling

**Graceful Degradation:**
```python
def __init__(self):
    try:
        self.asr_service = ASRService()
        self.tts_service = TTSService()
        self.nlu_service = NLUService()
    except ImportError as e:
        logger.warning(f"Voice services not available: {e}")
        # Continue with text-only mode
        self.asr_service = None
        self.tts_service = None
        self.nlu_service = None
```

## State Management

### 1. Conversation State Model

```python
class ConversationState(BaseModel):
    candidate_id: str
    current_field: str = "greeting"
    current_step: int = 0
    fields_completed: List[str] = []
    retry_count: int = 0
    last_confidence: float = 0.0
    needs_confirmation: bool = False
    pending_confirmation_value: Optional[str] = None
```

**State Transitions:**
- **Linear Progression**: Move through predefined flow
- **Dynamic Jumping**: Skip to any missing field based on user input
- **Retry Logic**: Handle unclear responses with limited retries
- **Confirmation Cycles**: Validate ambiguous or low-confidence extractions

### 2. Candidate Data Model

**Structured Profile:**
```python
class Candidate(BaseModel):
    # Identity
    candidate_id: str
    created_at: datetime
    
    # Location
    pincode: Optional[str]
    locality: Optional[str]
    
    # Preferences
    availability_date: Optional[str]
    preferred_shift: Optional[ShiftPreference]
    expected_salary: Optional[int]
    
    # Capabilities
    languages: List[LanguageSkill]
    has_two_wheeler: Optional[bool]
    total_experience_months: Optional[int]
    
    # Metadata
    conversation_completed: bool
    turn_count: int
```

**Design Justifications:**
- **Optional Fields**: Graceful handling of partial data
- **Type Safety**: Pydantic validation prevents data corruption
- **Extensibility**: Easy to add new fields without breaking existing logic
- **Serialization**: Direct JSON conversion for API responses

### 3. Memory Management

**In-Memory Storage Strategy:**
```python
# Conversation storage
self.candidates: Dict[str, Candidate] = {}
self.conversation_states: Dict[str, ConversationState] = {}
self.conversation_turns: Dict[str, list] = {}
```

**Production Considerations:**
- **Current**: In-memory for demo simplicity
- **Production**: Redis for session state, PostgreSQL for candidate profiles
- **Cleanup**: Automatic purging of incomplete conversations
- **Backup**: Periodic state snapshots for recovery

## Job Matching Engine

### 1. Scoring Algorithm

**Weighted Scoring Formula:**
```python
overall_score = (
    location_score × 0.30 +      # Distance and transportation
    salary_score × 0.25 +        # Expectation alignment
    shift_score × 0.20 +         # Schedule preference
    language_score × 0.15 +      # Communication capability
    vehicle_score × 0.05 +       # Transportation requirement
    experience_score × 0.05      # Skill level fit
)
```

**Justification for Weights:**
- **Location (30%)**: Most critical for blue-collar jobs (commute constraints)
- **Salary (25%)**: Primary motivation factor
- **Shift (20%)**: Work-life balance important for retention
- **Language (15%)**: Communication essential but most speak Hindi
- **Vehicle (5%)**: Binary requirement, affects fewer roles
- **Experience (5%)**: Less critical for entry-level positions

### 2. Location Scoring

**Distance-Based Calculation:**
```python
def _calculate_location_score(self, candidate_pincode: str, job_pincode: str) -> float:
    distance_km = self._calculate_distance(candidate_pincode, job_pincode)
    
    if distance_km <= 5:
        return 1.0  # Perfect score for nearby jobs
    elif distance_km <= 15:
        return 0.8  # Good score for reasonable commute
    elif distance_km <= 30:
        return 0.5  # Acceptable for higher-paying roles
    else:
        return 0.1  # Poor score for distant jobs
```

**Geographic Data:**
- **Pincode Database**: Maps to lat/lng coordinates
- **Distance Calculation**: Haversine formula for accuracy
- **Zone Awareness**: Metro vs suburban vs rural considerations

### 3. Salary Matching

**Range Overlap Algorithm:**
```python
def _calculate_salary_score(self, expected: int, job_min: int, job_max: int) -> float:
    if job_min <= expected <= job_max:
        return 1.0  # Perfect fit within range
    
    tolerance = expected * 0.2  # 20% tolerance
    if abs(expected - job_min) <= tolerance or abs(expected - job_max) <= tolerance:
        return 0.8  # Close enough
    
    # Penalty for significant mismatch
    if expected < job_min:
        return max(0.3, expected / job_min)
    else:
        return max(0.3, job_max / expected)
```

### 4. Rationale Generation

**Human-Readable Explanations:**
```python
async def _generate_match_rationale(self, candidate: Candidate, job: Job, match: JobMatch) -> str:
    strengths = []
    considerations = []
    
    # Distance analysis
    if match.location_score >= 0.8:
        strengths.append(f"Job is very close ({distance:.1f}km from your location)")
    
    # Salary analysis
    if match.salary_score >= 0.8:
        strengths.append(f"Salary range (₹{job.salary_min:,}-₹{job.salary_max:,}) matches your expectations")
    
    # Generate final rationale
    return f"{'Excellent' if match.match_score >= 0.8 else 'Good'} match! " + \
           f"Key strengths: {'; '.join(strengths)}. Contact: {job.contact_number}"
```

## Design Justifications

### 1. Architecture Choices

**Modular Service Design:**
- **Rationale**: Separation of concerns, independent testing, scalability
- **Alternative**: Monolithic approach would be simpler but less maintainable
- **Trade-off**: Slightly more complexity for much better modularity

**FastAPI Framework:**
- **Rationale**: Async support for voice processing, automatic API docs, WebSocket support
- **Alternative**: Flask would be simpler but lacks native async support
- **Trade-off**: Learning curve for modern async patterns vs. better performance

**In-Memory State:**
- **Rationale**: Simplicity for demo, no database setup required
- **Alternative**: Redis/PostgreSQL would be production-ready
- **Trade-off**: Data loss on restart vs. zero infrastructure requirements

### 2. Voice Processing Choices

**Multi-Provider Strategy:**
- **Rationale**: Reliability through redundancy, cost optimization
- **Alternative**: Single provider would be simpler but less reliable
- **Trade-off**: Implementation complexity vs. system robustness

**Google Cloud Primary:**
- **Rationale**: Best Hinglish support, handles code-switching naturally
- **Alternative**: Azure has good Hindi support but less code-switching capability
- **Trade-off**: Vendor lock-in vs. superior language support

**Confidence Thresholding:**
- **Rationale**: Quality gating prevents cascading errors from poor transcription
- **Alternative**: Accept all transcriptions would be simpler
- **Trade-off**: Occasional clarification requests vs. data quality

### 3. NLU Design Decisions

**Hybrid Rule-Based + LLM:**
- **Rationale**: Speed and cost of rules, flexibility of LLM for edge cases
- **Alternative**: Pure LLM would be more flexible but slower and expensive
- **Trade-off**: Implementation complexity vs. optimal performance/cost balance

**Field-Specific Extraction:**
- **Rationale**: Optimized patterns for each data type improve accuracy
- **Alternative**: Generic extraction would be simpler but less accurate
- **Trade-off**: More code to maintain vs. better extraction quality

### 4. Job Matching Philosophy

**Transparent Scoring:**
- **Rationale**: Builds trust with candidates, explains decision process
- **Alternative**: Black-box matching would be simpler
- **Trade-off**: More complex rationale generation vs. user trust and understanding

**Weighted Algorithm:**
- **Rationale**: Reflects real-world priorities for blue-collar job seekers
- **Alternative**: Equal weighting would be simpler but less realistic
- **Trade-off**: Need to tune weights vs. more relevant results

## Performance & Monitoring

### 1. Logging Architecture

**Structured Logging with Loguru:**
```python
# Different log streams for different purposes
logger.add("logs/voice_bot.log", serialize=True)          # General app logs
logger.add("logs/conversations.log", filter=conversation)  # User interactions
logger.add("logs/metrics.log", filter=metrics)            # Performance data
```

**Key Metrics Tracked:**
- **Latency**: ASR, NLU, TTS processing times
- **Quality**: ASR confidence scores, entity extraction success rates
- **Usage**: API call counts, conversation completion rates
- **Errors**: Service failures, fallback activations

### 2. Performance Targets

**Response Time Goals:**
- **Overall Turn**: < 2000ms (P95)
- **ASR Processing**: < 800ms (P95)
- **TTS Generation**: < 1000ms (P95)
- **NLU Extraction**: < 200ms (P95)

**Quality Targets:**
- **ASR Accuracy**: > 85% for clean audio, > 70% with background noise
- **Entity Extraction**: > 95% F1 score for clear speech
- **Conversation Completion**: > 90% for engaged users

### 3. Cost Monitoring

**Per-Conversation Estimates:**
```python
# Azure Speech: ₹2-3 (ASR + TTS)
# OpenAI API: ₹1-2 (entity extraction fallback)
# Total: ₹3-5 per complete conversation
```

**Cost Optimization Strategies:**
- **Rule-based NLU**: Reduces OpenAI API calls by ~70%
- **Confidence thresholding**: Prevents expensive re-processing
- **Local fallbacks**: Reduces cloud API dependency

## API Interface

### 1. RESTful Endpoints

**Core Conversation Flow:**
```http
POST /api/v1/conversation/start
POST /api/v1/conversation/{id}/turn
GET  /api/v1/conversation/{id}/matches
```

**Monitoring & Health:**
```http
GET /api/v1/health
GET /api/v1/metrics/dashboard
```

### 2. WebSocket Support

**Real-time Communication:**
- Streaming audio processing
- Live transcription updates
- Immediate feedback for user experience

### 3. Response Formats

**Standardized JSON Responses:**
```json
{
  "candidate_id": "uuid",
  "text": "response_text",
  "audio_data": "base64_encoded_audio",
  "audio_format": "wav",
  "conversation_complete": false,
  "metrics": {
    "response_time_ms": 1250,
    "confidence_score": 0.92,
    "completion_rate": 0.85
  }
}
```

**Design Justifications:**
- **Base64 Audio**: Web-compatible, no file handling required
- **Embedded Metrics**: Real-time performance visibility
- **Status Flags**: Clear conversation state communication
- **Consistent Schema**: Predictable client-side handling

---

*This architecture documentation provides a comprehensive view of the Hinglish Voice Bot system, covering all major components, data flows, design decisions, and operational considerations. The system is designed for reliability, scalability, and maintainability while delivering an excellent user experience for blue-collar job seekers in India.* 