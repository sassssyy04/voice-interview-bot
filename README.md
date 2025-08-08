# Hinglish Voice Bot for Blue-Collar Job Screening

A complete voice-based interview system that screens blue-collar candidates in Hinglish (Hindi + English code-switching) and provides job matching with transparent rationales.

## 🎯 Features

### ✅ Voice Conversation (Mandatory)
- **Real-time ASR**: Azure Speech + Google Speech with Hinglish support
- **Streaming TTS**: Hindi neural voices with <2.0s response times
- **Barge-in Support**: Automatic TTS interruption when user speaks
- **Error Recovery**: Explicit confirmation for fuzzy entities
- **Memory/Statefulness**: No re-asking, conversation recap
- **Low-literacy Friendly**: Short prompts, one concept per question
- **Fallback Systems**: Multiple ASR providers, confidence thresholds

### ✅ Job Matching Core
- **Transparent Scoring**: Location (30%), Salary (25%), Shift (20%), Language (15%), Vehicle (5%), Experience (5%)
- **Human Rationales**: Plain-language explanations for each match
- **Hard Constraints**: Enforced requirements with clear explanations
- **Top 3 Results**: Best matches with detailed breakdowns

### ✅ Telemetry & Reliability
- **Comprehensive Logging**: Per-turn ASR confidence, entity extraction, latency timestamps
- **Performance Metrics**: P50/P95 latency, completion rates, slot capture rates
- **Cost Tracking**: Character counts for TTS, API usage monitoring
- **Graceful Degradation**: Multiple ASR fallbacks, error handling

### ✅ Voice Requirements
- **Hinglish ASR**: Handles code-switching, accents, noisy environments
- **Neutral TTS**: Indian voice, <7s responses, proper pacing
- **Text Normalization**: Salaries (₹/mo), dates ("parso"), localities
- **Consent Compliance**: Recording disclosure in first message

## 🏗️ Architecture

```
Browser Mic → ASR (Azure/Google) → NLU (OpenAI + Rules) → 
State Management → TTS (Azure/Google) → Browser Speaker
                     ↓
Job Matching Engine → Top 3 Results + Rationales
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Poetry (for dependency management)
- Modern web browser with microphone support

### 1. Install Dependencies

```bash
# Clone and navigate to project
cd hinglish-voice-bot

# Install with Poetry (enters virtual environment automatically)
poetry install
poetry shell
```

### 2. Configure API Keys

```bash
# Copy environment template
cp env_example.txt .env

# Edit .env with your API keys
OPENAI_API_KEY=your_openai_api_key_here
AZURE_SPEECH_KEY=your_azure_speech_key_here
AZURE_SPEECH_REGION=your_azure_region_here
GOOGLE_APPLICATION_CREDENTIALS=path/to/google/credentials.json
```

### 3. Run the Application

```bash
# Start the voice bot server
python -m app.main

# Or using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open Web Interface

Navigate to: `http://localhost:8000`

## 🎤 Demo Instructions

### Basic Demo Flow
1. **Click "Start Interview"** - Grants microphone permission
2. **Listen to Hinglish greeting** - Bot introduces itself
3. **Speak responses** - Use Hinglish like:
   - "Mera pincode 110001 hai"
   - "Main kal se start kar sakta hun"
   - "Evening shift prefer karta hun"
   - "Salary expectation 18 hazaar hai"

### Testing Noise Resistance
1. **Play background audio** during conversation:
   - Traffic noise
   - Scooter sounds  
   - Office chatter
2. **Check confidence scores** in real-time metrics
3. **Verify fallback behavior** when confidence drops

### Conversation Fields Collected
- **Location**: Pincode/locality
- **Availability**: Start date preference
- **Shift**: Morning/afternoon/evening/night/flexible
- **Salary**: Monthly expectation in INR
- **Languages**: Hindi, English, regional languages
- **Vehicle**: Two-wheeler ownership
- **Experience**: Total months of work experience

## 📊 Metrics & Monitoring

### Real-time Dashboard
- **Response Time**: Target <2000ms (P50/P95 tracking)
- **Voice Confidence**: ASR accuracy scores
- **Progress Tracking**: Fields completed percentage
- **Turn Analytics**: Average conversation length

### Logs Location
- **General Logs**: `logs/voice_bot.log`
- **Conversations**: `logs/conversations.log` 
- **Performance**: `logs/metrics.log`

### API Endpoints
- `GET /api/v1/health` - Health check
- `POST /api/v1/conversation/start` - Start new conversation
- `POST /api/v1/conversation/{id}/turn` - Process voice turn
- `GET /api/v1/conversation/{id}/matches` - Get job matches
- `GET /api/v1/metrics/dashboard` - Aggregated metrics

## 🔄 Job Matching Algorithm

### Scoring Formula
```python
Overall Score = (
    Location Score × 0.30 +     # Distance-based
    Salary Score × 0.25 +       # Expectation fit
    Shift Score × 0.20 +        # Preference match
    Language Score × 0.15 +     # Required languages
    Vehicle Score × 0.05 +      # Two-wheeler requirement
    Experience Score × 0.05     # Experience fit
)
```

### Sample Rationale Output
> "Excellent match for Delivery Executive at QuickDeliver! Key strengths: Job is very close (1.2km from your location); Salary range (₹15,000-₹22,000) matches your expectations; You have the required two-wheeler. Contact: +91-9876543210"

## 🛠️ Development

### Project Structure
```
app/
├── api/routes.py          # FastAPI endpoints
├── core/
│   ├── config.py          # Settings management
│   └── logger.py          # Structured logging
├── models/
│   ├── candidate.py       # Data models
│   └── job.py            # Job matching models
├── services/
│   ├── speech_recognition.py  # ASR service
│   ├── text_to_speech.py     # TTS service  
│   ├── nlu.py                # Entity extraction
│   ├── conversation.py       # Orchestrator
│   └── job_matching.py       # Matching engine
└── main.py               # FastAPI app

static/
├── index.html            # Web interface
└── app.js               # Frontend JavaScript

data/
├── candidates.json       # Sample candidate data
└── jobs.json            # Sample job listings
```

### Adding New Features

1. **New Conversation Fields**: Update `conversation_flow` in `conversation.py`
2. **New Job Categories**: Add to `JobCategory` enum in `models/job.py`
3. **New Languages**: Extend `LanguageSkill` enum in `models/candidate.py`
4. **Custom Matching Rules**: Modify scoring in `job_matching.py`

### Testing

```bash
# Run tests
poetry run pytest

# Test specific components
python -m pytest tests/test_asr.py
python -m pytest tests/test_matching.py

# Load testing
python scripts/load_test.py
```

## 🔧 Configuration

### Voice Settings
```env
# Speech recognition
CONFIDENCE_THRESHOLD=0.7
MAX_RESPONSE_TIME=2.0

# Text-to-speech  
SPEECH_RATE=1.0
SPEECH_PITCH=0.0
DEFAULT_VOICE_LANGUAGE=hi-IN

# Job matching
MAX_DISTANCE_KM=50
SALARY_TOLERANCE_PERCENT=20
```

### API Provider Priorities
1. **ASR**: Azure Speech (primary) → Google Speech (fallback) → SpeechRecognition (last resort)
2. **TTS**: Azure Speech (primary) → Google TTS (fallback)
3. **NLU**: Rule-based extraction (primary) → OpenAI GPT-3.5 (fallback)

## 📈 Performance Benchmarks

### Target Metrics
- **Response Time**: P50 < 1500ms, P95 < 2500ms
- **ASR Accuracy**: >85% for clean audio, >70% with noise
- **Conversation Completion**: >90% for engaged users
- **Entity Extraction**: >95% F1 score for clear speech

### Cost Estimates (per candidate)
- **Azure Speech**: ~₹2-3 (ASR + TTS)
- **OpenAI API**: ~₹1-2 (entity extraction fallback)
- **Total**: ~₹3-5 per complete conversation

## 🔒 Security & Privacy

### Data Protection
- **No audio storage**: Audio processed in real-time, not saved
- **Encrypted transmission**: HTTPS for all API calls
- **Consent compliance**: Recording disclosure in first message
- **Data retention**: Logs auto-purged after 30 days

### Production Considerations
- Rate limiting on API endpoints
- Input validation for all voice data
- CORS configuration for production domains
- API key rotation and monitoring

## 🐛 Troubleshooting

### Common Issues

1. **Microphone Permission Denied**
   - Use HTTPS in production
   - Check browser security settings
   - Try different browser

2. **High Response Times**
   - Check API key limits
   - Monitor network latency
   - Verify service status

3. **Low ASR Confidence**
   - Reduce background noise
   - Speak clearly and slowly
   - Check microphone quality

4. **TTS Playback Issues**
   - Check browser audio settings
   - Verify CORS headers
   - Try different audio format

### Debug Mode
```bash
# Enable debug logging
LOG_LEVEL=DEBUG python -m app.main

# Check real-time logs
tail -f logs/voice_bot.log
```

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📞 Support

For issues and questions:
- **GitHub Issues**: Technical problems and feature requests
- **Documentation**: Check this README and inline code comments
- **Performance Issues**: Check metrics dashboard and logs

---

**Built with ❤️ for the blue-collar workforce in India** 