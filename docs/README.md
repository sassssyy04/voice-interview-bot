# Hinglish Voice Bot - Development Guide

## 1. Setup

```bash
poetry lock --no-interaction
poetry install --no-interaction
```

### Environment Configuration

Create a `.env` file at project root as needed (keys are optional for demo mode):

```env
# Optional API Keys
OPENAI_API_KEY=...
ELEVENLABS_API_KEY=...
GOOGLE_APPLICATION_CREDENTIALS=path\to\gcp.json

# Server Configuration
HOST=127.0.0.1
PORT=8000

```

## 2. Local Development

```bash
make dev
```

**Without make:**

```bash
poetry run python scripts/dev.py
```


## 3. Evaluation

```bash
make eval
```

**Without make:**

```bash
poetry run python scripts/eval.py
```
