Quick Start
1. Environment Setup
Create a .env file in the project root with your API keys:

OPENAI_API_KEY=your_openai_api_key_here

# For Google Cloud TTS and ASR
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/google_credentials.json

# Optional: Fallback voice services
ELEVENLABS_API_KEY=your_elevenlabs_key_here
SARVAM_API_KEY=your_sarvam_key_here
Make sure the Google credentials file is downloaded from your Google Cloud project and that TTS and ASR APIs are enabled in the console.

2. Local Development

make dev
Installs dependencies, starts the dev server, and opens it in your browser.
(Server runs on http://localhost:8000 by default.)

3. Evaluation

make eval
Runs the evaluator and prints concise metrics.
Requires the dev server to be running.