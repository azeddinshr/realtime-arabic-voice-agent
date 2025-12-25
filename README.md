# Real Time Arabic Voice Agent with RAG

Real-time Arabic voice assistant with RAG (Retrieval-Augmented Generation) using OpenAI Realtime API and LiveKit.

## Features

- 🎙️ **Real-time Voice Interaction**: Unified speech-to-speech model (no STT/TTS pipeline)
- 🧠 **RAG System**: Arabic knowledge base from ArabicaQA
- 🌤️ **Weather Tool**: Real-time weather for any city worldwide
- 🔎 **Web Search**: Current information via Tavily API
- 🇦🇪 **Arabic Support**: Native Arabic language processing

## Architecture

- **Voice Model**: OpenAI Realtime API (gpt-4o-realtime-preview)
- **Framework**: LiveKit Agents
- **Vector DB**: ChromaDB with persistent storage
- **Embeddings**: paraphrase-multilingual-MiniLM-L12-v2
- **Turn Detection**: Semantic VAD

## Installation

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/azeddinshr/realtime-arabic-voice-agent.git
cd realtime-arabic-voice-agent
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
pip install livekit-agents livekit-plugins-openai livekit-plugins-noise-cancellation livekit-plugins-silero 
pip install chromadb sentence-transformers
pip install requests python-dotenv
pip install "numpy<2"
```

4. **Download ChromaDB data**
```bash
# Download from HuggingFace
git lfs install
git clone https://huggingface.co/datasets/azeddinShr/arabica-qa-chromadb
mv arabica-qa-chromadb/chroma_db ./
```

5. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Create `.env` file:
```bash
# LiveKit Configuration (local development)
LIVEKIT_URL=ws://Your-livekit-url-from-ur-account
LIVEKIT_API_KEY=ur-key
LIVEKIT_API_SECRET=ur-secret

# API Keys
OPENAI_API_KEY=sk-...
WEATHERAPI_KEY=...
TAVILY_API_KEY=tvly-...
```

## Usage

### Run Agent
```bash
python agent.py console
```

## Project Structure
```
arabic-voice-agent/
├── agent.py              # Main agent logic
├── tools.py              # RAG, Weather, and Web Search tools
├── chroma_db/            # Vector database (not in git)
├── venv/                 # Virtual environment (not in git)
├── .env                  # Environment variables (not in git)
├── .env.example          # Example environment file
├── .gitignore
├── README.md
└── requirements.txt
```

## Technical Details

### Latency
- Traditional pipeline: ~1050ms (STT + LLM + TTS)
- Unified model: ~350-450ms (60% reduction)

### Turn Detection
- Type: `semantic_vad`
- Eagerness: `low` (waits for complete thoughts)
- No interruption of AI responses

### Vector Search
- Method: Dense retrieval (cosine similarity)
- Chunk size: Original ArabicaQA context paragraphs
- Storage: Persistent ChromaDB


## License

MIT
