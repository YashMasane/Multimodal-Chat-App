# PolyForma ChatBot 🤖🎤
 
*A multimodal chatbot supporting text/audio input with conversation history and audio summarization*

## Table of Contents
- [Features](#features-)
- [Tech Stack](#tech-stack-)
- [Installation](#installation-)
- [Usage Guide](#usage-guide-)
- [Project Structure](#project-structure-)
- [Configuration](#configuration-)
- [License](#license-)
- [Acknowledgments](#acknowledgments-)
- [Contact](#contact-)

## Features ✨

### Input Modes
- **📝 Text Chat**: Natural language interactions
- **🎙️ Voice Input**: Real-time speech-to-text conversion
- **📁 Audio Upload**: Process pre-recorded audio files

### Core Capabilities
- **🗂️ Session Management**: 
  - Auto-saving conversation history
  - Session-specific context tracking
- **📑 Audio Summarization**:
  - Generate text summaries from audio files
  - Supports MP3/WAV formats (up to 25MB)
- **🧠 AI Integration**:
  - Mistral-7B for intelligent responses
  - Whisper AI for speech processing
  - BAAI embeddings for semantic analysis

## Tech Stack 🛠️

| Component              | Technology                          | Version/Model                  |
|------------------------|-------------------------------------|---------------------------------|
| Language Model         | Mistral                             | 7b-instruct-v0.2.Q5_K_M        |
| UI Framework           | Streamlit                           | 1.28+                          |
| Text Embeddings        | BAAI                                | bge-large-en-v1.5              |
| Speech-to-Text         | OpenAI Whisper                      | small                          |
| Core Libraries         | Hugging Face Transformers           | 4.30+                          |
| Audio Processing       | Librosa/PyAudio                     | 0.10+                          |

## Installation 💻

### Requirements
- Python 3.8+
- FFmpeg (`sudo apt install ffmpeg` for Linux)
- 8GB RAM (16GB recommended)

### Setup Guide

1. **Clone Repository**
   ```bash
   git clone https://github.com/your-username/PolyForma-ChatBot.git
   cd PolyForma-ChatBot

2. **Initialize Virtual Environment**
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    venv\Scripts\activate     # Windows

3. **Install Dependencies**
    pip install -r requirements.txt

4. **Model Setup**


5. **Launch Application**
    streamlit run app.py

## Advanced Features 🚀

### Session History 📚
- **View Conversations**: Access previous chats via the sidebar
- **Export Options**: Download history as:
  ```python
  📁 JSON (structured data format)
  📄 TXT (readable text format)

## Project Structure 📂

  PolyForma-ChatBot/
├── models/                   # Local model storage
│   └── mistral-7b...gguf    # Quantized LLM
│                 
├── audio_handler.py     # Voice processing
├── llm_processor.py     # Model interactions
├── session_manager.py   # History tracking
├── app.py                    # Main application
├── requirements.txt          # Dependency list

