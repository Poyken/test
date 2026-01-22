# AI Flow - AI-Driven Development Workflow

Hệ thống tự động hóa phát triển phần mềm từ meeting notes đến product demo.

## 🌟 Features

- **Transcriber Agent**: Chuyển đổi audio/video meetings thành text
- **PM Agent**: Trích xuất requirements và user stories
- **Architect Agent**: Thiết kế kiến trúc kỹ thuật
- **Task Agent**: Chia nhỏ features thành atomic tasks
- **Code Agent**: Generate production-ready code
- **Review Agent**: Tự động review code
- **QA Agent**: Generate unit tests

## 📋 Prerequisites

- Python 3.11+
- Google API Key (free tier) - [Get it here](https://makersuite.google.com/app/apikey)

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd ai-flow
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Run the demo

```bash
# With the e-commerce project docs
python main.py demo

# Or with your own input
python main.py path/to/meeting_notes.txt -o ./output
```

## 📖 Usage

### From Text File

```bash
python main.py meeting_notes.txt -o ./generated
```

### From Audio/Video Recording

```bash
python main.py meeting_recording.mp4 -o ./generated
```

### With Project Documentation

```bash
python main.py requirements.txt -d ./docs -o ./generated
```

### With Existing Codebase (for RAG context)

```bash
python main.py notes.txt -c ./existing-project/src -o ./generated
```

## 🏗️ Project Structure

```
ai-flow/
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── config/
│   └── config.yaml        # Configuration
├── agents/
│   ├── base_agent.py      # Base agent class
│   ├── transcriber_agent.py
│   ├── pm_agent.py
│   ├── architect_agent.py
│   ├── task_agent.py
│   ├── code_agent.py
│   ├── review_agent.py
│   └── qa_agent.py
├── orchestrator/
│   ├── state.py           # Shared state definitions
│   └── workflow.py        # LangGraph workflow
└── rag/
    └── indexer.py         # RAG system for codebase context
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

- LLM provider (Gemini or Ollama)
- Model settings per agent
- Tech stack constraints
- Workflow settings

## 🔧 Tech Stack

- **LLM**: Google Gemini 2.0 Flash (Free Tier)
- **Orchestration**: LangGraph
- **RAG**: Gemini Embeddings + Vector Search
- **Target Stack**: NestJS + Next.js + PostgreSQL

## 📊 Workflow Pipeline

```
Meeting Notes/Audio
       ↓
[Transcriber Agent] → Text
       ↓
[PM Agent] → User Stories + Requirements
       ↓
[Architect Agent] → Technical Specs
       ↓
[Task Agent] → Atomic Tasks
       ↓
┌──────────────────────────────────┐
│  For each task:                  │
│    [Code Agent] → Generate Code  │
│    [Review Agent] → Review       │
│    [QA Agent] → Generate Tests   │
└──────────────────────────────────┘
       ↓
Generated Project Files
```

## 🆓 Free Tier Limits

Gemini API Free Tier:

- 15 requests per minute
- 1,000,000 tokens per day
- 1,500 requests per day

This is sufficient for most development workflows.

## 📝 License

MIT
