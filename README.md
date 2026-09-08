# Mind Power Artists — AI Knowledge Assistant

A RAG-based (Retrieval-Augmented Generation) chatbot for Mind Power Artists that answers questions about services, courses, booking, payments, and more using a curated knowledge base.

## Architecture

```
User Question
    │
    ▼
┌─────────────┐     ┌──────────────────┐
│  TF-IDF     │────▶│  Top-K Relevant  │
│  Retrieval  │     │  Knowledge Chunks │
└─────────────┘     └────────┬─────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  LLM Generation │
                    │  (Groq/Gemini)  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Grounded Answer│
                    │  + Source Tags  │
                    └─────────────────┘
```

## Features

- **RAG Pipeline**: TF-IDF vectorization + cosine similarity retrieval, LLM-grounded generation
- **Admin Panel**: Add new knowledge entries without touching code
- **Unanswered Question Log**: Tracks questions with no match for continuous improvement
- **Dual LLM Support**: Works with Groq (free) or Google Gemini (free tier)
- **Fallback Mode**: Works without API key using direct knowledge base lookup
- **35+ pre-loaded entries** covering all Mind Power Artists services, courses, FAQs, and policies

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Get a Free API Key

- **Groq** (recommended): https://console.groq.com — sign up, create API key
- **Gemini**: https://aistudio.google.com — sign up, create API key

## Deployment

### Streamlit Community Cloud (free)
1. Push this folder to a GitHub repo
2. Go to share.streamlit.io
3. Connect repo, set `app.py` as main file
4. Add API key in Secrets: `GROQ_API_KEY = "your-key"`

### Local
```bash
streamlit run app.py --server.port 8501
```

## Tech Stack

- **Frontend**: Streamlit
- **Retrieval**: TF-IDF + Cosine Similarity (pure Python, zero ML dependencies)
- **Generation**: Groq API (Llama 3.1) or Google Gemini
- **Knowledge Base**: JSON file (easily editable)
