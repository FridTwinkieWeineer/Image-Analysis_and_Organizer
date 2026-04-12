# Image Analysis & Organizer

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Prerequisites
- Python 3.10+
- Ollama running locally with `llama3.2-vision` model pulled
- `ollama pull llama3.2-vision`

## Architecture
- `app.py` — Streamlit UI entry point
- `lib/` — Core logic modules (state, image_utils, ollama_client, clustering, organizer)
- `project.json` — Auto-generated state file (gitignored)
