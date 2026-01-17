# SlugSyllabus (folder-only demo)

A pure-demo web app that stores uploaded syllabus PDFs in a folder and keeps metadata in a single JSON file (`index.json`).
Insights are generated on-demand via prompt keys (LLM stubbed) and optionally cached as text files in `cache/`.

## Prereqs
- Python 3.10+

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # mac/linux
pip install -r requirements.txt
```

## Run
```bash
uvicorn app.main:app --reload
```

Open:
- http://127.0.0.1:8000/

## How it works
- PDFs are stored in `uploads/`
- Metadata lives in `index.json` (auto-created)
- Insights tabs call `/insight/{slug}/{prompt_key}` which:
  - checks `cache/{slug}__{prompt_key}.txt`
  - if missing, calls the LLM stub and writes the cache file

## Where to plug in a real LLM
Edit `app/llm.py`:
- Replace `run_llm(prompt_key, syllabus_text)` with your provider call.
- For a demo, you can skip PDF->text and just send a short extracted snippet.

## Suggested demo script
1) Upload 2 PDFs
2) Open a syllabus detail page
3) Click TLDR / WORKLOAD / GRADING / PREREQS tabs
4) Refresh the page and click again to show caching is instant

## Gemini setup
Set your key:

```bash
export GEMINI_API_KEY="YOUR_KEY"
```

Install deps:

```bash
pip install -r requirements.txt
```
