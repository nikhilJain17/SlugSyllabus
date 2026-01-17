from __future__ import annotations

import os
import json
from google import genai

API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY)")

client = genai.Client(api_key=API_KEY)

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")  # good default  [oai_citation:3‡Google AI for Developers](https://ai.google.dev/gemini-api/docs/models?utm_source=chatgpt.com)

PROMPT_SPECS: dict[str, dict] = {
    "tldr": {
        "mode": "text",
        "prompt": "Summarize this syllabus in 6 concise bullet points."
    },
    "workload": {
        "mode": "json",
        "prompt": "Estimate workload and identify heavy weeks from the syllabus.",
        "schema": {
            "hours_per_week_estimate": None,
            "workload_shape": None,
            "heavy_weeks": [],
            "why_heavy": "",
            "evidence_quotes": []
        }
    },
    "grading": {
        "mode": "json",
        "prompt": "Extract grading breakdown and major deliverables from the syllabus.",
        "schema": {
            "grading_components": [],
            "deliverables": [],
            "late_policy": "",
            "collaboration_policy": "",
            "evidence_quotes": []
        }
    },
    "prereqs": {
        "mode": "json",
        "prompt": "Infer implied prerequisites and recommended background from the syllabus text.",
        "schema": {
            "official_prereqs": [],
            "implied_background": [],
            "tools_languages": [],
            "math_background": [],
            "evidence_quotes": []
        }
    },
}

def run_llm(prompt_key: str, syllabus_text: str) -> str:
    spec = PROMPT_SPECS[prompt_key]
    if not syllabus_text.strip():
        return "No text could be extracted from this PDF (or it’s scanned)."

    if spec["mode"] == "text":
        prompt = _wrap_text(spec["prompt"], syllabus_text)
        resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)  #  [oai_citation:4‡Google AI for Developers](https://ai.google.dev/api/generate-content?utm_source=chatgpt.com)
        return (resp.text or "").strip()

    schema_json = json.dumps(spec["schema"], indent=2)
    prompt = _wrap_json(spec["prompt"], schema_json, syllabus_text)
    resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)  #  [oai_citation:5‡Google AI for Developers](https://ai.google.dev/api/generate-content?utm_source=chatgpt.com)
    out = (resp.text or "").strip()

    parsed = _extract_json(out)
    return json.dumps(parsed, indent=2, ensure_ascii=False) if parsed is not None else out

def _wrap_text(task: str, text: str) -> str:
    return f"""You analyze university course syllabi.

TASK:
{task}

SYLLABUS TEXT:
{text}
"""

def _wrap_json(task: str, schema_json: str, text: str) -> str:
    return f"""You are a careful information extraction system for university syllabi.

TASK:
{task}

OUTPUT:
Return ONLY valid JSON matching this template. Use null/empty arrays when unknown. Do not invent facts.
JSON TEMPLATE:
{schema_json}

RULES:
- evidence_quotes: short verbatim snippets (<=25 words) from the syllabus when possible.
- If the syllabus doesn't say it, leave it null/[]/"".

SYLLABUS TEXT:
{text}
"""

def _extract_json(s: str):
    s2 = s.strip()
    if s2.startswith("```"):
        parts = s2.split("```")
        if len(parts) >= 3:
            s2 = parts[1].strip()
    start = s2.find("{")
    end = s2.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(s2[start:end+1])
    except Exception:
        return None