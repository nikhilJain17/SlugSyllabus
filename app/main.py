from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .llm import PROMPTS, run_llm

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
UPLOADS_DIR = ROOT_DIR / "uploads"
CACHE_DIR = ROOT_DIR / "cache"
INDEX_PATH = ROOT_DIR / "index.json"
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"

app = FastAPI(title="SlugSyllabus (Folder Demo)")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@dataclass
class SyllabusMeta:
    slug: str
    filename: str
    course_code: str
    title: Optional[str] = None
    instructor: Optional[str] = None
    quarter: Optional[str] = None
    year: Optional[int] = None
    created_at: str = ""


def _load_index() -> dict[str, Any]:
    if not INDEX_PATH.exists():
        return {"syllabi": []}
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def _save_index(data: dict[str, Any]) -> None:
    INDEX_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"^-+|-+$", "", s)
    return s or "syllabus"


def _unique_slug(base: str, existing: set[str]) -> str:
    if base not in existing:
        return base
    i = 2
    while f"{base}-{i}" in existing:
        i += 1
    return f"{base}-{i}"


def _find_meta(slug: str) -> Optional[dict[str, Any]]:
    idx = _load_index()
    for s in idx.get("syllabi", []):
        if s.get("slug") == slug:
            return s
    return None


@app.on_event("startup")
def _startup() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_PATH.exists():
        _save_index({"syllabi": []})


@app.get("/", response_class=HTMLResponse)
def index(request: Request, q: str = ""):
    idx = _load_index()
    syllabi = idx.get("syllabi", [])

    q_str = (q or "").strip().lower()
    if q_str:
        def hit(s: dict[str, Any]) -> bool:
            hay = " ".join(
                str(s.get(k) or "")
                for k in ("course_code", "title", "instructor", "quarter", "year")
            ).lower()
            return q_str in hay
        syllabi = [s for s in syllabi if hit(s)]

    syllabi = sorted(syllabi, key=lambda s: s.get("created_at", ""), reverse=True)

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "syllabi": syllabi, "q": q},
    )


@app.get("/upload", response_class=HTMLResponse)
def upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/upload")
def upload_syllabus(
    course_code: str = Form(...),
    title: str = Form(""),
    instructor: str = Form(""),
    quarter: str = Form(""),
    year: int = Form(0),
    pdf: UploadFile = File(...),
):
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload must be a PDF")

    idx = _load_index()
    existing_slugs = {s.get("slug") for s in idx.get("syllabi", [])}

    # slug based on course + instructor + quarter + year (best-effort)
    base = "_".join(
        x for x in [course_code, instructor, quarter, str(year) if year else ""] if x
    )
    slug = _unique_slug(_slugify(base), existing_slugs)

    safe_name = pdf.filename.replace("/", "_").replace("\\", "_")
    stored_name = f"{slug}__{safe_name}"
    dest = UPLOADS_DIR / stored_name

    with dest.open("wb") as f:
        shutil.copyfileobj(pdf.file, f)

    meta = {
        "slug": slug,
        "filename": stored_name,
        "course_code": course_code.strip(),
        "title": (title.strip() or None),
        "instructor": (instructor.strip() or None),
        "quarter": (quarter.strip() or None),
        "year": (year if year > 0 else None),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    idx.setdefault("syllabi", []).append(meta)
    _save_index(idx)

    return RedirectResponse(url=f"/s/{slug}", status_code=303)


@app.get("/s/{slug}", response_class=HTMLResponse)
def syllabus_page(request: Request, slug: str):
    meta = _find_meta(slug)
    if not meta:
        raise HTTPException(status_code=404, detail="Not found")

    return templates.TemplateResponse(
        "syllabus.html",
        {
            "request": request,
            "s": meta,
            "prompt_keys": list(PROMPTS.keys()),
        },
    )


@app.get("/pdf/{slug}")
def syllabus_pdf(slug: str):
    meta = _find_meta(slug)
    if not meta:
        raise HTTPException(status_code=404, detail="Not found")

    path = UPLOADS_DIR / meta["filename"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="PDF missing on disk")

    return FileResponse(str(path), media_type="application/pdf", filename=path.name)


@app.get("/insight/{slug}/{prompt_key}", response_class=HTMLResponse)
def insight_partial(slug: str, prompt_key: str):
    if prompt_key not in PROMPTS:
        raise HTTPException(status_code=400, detail="Unknown prompt_key")

    meta = _find_meta(slug)
    if not meta:
        raise HTTPException(status_code=404, detail="Not found")

    cache_path = CACHE_DIR / f"{slug}__{prompt_key}.txt"

    if cache_path.exists():
        text = cache_path.read_text(encoding="utf-8")
        source = "cache"
    else:
        # For demo: we aren't extracting PDF text.
        syllabus_text = ""
        text = run_llm(prompt_key, syllabus_text)
        cache_path.write_text(text, encoding="utf-8")
        source = "generated"

    html = f"""
    <div class=\"rounded-xl border border-slate-800 bg-slate-950 p-4\">
      <div class=\"flex items-start justify-between gap-3\">
        <div>
          <div class=\"text-sm font-semibold tracking-tight\">{prompt_key.upper()}</div>
          <div class=\"mt-1 text-xs text-slate-400\">Source: {source}</div>
        </div>
        <span class=\"rounded-full border border-slate-800 bg-slate-900 px-2 py-1 text-xs text-slate-300\">LLM</span>
      </div>
      <pre class=\"mono mt-3\">{_escape(text)}</pre>
    </div>
    """
    return HTMLResponse(content=html)


@app.post("/cache/clear/{slug}")
def clear_cache(slug: str):
    # delete all cache entries for this syllabus
    for p in CACHE_DIR.glob(f"{slug}__*.txt"):
        try:
            p.unlink()
        except Exception:
            pass
    return RedirectResponse(url=f"/s/{slug}", status_code=303)


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
