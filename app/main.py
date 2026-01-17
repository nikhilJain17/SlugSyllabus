from __future__ import annotations

import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import markdown
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pypdf import PdfReader

from .llm import PROMPT_SPECS, run_llm

APP_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = APP_DIR / "uploads"
CACHE_DIR = APP_DIR / "cache"
INDEX_PATH = APP_DIR.parent / "index.json"
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"

app = FastAPI()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# -------------------------
# Index helpers (folder DB)
# -------------------------

def _load_index() -> dict:
    if not INDEX_PATH.exists():
        return {"syllabi": []}
    return json.loads(INDEX_PATH.read_text())


def _save_index(idx: dict) -> None:
    INDEX_PATH.write_text(json.dumps(idx, indent=2))


def _slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def _unique_slug(base: str, existing: set[str]) -> str:
    slug = base
    i = 2
    while slug in existing:
        slug = f"{base}-{i}"
        i += 1
    return slug


def _find_meta(slug: str) -> Optional[dict[str, Any]]:
    idx = _load_index()
    for s in idx.get("syllabi", []):
        if s.get("slug") == slug:
            return s
    return None


def _prune_missing_files() -> None:
    idx = _load_index()
    kept = []
    changed = False

    for s in idx.get("syllabi", []):
        filename = s.get("filename")
        if not filename:
            changed = True
            continue

        if (UPLOADS_DIR / filename).exists():
            kept.append(s)
        else:
            changed = True

    if changed:
        idx["syllabi"] = kept
        _save_index(idx)


# -------------------------
# PDF helpers
# -------------------------

def _extract_pdf_text(pdf_path: Path, max_chars: int = 45_000) -> str:
    """Best-effort PDF -> text. Truncates to keep LLM calls small."""
    try:
        reader = PdfReader(str(pdf_path))
        parts: list[str] = []
        total = 0
        for page in reader.pages:
            t = page.extract_text() or ""
            if t:
                parts.append(t)
                total += len(t)
            if total >= max_chars:
                break
        return "\n\n".join(parts)[:max_chars]
    except Exception:
        return ""


def _cache_file(slug: str, prompt_key: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", prompt_key)
    return CACHE_DIR / f"{slug}__{safe}.txt"


# -------------------------
# Precompute (upload-time)
# -------------------------

def _precompute_insights(slug: str, filename: str) -> None:
    """
    Run all prompt queries once at upload time and write cache files.
    Extracts PDF text once. Uses a small thread pool to parallelize LLM calls.
    """
    pdf_path = UPLOADS_DIR / filename
    text = _extract_pdf_text(pdf_path)

    prompt_keys = list(PROMPT_SPECS.keys())

    if not text.strip():
        msg = "No text could be extracted from this PDF (scanned image?)."
        for k in prompt_keys:
            _cache_file(slug, k).write_text(msg)
        return

    def run_one(k: str) -> tuple[str, str]:
        out = run_llm(k, text)
        return k, out

    max_workers = max(1, min(4, len(prompt_keys)))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_one, k) for k in prompt_keys]
        for f in as_completed(futures):
            k, out = f.result()
            _cache_file(slug, k).write_text(out)


# -------------------------
# App lifecycle
# -------------------------

@app.on_event("startup")
def _startup() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_PATH.exists():
        _save_index({"syllabi": []})
    _prune_missing_files()


# -------------------------
# Routes
# -------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    _prune_missing_files()
    idx = _load_index()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "syllabi": idx.get("syllabi", [])},
    )


@app.get("/upload", response_class=HTMLResponse)
def upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/upload")
def upload_syllabus(
    background_tasks: BackgroundTasks,
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
    existing = {s["slug"] for s in idx.get("syllabi", [])}

    base_slug = _slugify(f"{course_code}-{instructor}-{quarter}-{year}")
    slug = _unique_slug(base_slug, existing)

    filename = f"{slug}.pdf"
    pdf_path = UPLOADS_DIR / filename
    with pdf_path.open("wb") as f:
        shutil.copyfileobj(pdf.file, f)

    idx["syllabi"].append(
        {
            "slug": slug,
            "course_code": course_code,
            "title": title,
            "instructor": instructor,
            "quarter": quarter,
            "year": year,
            "filename": filename,
            "uploaded_at": datetime.utcnow().isoformat(),
        }
    )
    _save_index(idx)

    # Kick off precompute in the background (fast upload -> instant tabs later)
    background_tasks.add_task(_precompute_insights, slug, filename)

    return RedirectResponse(url=f"/syllabus/{slug}", status_code=303)


@app.get("/syllabus/{slug}", response_class=HTMLResponse)
def syllabus_page(request: Request, slug: str):
    meta = _find_meta(slug)
    if not meta:
        raise HTTPException(status_code=404)
    return templates.TemplateResponse(
        "syllabus.html",
        {
            "request": request,
            "s": meta,
            "prompt_keys": list(PROMPT_SPECS.keys()),
        },
    )


@app.get("/pdf/{slug}")
def syllabus_pdf(slug: str):
    meta = _find_meta(slug)
    if not meta:
        raise HTTPException(status_code=404)

    path = UPLOADS_DIR / meta["filename"]
    if not path.exists():
        raise HTTPException(status_code=404)

    return FileResponse(
        str(path),
        media_type="application/pdf",
        filename=path.name,
        headers={"Content-Disposition": f'inline; filename="{path.name}"'},
    )

@app.get("/compare/{slug_a}/{slug_b}", response_class=HTMLResponse)
def compare(slug_a: str, slug_b: str):
    a = _find_meta(slug_a)
    b = _find_meta(slug_b)
    if not a or not b:
        html = f"""
        <div class="llm-card">
        <div class="llm-content">
            {markdown.markdown(result, extensions=["fenced_code", "tables"])}
        </div>
        </div>
        """
        return HTMLResponse(html)
    def load_cached(slug: str, key: str) -> str:
        p = _cache_file(slug, key)
        return p.read_text() if p.exists() else "(not generated)"

    a_text = f"""
TLDR:
{load_cached(slug_a, "tldr")}

WORKLOAD:
{load_cached(slug_a, "workload")}

GRADING:
{load_cached(slug_a, "grading")}
"""

    b_text = f"""
TLDR:
{load_cached(slug_b, "tldr")}

WORKLOAD:
{load_cached(slug_b, "workload")}

GRADING:
{load_cached(slug_b, "grading")}
"""

    prompt = f"""
Compare these two classes for a student choosing one.

Discuss:
- workload intensity
- grading style
- who should take each

End with a clear recommendation.
Format your response using markdown.
Use clear section headers and bullet points.
Prefer concise comparisons over long paragraphs.

Class A ({a["course_code"]}):
{a_text}

Class B ({b["course_code"]}):
{b_text}
"""

    result = run_llm("compare", prompt)

    return HTMLResponse(
        markdown.markdown(result, extensions=["fenced_code", "tables"])
    )


@app.on_event("startup")
def _debug_routes():
    print("=== REGISTERED ROUTES ===")
    for r in app.routes:
        print(r.path)

@app.get("/insight/{slug}/{prompt_key}", response_class=HTMLResponse)
def insight_partial(slug: str, prompt_key: str):
    if prompt_key not in PROMPT_SPECS:
        raise HTTPException(status_code=400)

    meta = _find_meta(slug)
    if not meta:
        raise HTTPException(status_code=404)

    cache_path = _cache_file(slug, prompt_key)
    if cache_path.exists():
        text = cache_path.read_text()
        source = "cache"
    else:
        # Precompute might still be running; fall back to on-demand generation
        pdf_text = _extract_pdf_text(UPLOADS_DIR / meta["filename"])
        text = run_llm(prompt_key, pdf_text)
        cache_path.write_text(text)
        source = "generated"

    body_html: str

    if prompt_key == "tldr":
        body_html = markdown.markdown(text, extensions=["fenced_code", "tables"])

    elif prompt_key == "prereqs":
        try:
            parsed = json.loads(text)
            body_html = _render_prereqs(parsed)
        except Exception:
            body_html = f"<pre class='mono'>{_escape(text)}</pre>"

    elif prompt_key == "workload":
        try:
            parsed = json.loads(text)
            body_html = _render_workload(parsed)
        except Exception:
            body_html = f"<pre class='mono'>{_escape(text)}</pre>"

    elif prompt_key == "grading":
        try:
            parsed = json.loads(text)
            body_html = _render_grading(parsed)
        except Exception:
            body_html = f"<pre class='mono'>{_escape(text)}</pre>"

    else:
        body_html = f"<pre class='mono'>{_escape(text)}</pre>"

    html = f"""
    <div class="panel">
      <div class="panel-h">
        <div>
          <div class="panel-title">{prompt_key.upper()}</div>
          <div class="panel-sub">Source: {source}</div>
        </div>
      </div>
      <div class="prose prose-invert max-w-none">
        {body_html}
      </div>
    </div>
    """
    return HTMLResponse(content=html)


@app.post("/cache/clear/{slug}")
def clear_cache(slug: str):
    for p in CACHE_DIR.glob(f"{slug}__*.txt"):
        try:
            p.unlink()
        except Exception:
            pass
    return RedirectResponse(url=f"/syllabus/{slug}", status_code=303)


# -------------------------
# Render helpers
# -------------------------

def _render_prereqs(p: dict) -> str:
    def ul(items):
        if not items:
            return "<div class='text-slate-400'>Not specified</div>"
        return "<ul class='list-disc pl-5'>" + "".join(
            f"<li>{_escape(str(x))}</li>" for x in items
        ) + "</ul>"

    quotes = p.get("evidence_quotes", []) or []
    quotes_html = (
        "<div class='text-slate-400'>No quotes found</div>"
        if not quotes
        else "<ul class='list-disc pl-5 text-slate-400'>"
        + "".join(f"<li>“{_escape(str(q))}”</li>" for q in quotes)
        + "</ul>"
    )

    return f"""
    <div class="grid gap-4">
      <div>
        <h3 class="font-semibold">Official prerequisites</h3>
        {ul(p.get("official_prereqs", []) or [])}
      </div>

      <div>
        <h3 class="font-semibold">Implied background</h3>
        {ul(p.get("implied_background", []) or [])}
      </div>

      <div>
        <h3 class="font-semibold">Tools &amp; languages</h3>
        {ul(p.get("tools_languages", []) or [])}
      </div>

      <div>
        <h3 class="font-semibold">Math background</h3>
        {ul(p.get("math_background", []) or [])}
      </div>

      <div>
        <h3 class="font-semibold">Evidence from syllabus</h3>
        {quotes_html}
      </div>
    </div>
    """


def _render_workload(p: dict) -> str:
    hours = p.get("hours_per_week_estimate")
    shape = p.get("workload_shape")
    heavy_weeks = p.get("heavy_weeks", []) or []
    why = p.get("why_heavy", "")
    quotes = p.get("evidence_quotes", []) or []

    def pill(text: str) -> str:
        return (
            "<span class='inline-block rounded-full bg-slate-800 px-3 py-1 "
            "text-xs text-slate-200'>"
            f"{_escape(text)}</span>"
        )

    heavy_html = (
        "<div class='text-slate-400'>No specific heavy weeks identified</div>"
        if not heavy_weeks
        else "<div class='flex flex-wrap gap-2'>" + "".join(pill(str(w)) for w in heavy_weeks) + "</div>"
    )

    quotes_html = (
        "<div class='text-slate-400'>No direct quotes found</div>"
        if not quotes
        else "<ul class='list-disc pl-5 text-slate-400'>"
        + "".join(f"<li>“{_escape(str(q))}”</li>" for q in quotes)
        + "</ul>"
    )

    hours_str = "Not explicitly specified" if hours is None else f"{hours} hours/week"
    shape_str = "Unknown / uneven" if not shape else str(shape)
    why_str = "Not specified" if not why else str(why)

    return f"""
    <div class="grid gap-4">
      <div class="grid gap-2">
        <h3 class="font-semibold">Estimated workload</h3>
        <div class="text-slate-300">{_escape(hours_str)}</div>
      </div>

      <div class="grid gap-2">
        <h3 class="font-semibold">Workload pattern</h3>
        <div class="text-slate-300">{_escape(shape_str)}</div>
      </div>

      <div class="grid gap-2">
        <h3 class="font-semibold">Heavy weeks</h3>
        {heavy_html}
      </div>

      <div class="grid gap-2">
        <h3 class="font-semibold">Why those weeks are heavy</h3>
        <div class="text-slate-300">{_escape(why_str)}</div>
      </div>

      <div class="grid gap-2">
        <h3 class="font-semibold">Evidence from syllabus</h3>
        {quotes_html}
      </div>
    </div>
    """


def _render_grading(p: dict) -> str:
    comps = p.get("grading_components", []) or []
    dels = p.get("deliverables", []) or []
    late = (p.get("late_policy") or "").strip()
    collab = (p.get("collaboration_policy") or "").strip()
    quotes = p.get("evidence_quotes", []) or []

    if comps:
        rows = []
        for c in comps:
            name = _escape(str(c.get("name", "")).strip() or "Unknown")
            w = c.get("weight_percent", None)
            if w is None or str(w).strip() == "":
                weight = "?"
            else:
                ws = str(w).strip()
                weight = _escape(ws if ws.endswith("%") else ws + "%")
            rows.append((name, weight))

        grading_table = f"""
        <div class="overflow-hidden rounded-xl border border-slate-800 bg-slate-950">
          <table class="w-full text-sm">
            <thead class="border-b border-slate-800 text-slate-300">
              <tr>
                <th class="py-2 px-3 text-left font-semibold">Component</th>
                <th class="py-2 px-3 text-right font-semibold">Weight</th>
              </tr>
            </thead>
            <tbody class="text-slate-200">
              {''.join(f"<tr class='border-b border-slate-900'><td class='py-2 px-3'>{n}</td><td class='py-2 px-3 text-right'>{w}</td></tr>" for (n, w) in rows)}
            </tbody>
          </table>
        </div>
        """
    else:
        grading_table = "<div class='text-slate-400'>No grading breakdown found</div>"

    def deliverable_line(d: dict) -> str:
        typ = _escape(str(d.get("type", "other")))
        count = d.get("count", None)
        count_str = "" if count is None or str(count).strip() == "" else f" · { _escape(str(count)) }"
        notes = (d.get("notes") or "").strip()
        notes_str = "" if not notes else f" — <span class='text-slate-300'>{_escape(notes)}</span>"
        return f"<li><span class='font-medium text-slate-200'>{typ}{count_str}</span>{notes_str}</li>"

    deliverables_html = (
        "<div class='text-slate-400'>No deliverables found</div>"
        if not dels
        else "<ul class='list-disc pl-5'>" + "".join(deliverable_line(d) for d in dels) + "</ul>"
    )

    late_html = f"<div class='text-slate-300'>{_escape(late)}</div>" if late else "<div class='text-slate-400'>Not specified</div>"
    collab_html = f"<div class='text-slate-300'>{_escape(collab)}</div>" if collab else "<div class='text-slate-400'>Not specified</div>"

    quotes_html = (
        "<div class='text-slate-400'>No direct quotes found</div>"
        if not quotes
        else "<ul class='list-disc pl-5 text-slate-400'>"
        + "".join(f"<li>“{_escape(str(q))}”</li>" for q in quotes)
        + "</ul>"
    )

    return f"""
    <div class="grid gap-5">
      <div class="grid gap-2">
        <h3 class="font-semibold">Grading breakdown</h3>
        {grading_table}
      </div>

      <div class="grid gap-2">
        <h3 class="font-semibold">Deliverables</h3>
        {deliverables_html}
      </div>

      <div class="grid gap-2">
        <h3 class="font-semibold">Late policy</h3>
        {late_html}
      </div>

      <div class="grid gap-2">
        <h3 class="font-semibold">Collaboration policy</h3>
        {collab_html}
      </div>

      <div class="grid gap-2">
        <h3 class="font-semibold">Evidence from syllabus</h3>
        {quotes_html}
      </div>
    </div>
    """


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )