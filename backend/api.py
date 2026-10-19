"""
api.py - FastAPI backend with Swagger UI.

Run from backend/:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any, Optional

import requests
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ragcore import config, rag, retrieval, store
from ragcore.ingest import ingest

CONSOLE_HTML = """<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>RAG API Console</title>
<style>
:root { --bg:#0f1115; --card:#1a1d24; --fg:#e8eaed; --mut:#9aa0a6; --ac:#3b82f6; --ok:#22c55e; --bad:#ef4444; --bd:#2a2f3a; }
* { box-sizing: border-box; }
body { margin:0; font:15px/1.45 system-ui,sans-serif; background:var(--bg); color:var(--fg); }
main { max-width:640px; margin:0 auto; padding:24px 16px; }
h1 { font-size:1.25rem; margin:0 0 8px; }
a { color:var(--ac); }
.card { background:var(--card); border:1px solid var(--bd); border-radius:12px; padding:16px; margin:14px 0; }
.mut { color:var(--mut); font-size:13px; }
.row { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:12px; }
button { font:inherit; cursor:pointer; border-radius:8px; border:1px solid var(--bd); background:#252a33; color:var(--fg); padding:10px 14px; }
button.p { background:var(--ac); border-color:var(--ac); color:#fff; font-weight:600; }
label.row { gap:6px; margin:0; }
pre { background:#111; border:1px solid var(--bd); border-radius:8px; padding:12px; overflow:auto; font-size:12px; white-space:pre-wrap; margin:10px 0 0; max-height:280px; }
.bad { color:var(--bad); }
</style>
</head>
<body>
<main>
  <h1>RAG API Console</h1>
  <p class=\"mut\">Buttons call this same server. Full OpenAPI: <a href=\"/docs\">/docs</a></p>
  <div class=\"card\">
    <b>Quick checks</b>
    <div class=\"row\">
      <button type=\"button\" id=\"btn-health\">GET /health</button>
      <button type=\"button\" id=\"btn-status\">GET /status</button>
      <button type=\"button\" id=\"btn-llms\">GET /llms</button>
    </div>
  </div>
  <div class=\"card\">
    <b>Temporary reindex job</b>
    <p class=\"mut\">POST /jobs/reindex — starts GitHub Actions (needs GITHUB_TOKEN on the server).</p>
    <label class=\"row mut\"><input type=\"checkbox\" id=\"evals\"> Also run evals</label>
    <div class=\"row\">
      <button type=\"button\" class=\"p\" id=\"btn-reindex\">Start reindex job</button>
    </div>
    <p class=\"mut\" id=\"run-link\"></p>
  </div>
  <div class=\"card\">
    <b>Response</b>
    <pre id=\"out\">Click a button…</pre>
  </div>
</main>
<script>
const out = document.getElementById('out');
const runLink = document.getElementById('run-link');
function show(x, err) {
  out.textContent = typeof x === 'string' ? x : JSON.stringify(x, null, 2);
  out.className = err ? 'bad' : '';
}
async function call(path, opts={}) {
  show('Loading ' + path + '…');
  runLink.innerHTML = '';
  try {
    const r = await fetch(path, {
      ...opts,
      headers: { ...(opts.body ? {'Content-Type':'application/json'} : {}), ...opts.headers },
    });
    const t = await r.text();
    let j; try { j = JSON.parse(t); } catch { j = t; }
    if (!r.ok) throw new Error(typeof j === 'object' ? (j.detail || JSON.stringify(j)) : t);
    show(j);
    if (j && j.html_url) {
      runLink.innerHTML = 'Open run: <a href=\"' + j.html_url + '\" target=\"_blank\" rel=\"noopener\">' + j.html_url + '</a>';
    }
  } catch (e) {
    show(String(e.message || e), true);
  }
}
document.getElementById('btn-health').onclick = () => call('/health');
document.getElementById('btn-status').onclick = () => call('/status');
document.getElementById('btn-llms').onclick = () => call('/llms');
document.getElementById('btn-reindex').onclick = () => {
  const run_evals = document.getElementById('evals').checked;
  call('/jobs/reindex', { method: 'POST', body: JSON.stringify({ run_evals }) });
};
</script>
</body>
</html>
"""

app = FastAPI(
    title="RAG Project API",
    description=(
        "Modular Retrieval-Augmented Generation backend. "
        "JSON + NumPy source of truth, Chroma index, hybrid/BM25/rerank, "
        "pluggable embeddings + LLM gateways, per-stage tracing. "
        "POST /jobs/reindex starts a temporary GitHub Actions runner. "
        "Open **/console** for one-click buttons."
    ),
    version="1.4.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GH_API = "https://api.github.com"
WORKFLOW_FILE = "reindex.yml"


def _gh_env():
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or ""
    owner = os.getenv("GITHUB_OWNER", "allanvarghese567-lab")
    repo = os.getenv("GITHUB_REPO", "rag-project")
    branch = os.getenv("GITHUB_BRANCH", "main")
    return token, owner, repo, branch


def _gh_headers(token: str) -> dict:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _dispatch_reindex(run_evals: bool) -> dict[str, Any]:
    token, owner, repo, branch = _gh_env()
    if not token:
        raise HTTPException(
            503,
            "GITHUB_TOKEN (or GH_TOKEN) is not set on the server. "
            "Add a fine-grained PAT with Actions: write and Contents: read.",
        )
    headers = _gh_headers(token)
    dispatch_url = (
        f"{GH_API}/repos/{owner}/{repo}/actions/workflows/{WORKFLOW_FILE}/dispatches"
    )
    body = {"ref": branch, "inputs": {"run_evals": str(bool(run_evals)).lower()}}
    r = requests.post(dispatch_url, headers=headers, json=body, timeout=30)
    if r.status_code not in (204, 200):
        detail = r.text
        try:
            detail = r.json().get("message", detail)
        except Exception:
            pass
        raise HTTPException(
            r.status_code if r.status_code >= 400 else 502,
            f"workflow_dispatch failed: {detail}",
        )
    runs_url = (
        f"{GH_API}/repos/{owner}/{repo}/actions/workflows/{WORKFLOW_FILE}/runs"
        f"?per_page=5&event=workflow_dispatch"
    )
    run = None
    for _ in range(8):
        time.sleep(1.2)
        rr = requests.get(runs_url, headers=headers, timeout=30)
        if not rr.ok:
            continue
        runs = rr.json().get("workflow_runs") or []
        if not runs:
            continue
        for candidate in runs:
            if candidate.get("status") in ("queued", "waiting", "in_progress", "requested"):
                run = candidate
                break
        if run is None:
            run = runs[0]
        if run:
            break
    actions_page = f"https://github.com/{owner}/{repo}/actions/workflows/{WORKFLOW_FILE}"
    if not run:
        return {
            "ok": True,
            "dispatched": True,
            "run_id": None,
            "status": "unknown",
            "html_url": actions_page,
            "message": "Dispatch accepted; open the workflow page for the latest run.",
            "run_evals": run_evals,
        }
    return {
        "ok": True,
        "dispatched": True,
        "run_id": run.get("id"),
        "run_number": run.get("run_number"),
        "status": run.get("status"),
        "conclusion": run.get("conclusion"),
        "html_url": run.get("html_url") or actions_page,
        "jobs_url": run.get("jobs_url"),
        "created_at": run.get("created_at"),
        "run_evals": run_evals,
    }


def _get_run(run_id: int) -> dict[str, Any]:
    token, owner, repo, _ = _gh_env()
    if not token:
        raise HTTPException(503, "GITHUB_TOKEN is not set on the server.")
    url = f"{GH_API}/repos/{owner}/{repo}/actions/runs/{run_id}"
    r = requests.get(url, headers=_gh_headers(token), timeout=30)
    if r.status_code == 404:
        raise HTTPException(404, f"run {run_id} not found")
    if not r.ok:
        raise HTTPException(r.status_code, r.text)
    run = r.json()
    return {
        "run_id": run.get("id"),
        "run_number": run.get("run_number"),
        "status": run.get("status"),
        "conclusion": run.get("conclusion"),
        "html_url": run.get("html_url"),
        "jobs_url": run.get("jobs_url"),
        "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"),
        "display_title": run.get("display_title") or run.get("name"),
    }


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What is the refund window?"])
    backend: str = Field("chroma", description="numpy | chroma | hybrid | +rerank")
    k: int = Field(config.TOP_K, ge=1, le=50)
    llm_profile: Optional[str] = Field(None, description="Optional LLM profile id")


class AskResponse(BaseModel):
    answer: str
    hits: list[dict[str, Any]]
    backend: str
    k: int
    llm_profile: Optional[str] = None


class RetrieveResponse(BaseModel):
    hits: list[dict[str, Any]]
    backend: str
    k: int


class IngestResponse(BaseModel):
    file: str
    chunks: int


class StatusResponse(BaseModel):
    chunks: int
    sources: list[str]
    embedding_profile: dict[str, Any]
    llm_profile: dict[str, Any]
    store_dir: str


class EvalRequest(BaseModel):
    label: str = Field("api-run")
    backend: str = "chroma"
    k: int = Field(config.TOP_K, ge=1, le=50)


class ReindexRequest(BaseModel):
    run_evals: bool = Field(False, description="Run evals after indexing")


class ReindexResponse(BaseModel):
    ok: bool
    dispatched: bool
    run_id: Optional[int] = None
    run_number: Optional[int] = None
    status: Optional[str] = None
    conclusion: Optional[str] = None
    html_url: str
    jobs_url: Optional[str] = None
    created_at: Optional[str] = None
    run_evals: bool = False
    message: Optional[str] = None


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


@app.get("/status", response_model=StatusResponse, tags=["meta"])
def status():
    rows, _ = store.load_store()
    sources = sorted({r["source"] for r in rows}) if rows else []
    try:
        emb = config.active_embedding()
    except Exception as e:
        emb = {"error": str(e)}
    try:
        llm_p = config.active_llm()
        llm_safe = {k: v for k, v in llm_p.items() if k != "api_key"}
    except Exception as e:
        llm_safe = {"error": str(e)}
    return StatusResponse(
        chunks=len(rows),
        sources=sources,
        embedding_profile=emb,
        llm_profile=llm_safe,
        store_dir=config.STORE_DIR,
    )


@app.post("/ask", response_model=AskResponse, tags=["rag"])
def ask(body: AskRequest):
    if not body.question.strip():
        raise HTTPException(400, "question is required")
    try:
        answer, hits = rag.ask(
            body.question, backend=body.backend, k=body.k, llm_profile=body.llm_profile
        )
    except Exception as e:
        raise HTTPException(500, f"ask failed: {e}") from e
    return AskResponse(
        answer=answer,
        hits=hits,
        backend=body.backend,
        k=body.k,
        llm_profile=body.llm_profile or config.active_llm().get("id"),
    )


@app.get("/retrieve", response_model=RetrieveResponse, tags=["rag"])
def retrieve(
    q: str = Query(..., min_length=1),
    backend: str = Query("chroma"),
    k: int = Query(config.TOP_K, ge=1, le=50),
):
    try:
        hits = retrieval.retrieve(q, k=k, backend=backend)
    except Exception as e:
        raise HTTPException(500, f"retrieve failed: {e}") from e
    return RetrieveResponse(hits=hits, backend=backend, k=k)


@app.post("/ingest", response_model=IngestResponse, tags=["rag"])
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")
    suffix = os.path.splitext(file.filename)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    dest = None
    try:
        dest = os.path.join(tempfile.gettempdir(), os.path.basename(file.filename))
        os.replace(tmp_path, dest)
        n = ingest(dest)
    except Exception as e:
        raise HTTPException(500, f"ingest failed: {e}") from e
    finally:
        for p in (tmp_path, dest):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass
    return IngestResponse(file=os.path.basename(file.filename), chunks=n)


@app.get("/traces", tags=["observability"])
def traces(limit: int = Query(20, ge=1, le=200)):
    path = config.TRACE_PATH
    if not os.path.exists(path):
        return {"traces": []}
    with open(path, encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    items = []
    for line in reversed(lines[-limit:]):
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {"traces": items}


@app.post("/evals", tags=["evals"])
def run_evals(body: EvalRequest):
    from ragcore import evals

    try:
        evals.run(body.label, backend=body.backend, k=body.k)
    except Exception as e:
        raise HTTPException(500, f"evals failed: {e}") from e
    return {"ok": True, "label": body.label, "backend": body.backend, "k": body.k}


@app.post("/jobs/reindex", response_model=ReindexResponse, tags=["jobs"])
def jobs_reindex(body: ReindexRequest = ReindexRequest()):
    return ReindexResponse(**_dispatch_reindex(body.run_evals))


@app.get("/jobs/reindex/{run_id}", tags=["jobs"])
def jobs_reindex_status(run_id: int):
    return _get_run(run_id)


@app.get("/embeddings", tags=["config"])
def embeddings_config():
    path = config.EMBED_CONFIG_PATH
    if not os.path.exists(path):
        raise HTTPException(404, "embeddings.json not found")
    return config.load_json(path)


@app.get("/llms", tags=["config"])
def llms_config():
    path = config.LLM_CONFIG_PATH
    if not os.path.exists(path):
        raise HTTPException(404, "llms.json not found")
    return config.load_json(path)


@app.get("/console", response_class=HTMLResponse, include_in_schema=False)
def console():
    return CONSOLE_HTML


@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "RAG Project API",
        "docs": "/docs",
        "console": "/console",
        "health": "/health",
        "status": "/status",
        "jobs": {"reindex": "POST /jobs/reindex"},
    }
