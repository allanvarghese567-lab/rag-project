"""
api.py - FastAPI backend with Swagger UI.

Exposes the RAG pipeline (ask, retrieve, ingest, evals, traces, status)
as a free-hostable HTTP service. Open /docs for interactive Swagger.

Also: POST /jobs/reindex triggers the GitHub Actions reindex workflow
(temporary runner) and returns the Actions run URL.

Run locally:
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
from pydantic import BaseModel, Field

import config
import rag
import retrieval
import store
from ingest import ingest

app = FastAPI(
    title="RAG Project API",
    description=(
        "Modular Retrieval-Augmented Generation backend. "
        "JSON + NumPy source of truth, Chroma index, hybrid/BM25/rerank, "
        "pluggable embeddings + LLM gateways, per-stage tracing. "
        "POST /jobs/reindex starts a temporary GitHub Actions runner. "
        "Use as a custom connector in automation tools via OpenAPI."
    ),
    version="1.2.0",
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

# ---------- GitHub Actions helpers (temporary reindex workers) ----------

GH_API = "https://api.github.com"
WORKFLOW_FILE = "reindex.yml"


def _gh_env():
    """Owner/repo/token/branch for dispatching Actions."""
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
    """
    Trigger workflow_dispatch on reindex.yml, then resolve the new run URL.
    Returns dict with run_id, html_url, status, etc.
    """
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
    body = {
        "ref": branch,
        "inputs": {"run_evals": str(bool(run_evals)).lower()},
    }
    r = requests.post(dispatch_url, headers=headers, json=body, timeout=30)
    if r.status_code not in (204, 200):
        detail = r.text
        try:
            detail = r.json().get("message", detail)
        except Exception:
            pass
        raise HTTPException(r.status_code if r.status_code >= 400 else 502,
                            f"workflow_dispatch failed: {detail}")

    # workflow_dispatch returns 204 with no run id — poll recent runs
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
        # Prefer a run that just started (queued / in_progress)
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


# ---------- request / response models ----------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What is the refund window?"])
    backend: str = Field(
        "chroma",
        description="numpy | chroma | hybrid | any of those +rerank (e.g. hybrid+rerank)",
    )
    k: int = Field(config.TOP_K, ge=1, le=50)
    llm_profile: Optional[str] = Field(
        None,
        description="Optional LLM profile id from config/llms.json (default: active)",
    )


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
    label: str = Field("api-run", description="Label stored with the eval summary")
    backend: str = "chroma"
    k: int = Field(config.TOP_K, ge=1, le=50)


class ReindexRequest(BaseModel):
    run_evals: bool = Field(
        False,
        description="If true, the Actions job runs evals after indexing",
    )


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


# ---------- endpoints ----------

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
            body.question,
            backend=body.backend,
            k=body.k,
            llm_profile=body.llm_profile,
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
    q: str = Query(..., min_length=1, description="Question / query text"),
    backend: str = Query("chroma"),
    k: int = Query(config.TOP_K, ge=1, le=50),
):
    try:
        hits = retrieval.retrieve(q, k=k, backend=backend)
    except Exception as e:
        raise HTTPException(500, f"retrieve failed: {e}") from e
    return RetrieveResponse(hits=hits, backend=backend, k=k)


@app.post("/ingest", response_model=IngestResponse, tags=["rag"])
async def ingest_pdf(file: UploadFile = File(..., description="PDF to index")):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")
    suffix = os.path.splitext(file.filename)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
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
    lines = open(path).read().strip().splitlines()
    items = []
    for line in reversed(lines[-limit:]):
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {"traces": items}


@app.post("/evals", tags=["evals"])
def run_evals(body: EvalRequest):
    import evals
    try:
        evals.run(body.label, backend=body.backend, k=body.k)
    except Exception as e:
        raise HTTPException(500, f"evals failed: {e}") from e
    return {"ok": True, "label": body.label, "backend": body.backend, "k": body.k}


@app.post(
    "/jobs/reindex",
    response_model=ReindexResponse,
    tags=["jobs"],
    summary="Start temporary reindex runner (GitHub Actions)",
)
def jobs_reindex(body: ReindexRequest = ReindexRequest()):
    """
    Dispatch the `reindex` workflow on GitHub Actions (a temporary runner).

    Requires server env:
      GITHUB_TOKEN  — PAT with Actions: write (and Contents: read)
      GITHUB_OWNER  — default allanvarghese567-lab
      GITHUB_REPO   — default rag-project
      GITHUB_BRANCH — default main

    Returns the Actions run URL when GitHub has registered the run.
    """
    result = _dispatch_reindex(body.run_evals)
    return ReindexResponse(**result)


@app.get(
    "/jobs/reindex/{run_id}",
    tags=["jobs"],
    summary="Poll status of a reindex Actions run",
)
def jobs_reindex_status(run_id: int):
    """Return status/conclusion/html_url for a workflow run id."""
    return _get_run(run_id)


@app.get("/embeddings", tags=["config"])
def embeddings_config():
    path = config.EMBED_CONFIG_PATH
    if not os.path.exists(path):
        raise HTTPException(404, "embeddings.json not found")
    return json.load(open(path))


@app.get("/llms", tags=["config"])
def llms_config():
    """Current LLM profiles and active id (from config/llms.json)."""
    path = config.LLM_CONFIG_PATH
    if not os.path.exists(path):
        raise HTTPException(404, "llms.json not found")
    return json.load(open(path))


@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "RAG Project API",
        "docs": "/docs",
        "health": "/health",
        "status": "/status",
        "jobs": {"reindex": "POST /jobs/reindex"},
    }
