"""
api.py - FastAPI backend with Swagger UI.

Exposes the RAG pipeline (ask, retrieve, ingest, evals, traces, status)
as a free-hostable HTTP service. Open /docs for interactive Swagger.

Run locally:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Optional

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
        "Use as a custom connector in automation tools via OpenAPI."
    ),
    version="1.1.0",
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
    }
