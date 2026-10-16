# RAG Project: Retrieval + Observability + Live API

Modular Retrieval-Augmented Generation system in Python.

- **Storage:** JSON + NumPy are the source of truth; Chroma is a rebuildable index
- **Retrieval:** vector (NumPy / Chroma), keyword (BM25), hybrid (RRF fusion), optional cross-encoder reranking
- **Observability:** traces, per-stage timing, token cost, retrieval quality
- **Evals:** LLM-judged accuracy plus retrieval hit rate
- **Live API:** FastAPI + Swagger (`api.py`) for free hosting and automation connectors
- **UI + Actions:** GitHub Pages UI manages embedding profiles and cron/reindex jobs; optional Cloudflare Worker for live webhooks

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY and ANTHROPIC_API_KEY
```

CLI:

```bash
python rag.py my.pdf "What is the refund window?"
python rag.py my.pdf "..." hybrid+rerank
```

Live API (Swagger at `/docs`):

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## API endpoints

| Method | Path | Purpose |
|--------|------|--------|
| GET | `/health` | Liveness |
| GET | `/status` | Chunks, sources, active embedding, LLM |
| POST | `/ask` | Answer a question |
| GET | `/retrieve` | Top-k chunks only |
| POST | `/ingest` | Upload a PDF |
| GET | `/traces` | Recent traces |
| POST | `/evals` | Run offline eval set |
| GET | `/embeddings` | Embedding profiles |

OpenAPI at `/openapi.json` can be imported as a custom connector in Zapier, Make, n8n, etc.

## Free live hosting

Use `render.yaml` for one-click deploy on Render free tier, or any Python host:

- Build: `pip install -r requirements.txt`
- Start: `uvicorn api:app --host 0.0.0.0 --port $PORT`
- Secrets: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`

## Project layout

```
config.py, helpers.py, store.py, ingest.py, embeddings.py
kwsearch.py, rerank.py, retrieval.py, tracing.py, rag.py
api.py          # FastAPI + Swagger
evals.py, report.py, compare.py
jobs/run_job.py
ui/             # GitHub Pages config & jobs UI
.github/workflows/
worker/         # optional Cloudflare webhook receiver
```

## GitHub UI & Actions

1. Enable GitHub Pages (source: GitHub Actions).
2. Add repo secrets: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`.
3. Open the Pages URL → configure owner/repo/token to manage embedding profiles, trigger reindex, edit cron, and watch jobs.

Optional real-time webhooks: deploy the Worker (`CLOUDFLARE_*`, `WEBHOOK_SECRET`, `VIEW_KEY` secrets) or use [smee.io](https://smee.io).

## AI gateways

Embeddings support OpenAI-compatible base URLs via `config/embeddings.json` (`provider: openai_compatible`). Answer LLM is Anthropic by default (`LLM_MODEL` in `config.py`); swap the client in `rag.py` / `evals.py` for any OpenAI-compatible gateway when needed.
