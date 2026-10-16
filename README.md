# RAG Project: Retrieval + Observability + Live API

Modular Retrieval-Augmented Generation system in Python.

- **Storage:** JSON + NumPy are the source of truth; Chroma is a rebuildable index
- **Retrieval:** vector (NumPy / Chroma), keyword (BM25), hybrid (RRF fusion), optional cross-encoder reranking
- **Observability:** traces, per-stage timing, token cost, retrieval quality
- **Evals:** LLM-judged accuracy plus retrieval hit rate
- **Live API:** FastAPI + Swagger (`api.py`) for free hosting and automation connectors
- **UI:** GitHub Pages control panel — **Live API** tab calls `/ask`, `/ingest`, `/status`, `/traces`

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # OPENAI_API_KEY + ANTHROPIC_API_KEY
uvicorn api:app --reload --host 0.0.0.0 --port 8000
# Swagger: http://localhost:8000/docs
```

CLI still works: `python rag.py my.pdf "Your question"`

## Live API endpoints

| Method | Path | Purpose |
|--------|------|--------|
| GET | `/health` | Liveness |
| GET | `/status` | Chunks, sources, embedding, LLM |
| POST | `/ask` | Answer a question |
| GET | `/retrieve` | Top-k chunks only |
| POST | `/ingest` | Upload a PDF |
| GET | `/traces` | Recent traces |
| POST | `/evals` | Run offline eval set |
| GET | `/embeddings` | Embedding profiles |

OpenAPI: `/openapi.json` — import as a custom connector in Zapier / Make / n8n.

## UI → Live API

1. Deploy the API (Render free tier works with `render.yaml`).
2. Open the GitHub Pages UI (enable Pages → source GitHub Actions, or open `ui/index.html` locally).
3. **Connect** tab → set **API base URL** (e.g. `https://your-service.onrender.com`) → Save.
4. **Live API** tab:
   - **Refresh status** — chunk count / active embedding
   - **Ask** — question + backend (`chroma`, `hybrid`, `+rerank`, …)
   - **Upload & index** — PDF ingest
   - **Load traces** — recent timed runs

Settings are stored in `sessionStorage` (API URL + optional GitHub token).

## Free hosting

- **Render:** use `render.yaml` (free web service; cold starts after idle).
- Build: `pip install -r requirements.txt`
- Start: `uvicorn api:app --host 0.0.0.0 --port $PORT`
- Secrets: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`

## GitHub Actions / embeddings UI

Connect a fine-grained PAT (Contents, Actions, Workflows, Webhooks) under **Connect** to manage `config/embeddings.json` and trigger the `reindex` workflow. Optional Cloudflare Worker under `worker/` for live webhook push.

## AI gateways

Embeddings already support OpenAI-compatible base URLs via `config/embeddings.json` (`provider: openai_compatible`). The answer LLM defaults to Anthropic (`LLM_MODEL` in `config.py`); swap the client in `rag.py` / `evals.py` for any OpenAI-compatible gateway when needed.
