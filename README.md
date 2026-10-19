# RAG Project: Backend + UI + Worker

Modular Retrieval-Augmented Generation system, split by concern:

| Folder | Role |
|--------|------|
| **`backend/`** | FastAPI API, `ragcore` library, embeddings/LLM config, jobs, tests |
| **`ui/`** | Static web UI (GitHub Pages) |
| **`worker/`** | Cloudflare Worker — GitHub webhooks → live WebSocket fan-out |
| **`docs/`** | PDFs to index (reindex job) |

## Layout

```
rag-project/
├── backend/
│   ├── api.py              # FastAPI (uvicorn api:app)
│   ├── ragcore/            # library package
│   ├── config/             # embeddings.json, llms.json
│   ├── jobs/run_job.py     # GitHub Actions reindex
│   ├── tests/
│   ├── requirements.txt
│   └── .env.example
├── ui/                     # Pages UI
├── worker/                 # Cloudflare webhook hub
├── docs/                   # source PDFs
└── .github/workflows/
```

## Backend quick start

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add keys
uvicorn api:app --reload --host 0.0.0.0 --port 8000
# → http://localhost:8000/docs
```

Tests:

```bash
cd backend && pytest tests/ -q
```

## UI

Static files in `ui/`. Deployed by `.github/workflows/pages.yml` on push to `ui/**`.

## Worker

Cloudflare Worker in `worker/` (webhook + WebSocket hub). Deployed by `deploy-worker.yml`.

## Free API host (Render)

`render.yaml` builds/runs from **`backend/`**:

```
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port $PORT
```
