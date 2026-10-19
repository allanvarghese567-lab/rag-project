# RAG Project: Backend + UI + Worker

Modular Retrieval-Augmented Generation system, split by concern:

| Folder | Role |
|--------|------|
| **`backend/`** | FastAPI API, `ragcore` library, embeddings/LLM config, jobs, tests |
| **`ui/`** | React + Vite + TypeScript control panel (GitHub Pages) |
| **`worker/`** | Cloudflare Worker — GitHub webhooks → live WebSocket fan-out |
| **`docs/`** | PDFs to index (reindex job) |

## Layout

```
rag-project/
├── backend/
│   ├── api.py
│   ├── ragcore/
│   ├── config/
│   ├── jobs/
│   └── tests/
├── ui/                     # React + Vite + TS
│   ├── src/
│   │   ├── components/     # reusable UI pieces
│   │   ├── hooks/
│   │   ├── lib/
│   │   └── App.tsx
│   ├── package.json
│   └── vite.config.ts
├── worker/
├── docs/
└── .github/workflows/
```

## Backend quick start

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

```bash
cd backend && pytest tests/ -q
```

## UI (React + Vite + TypeScript)

Reusable components: `Card`, `Nav`, `Button`, `Field`, `Toast`, plus feature panels
(`ConnectPanel`, `LiveApiPanel`, `LlmsPanel`, `EmbeddingsPanel`, `JobsPanel`).

```bash
cd ui
npm install
npm run dev      # http://localhost:5173
npm run build    # output → ui/dist
```

Deploy: push to `ui/**` triggers Pages workflow (builds and publishes `dist/`).
Set `base` in `vite.config.ts` to match the repo name for project Pages.

## Worker

Cloudflare Worker in `worker/`. Deployed by `deploy-worker.yml`.

## Free API host (Render)

`render.yaml` uses `rootDir: backend`.
