# RAG Project: Retrieval + Observability + Live API

Modular Retrieval-Augmented Generation system in Python.

- **Storage:** JSON + NumPy source of truth; Chroma rebuildable index
- **Retrieval:** vector / BM25 / hybrid (RRF) + optional cross-encoder rerank
- **Pluggable embeddings** via `config/embeddings.json`
- **Pluggable LLM gateways** via `config/llms.json` (Anthropic, OpenAI, OpenAI-compatible)
- **Observability:** per-stage traces, tokens, cost
- **Live API:** FastAPI + Swagger — free-hostable, OpenAPI for automation tools
- **UI:** Live API tab calls `/ask` / `/ingest`; LLMs tab lists gateway profiles

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # OPENAI_API_KEY, ANTHROPIC_API_KEY, optional LLM_API_KEY
uvicorn api:app --reload --host 0.0.0.0 --port 8000
# → http://localhost:8000/docs
```

## LLM gateways (`config/llms.json`)

```json
{
  "active": "anthropic-sonnet",
  "profiles": [
    {"id": "anthropic-sonnet", "provider": "anthropic", "model": "claude-sonnet-4-5", "api_key_env": "ANTHROPIC_API_KEY"},
    {"id": "openai-gpt4o-mini", "provider": "openai", "model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
    {"id": "my-gateway", "provider": "openai_compatible", "model": "my-model",
     "base_url": "https://gateway.example.com/v1", "api_key_env": "LLM_API_KEY"}
  ]
}
```

- Override active profile: env `LLM_PROFILE=openai-gpt4o-mini`
- Optional `price_in` / `price_out` (USD per 1M tokens) for cost in traces
- `rag.ask(..., llm_profile="my-gateway")` or API body `{ "llm_profile": "..." }`

Providers: `anthropic` | `openai` | `openai_compatible` (any OpenAI-compatible chat API).

## API endpoints

| Method | Path | Purpose |
|--------|------|--------|
| GET | `/health` | Liveness |
| GET | `/status` | Chunks, embedding + **LLM** profiles |
| POST | `/ask` | Answer (`question`, `backend`, `k`, optional `llm_profile`) |
| GET | `/retrieve` | Top-k chunks |
| POST | `/ingest` | Upload PDF |
| GET | `/traces` | Recent traces |
| GET | `/llms` | LLM profiles JSON |
| GET | `/embeddings` | Embedding profiles JSON |
| POST | `/evals` | Offline eval set |

## UI → Live API

1. Deploy API (e.g. Render + `render.yaml`).
2. Open Pages UI or `ui/index.html`.
3. **Connect** → set API base URL → Save.
4. **Live API** → status / ask (pick LLM profile) / ingest / traces.
5. **LLMs** → list profiles from `/llms`.

## Free hosting

Build: `pip install -r requirements.txt`  
Start: `uvicorn api:app --host 0.0.0.0 --port $PORT`  
Secrets: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, optional `LLM_API_KEY`
