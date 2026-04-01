# Architecture

## Overview

The Agent Search Engine is an **A2A-protocol-native registry** — it is itself a valid A2A agent that other agents can query to discover peers. Humans can also browse it through a web UI.

```
┌──────────────────────────────────────────────────────────────────┐
│                     Agent Search Engine                          │
│                                                                  │
│  ┌────────────┐   ┌──────────────────┐   ┌──────────────────────┐   │
│  │  FastAPI   │   │ SQLite (local) /  │   │  fastembed (ONNX)    │   │
│  │  (HTTP)    │──▶│ PostgreSQL (prod) │   │  BGE-small-en-v1.5   │   │
│  └────────────┘   └──────────────────┘   └──────────────────────┘   │
│       │                                               │               │
│  ┌────▼──────────────────────────────────────────── ▼ ───────────┐  │
│  │             Cosine Similarity Search (numpy, in-RAM)           │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  Endpoints                                                            │
│  ├── GET  /.well-known/agent.json   ← own A2A agent card             │
│  ├── POST /                         ← A2A JSON-RPC 2.0               │
│  ├── POST /register                 ← submit an agent card [key]     │
│  ├── POST /search                   ← semantic search (REST)         │
│  ├── GET  /agents                   ← list agents (paginated)        │
│  ├── GET  /agents/{id}              ← get a single agent card        │
│  ├── DELETE /agents/{id}            ← remove an agent [key]          │
│  ├── GET  /health                   ← health check                   │
│  └── GET  /ui                       ← web UI (static files)          │
└──────────────────────────────────────────────────────────────────┘

         ▲ A2A JSON-RPC           ▲ REST / browser
         │                        │
   [AI Agents]            [Humans / other tools]
```

---

## Component Decisions

### FastAPI

**What:** Python async web framework.

**Why:**
- First-class async support — critical for handling many concurrent agent requests without blocking.
- Automatic OpenAPI/Swagger docs at `/docs` with zero extra work, which is useful for developers integrating agents.
- Pydantic is built in, so A2A's JSON schema (Agent Cards, JSON-RPC envelopes) maps directly to typed models with validation.
- Lightweight compared to Django; more structured than Flask.

---

### Storage — SQLite (local) / PostgreSQL (production)

**What:** Dual-backend store. `database.py` auto-selects at startup:
- `DATABASE_URL` set → PostgreSQL via `psycopg2` (Railway / any hosted Postgres)
- `DATABASE_URL` unset → SQLite (`agents.db`, file-based)

The public API (`upsert_agent`, `get_agent`, `list_all_agents`, `load_vector_index`, `cosine_search`) is identical in both modes — nothing else in the codebase knows which backend is active.

**Why SQLite for local dev:**
- **No infrastructure.** No Postgres server, no Docker, no connection strings. The database is a single file that travels with the repo.
- **Sufficient scale.** A registry of AI agents is not a high-write-throughput workload. SQLite handles thousands of agents and hundreds of concurrent reads comfortably.

**Why PostgreSQL for production:**
- **Persistence across deploys.** Railway ephemeral filesystems lose SQLite on every redeploy; a managed Postgres add-on survives.
- **Concurrent writes.** Multi-replica deployments and Railway's zero-downtime restarts need a network-accessible database.

**When to extend further:**
If you need vector-native indexing (ANN search, filtered queries at scale), add `pgvector` to the existing Postgres instance and replace the in-RAM numpy search. The `database.py` abstraction keeps that change contained.

---

### fastembed + ONNX runtime

**What:** A lightweight embedding library that runs models as ONNX graphs — no PyTorch, no CUDA required.

**Why:**
- **No PyTorch dependency.** PyTorch weighs ~2 GB and introduces complex version constraints (NumPy 1.x vs 2.x, CUDA drivers, etc.). fastembed replaces all of that with a single ONNX runtime (~50 MB).
- **Cross-platform.** Works on Intel Mac, Apple Silicon, Linux, and Windows without platform-specific wheel juggling.
- **Production-grade model.** We use `BAAI/bge-small-en-v1.5` — a 130 MB model that consistently ranks in the top tier of the MTEB embedding benchmark for its size class. It outperforms the older `all-MiniLM-L6-v2` on most retrieval tasks.
- **Fast cold start.** The ONNX model loads in ~1 second vs ~5–10 seconds for a PyTorch model.

**When to migrate away:**
If you need GPU inference at scale (millions of embeddings/day), switch to a hosted embedding API (e.g. Cohere, Voyage AI, or OpenAI) and store vectors in `pgvector`. The `embed_text()` function in `search.py` is the only call site to change.

---

### Cosine Similarity (numpy, in-RAM)

**What:** All agent embeddings are loaded from the database (SQLite or PostgreSQL) into a numpy matrix on every search, and cosine similarity is computed in one matrix multiply.

**Why:**
- **Zero extra dependencies.** No vector database (Pinecone, Weaviate, Qdrant) to manage.
- **Fast enough.** A numpy dot product over 10,000 384-dimensional vectors takes ~2 ms. For a registry of AI agents, this is more than sufficient.
- **Simple to reason about.** The entire search path is ~20 lines of numpy — easy to debug, test, and extend (e.g. adding tag pre-filtering, score boosting).

**When to migrate away:**
Beyond ~50,000 agents, or if you need ANN (approximate nearest neighbour) search for latency reasons, add `pgvector` to Postgres or a dedicated vector store like Qdrant.

---

### A2A Protocol (JSON-RPC 2.0 over HTTP)

**What:** Google's open Agent-to-Agent interoperability protocol.

**Why:**
- **Standardised discovery.** Every agent publishes a `/.well-known/agent.json` card describing its capabilities, endpoint, and auth. This search engine indexes those cards.
- **Interoperability.** Any A2A-compatible agent (built with LangChain, Google ADK, CrewAI, or a custom stack) can register with and query this engine without bespoke integration work.
- **The search engine is itself an agent.** It publishes its own agent card and responds to `message/send` — so an orchestrator agent can discover it the same way it discovers any other agent, with no special casing.

---

### Security

**What:** A layered defence-in-depth approach covering authentication, rate limiting, and input validation.

**API key authentication (`REGISTRY_API_KEY` env var)**
- Write endpoints (`POST /register`, `DELETE /agents/{id}`) require an `X-Api-Key` header matching the env var.
- Read/search endpoints remain public — discovery is intentionally open.
- If `REGISTRY_API_KEY` is unset the guard is skipped, keeping local dev frictionless. A warning is logged at startup.

**Rate limiting (slowapi, per IP)**
- `POST /search` and `POST /` (A2A): 30 req/min
- `GET /agents`: 20 req/min
- `GET /agents/{id}`: 60 req/min

**Pagination**
- `GET /agents` returns at most 100 agents per page (`skip` + `limit` query params). Prevents full-registry enumeration in a single request.

**Input validation**
- `SearchRequest.query` is capped at 500 characters via Pydantic field constraint. All other models are validated by Pydantic on ingress.

**When to extend:**
If the registry becomes private (internal tooling, enterprise), replace the single shared API key with per-client keys or OAuth2/JWT as described in the A2A auth spec.

---

## Data Flow

### Registration

```
Agent  →  POST /register { agent_card: {...} }
              │
              ▼
        Validate with Pydantic (AgentCard model)
              │
              ▼
        embed_agent_card()   →  fastembed ONNX  →  384-dim float32 vector
              │
              ▼
        DB: INSERT INTO agents (id, card_json, embedding)  ← SQLite or PostgreSQL
```

### Search (REST)

```
Client  →  POST /search { query: "summarise PDFs", top_k: 5 }
               │
               ▼
         embed_text(query)   →  fastembed ONNX  →  384-dim query vector
               │
               ▼
         load_vector_index()  →  all embeddings from DB into numpy matrix
               │
               ▼
         cosine_similarity(query_vec, matrix)  →  ranked (agent_id, score) list
               │
               ▼
         Fetch full AgentCard for each hit  →  return SearchResponse
```

### Search (A2A)

```
Agent  →  POST / { jsonrpc:"2.0", method:"message/send", params:{ message:{ parts:[{text:"..."}] } } }
              │
              ▼
        a2a_handler.handle_jsonrpc()
              │
              ▼
        Extract text from message parts  →  same search_fn as REST
              │
              ▼
        Return JSON-RPC result with:
          - text artifact  (human-readable summary)
          - data artifact  (structured list of matching agent cards)
```

---

## File Map

```
backend/
  main.py          App entry point, FastAPI routes, lifespan hooks
  models.py        Pydantic models — A2A Agent Card schema + JSON-RPC types
  database.py      Dual-backend store (SQLite local / PostgreSQL prod) + numpy cosine search
  search.py        fastembed model loader + agent card text builder
  a2a_handler.py   JSON-RPC 2.0 dispatcher (message/send, tasks/get, etc.)
  requirements.txt Pinned dependencies
  runtime.txt      Python version pin for Railway/Nixpacks

  frontend/        Served at /ui (moved inside backend/ for Railway packaging)
    index.html     Search UI, register form, agent list
    style.css      Dark theme
    app.js         Fetch calls to /search, /register, /agents

example-agents/
  pdf-summariser.json   Example A2A agent card
  code-reviewer.json    Example A2A agent card

railway.json       Railway deployment config (builder, start command, restart policy)
```

---

## Future Considerations

| Concern | Current approach | When to upgrade | Upgrade path |
|---|---|---|---|
| Storage | SQLite (local) + PostgreSQL (prod) | Multi-region / high write throughput | Managed Postgres with pgvector |
| Vector search | numpy cosine | >50k agents or <10ms SLA | pgvector or Qdrant |
| Embeddings | fastembed ONNX (local) | GPU scale / multilingual | Hosted API (Cohere, Voyage) |
| Auth | API key for writes (X-Api-Key), reads open | Per-user auth / private registry | OAuth2 or JWT per A2A spec |
| Agent card freshness | Manual re-register | Stale cards are a problem | Periodic background re-fetch via card_url |
