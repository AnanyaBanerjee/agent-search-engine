# Architecture

## Overview

The Agent Search Engine is an **A2A-protocol-native registry** — it is itself a valid A2A agent that other agents can query to discover peers. Humans can also browse it through a web UI.

```
┌──────────────────────────────────────────────────────────────────┐
│                     Agent Search Engine                          │
│                                                                  │
│  ┌────────────┐   ┌─────────────┐   ┌──────────────────────┐   │
│  │  FastAPI   │   │   SQLite    │   │  fastembed (ONNX)    │   │
│  │  (HTTP)    │──▶│  agents.db  │   │  BGE-small-en-v1.5   │   │
│  └────────────┘   └─────────────┘   └──────────────────────┘   │
│       │                                         │                │
│  ┌────▼──────────────────────────────────────── ▼ ──────────┐  │
│  │          Cosine Similarity Search (numpy, in-RAM)         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Endpoints                                                       │
│  ├── GET  /.well-known/agent.json   ← own A2A agent card        │
│  ├── POST /                         ← A2A JSON-RPC 2.0          │
│  ├── POST /register                 ← submit an agent card      │
│  ├── POST /search                   ← semantic search (REST)    │
│  ├── GET  /agents                   ← list all agents           │
│  └── GET  /ui                       ← web UI (static files)     │
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

### SQLite (`agents.db`)

**What:** An embedded, file-based relational database.

**Why:**
- **No infrastructure.** No Postgres server, no Docker, no connection strings. The database is a single file that travels with the repo. This matters a lot in an early-stage project — the fewer moving parts, the faster you can iterate.
- **Sufficient scale.** A registry of AI agents is not a high-write-throughput workload. SQLite handles thousands of agents and hundreds of concurrent reads comfortably.
- **Persistence without complexity.** Agent cards survive restarts, and SQLite's WAL mode handles concurrent reads safely.

**When to migrate away:**
If the registry grows to tens of thousands of agents with heavy concurrent writes, or if you need to run multiple server replicas, switch to PostgreSQL (with `pgvector` for native vector search). The `database.py` abstraction layer makes this swap straightforward.

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

**What:** All agent embeddings are loaded from SQLite into a numpy matrix on every search, and cosine similarity is computed in one matrix multiply.

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
        SQLite: INSERT INTO agents (id, card_json, embedding)
```

### Search (REST)

```
Client  →  POST /search { query: "summarise PDFs", top_k: 5 }
               │
               ▼
         embed_text(query)   →  fastembed ONNX  →  384-dim query vector
               │
               ▼
         load_vector_index()  →  all embeddings from SQLite into numpy matrix
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
  database.py      SQLite read/write + numpy cosine search
  search.py        fastembed model loader + agent card text builder
  a2a_handler.py   JSON-RPC 2.0 dispatcher (message/send, tasks/get, etc.)
  requirements.txt Pinned dependencies

frontend/
  index.html       Search UI, register form, agent list
  style.css        Dark theme
  app.js           Fetch calls to /search, /register, /agents

example-agents/
  pdf-summariser.json   Example A2A agent card
  code-reviewer.json    Example A2A agent card
```

---

## Future Considerations

| Concern | Current approach | When to upgrade | Upgrade path |
|---|---|---|---|
| Storage | SQLite | >50k agents or multi-replica | PostgreSQL |
| Vector search | numpy cosine | >50k agents or <10ms SLA | pgvector or Qdrant |
| Embeddings | fastembed ONNX (local) | GPU scale / multilingual | Hosted API (Cohere, Voyage) |
| Auth | None (open registry) | Production / private registry | API keys or OAuth2 per A2A spec |
| Agent card freshness | Manual re-register | Stale cards are a problem | Periodic background re-fetch via card_url |
