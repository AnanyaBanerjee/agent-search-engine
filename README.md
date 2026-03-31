# Agent Search Engine

A2A-compatible registry and **semantic search engine** for AI agents — with multi-signal ranking, health monitoring, card versioning, analytics, and agent reviews.

Agents register their [A2A Agent Card](https://a2a-protocol.org) and the engine indexes it with dense embeddings. Any other agent (or human) can query the registry in natural language to find the best agent for a task. Results are ranked by a composite score that blends semantic similarity, click-through rate, recency, task affinity, and reputation — not just keyword matching.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      Agent Search Engine                         │
│                                                                  │
│  /.well-known/agent.json   ←  own A2A agent card                │
│  POST /                    ←  A2A JSON-RPC 2.0 endpoint         │
│  POST /register            ←  submit an agent card  [key]       │
│  POST /search              ←  semantic search + reranking        │
│  GET  /agents              ←  list all agents (paginated)        │
│  GET  /agents/{id}         ←  single agent + health status       │
│  GET  /agents/{id}/history ←  card version history              │
│  POST /agents/{id}/click   ←  log a click (updates task affinity)│
│  POST /agents/{id}/review  ←  submit a star rating + comment     │
│  GET  /agents/{id}/reviews ←  list reviews + avg rating          │
│  DELETE /agents/{id}       ←  remove agent  [key]               │
│  GET  /analytics           ←  search & click analytics  [key]   │
│  GET  /ui                  ←  web UI                             │
└──────────────────────────────────────────────────────────────────┘
        ↑ agents call via A2A JSON-RPC
        ↑ humans browse via web UI
```

**Stack**
- **Backend**: FastAPI + uvicorn (Python)
- **Storage**: SQLite (local dev) / PostgreSQL (production) — auto-selected via `DATABASE_URL`
- **Embeddings**: `BAAI/bge-small-en-v1.5` via fastembed + ONNX runtime (384 dimensions)
- **Search**: Cosine similarity (numpy, in-memory) with multi-signal reranking
- **Protocol**: A2A v1.0 (JSON-RPC 2.0 over HTTP)
- **Deployment**: Railway

---

## Quick Start

```bash
# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. Start the server
python main.py
# → http://localhost:8000

# 3. Open the web UI
open http://localhost:8000/ui

# 4. Explore the auto-generated API docs
open http://localhost:8000/docs
```

No database setup needed — SQLite is created automatically on first run.

---

## Registering an Agent

### Via the Web UI
Open `http://localhost:8000/ui` → **Register a new agent** → paste JSON or enter a base URL.

### Via REST
```bash
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: your-key" \
  -d '{"agent_card": {"schemaVersion":"1.0","name":"My Agent","description":"What it does","url":"https://my-agent.example.com","agentVersion":"1.0.0","provider":{"name":"My Org"},"capabilities":{"a2aVersion":"1.0"},"authSchemes":[{"type":"none"}],"skills":[{"id":"s1","name":"My Skill","description":"...","tags":["nlp"]}],"tags":["nlp"]}}'
```

### By URL (engine fetches the card automatically)
```bash
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: your-key" \
  -d '{"agent_card":{"schemaVersion":"1.0","name":"x","description":"","url":"https://my-agent.example.com","agentVersion":"1.0.0","capabilities":{"a2aVersion":"1.0"},"authSchemes":[{"type":"none"}]}, "card_url":"https://my-agent.example.com"}'
```

---

## Searching

### A2A (agent-to-agent)
```bash
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0", "id": "1", "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Find an agent that summarises PDF documents"}],
        "messageId": "msg-001"
      }
    }
  }'
```

### REST
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "summarise PDF documents", "top_k": 5}'
```

Tag filtering:
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "extract text", "top_k": 5, "tags": ["nlp", "vision"]}'
```

---

## Multi-Signal Ranking

Search results are not ordered by semantic similarity alone. Each candidate is scored by a weighted formula:

```
score = 0.50 × semantic_similarity
      + 0.20 × click_through_rate
      + 0.15 × recency_score          (exponentially decayed recent clicks)
      + 0.10 × task_affinity           (query vs. agent's historical click-query centroid)
      + 0.05 × reputation              (avg review rating)
      + cold_start_bonus               (+0.10 for new / unseen agents)
```

The engine fetches a 3× enlarged candidate pool from cosine search, rerankss with all signals, then returns the top `top_k`. All weights are tunable via environment variables. See [`Agent_Engine_Optimization.md`](Agent_Engine_Optimization.md) for the full design.

---

## Reviews

```bash
# Submit a review
curl -X POST http://localhost:8000/agents/myorg__my-agent/review \
  -H "Content-Type: application/json" \
  -d '{"reviewer_id": "agent-xyz", "score": 5, "comment": "Excellent at summarisation"}'

# List reviews
curl http://localhost:8000/agents/myorg__my-agent/reviews
```

Reviews are one-per-reviewer per agent (upsert). The average rating feeds into ranking.

---

## Agent Card Format (A2A v1.0)

```json
{
  "schemaVersion": "1.0",
  "humanReadableId": "myorg/my-agent",
  "name": "My Agent",
  "description": "What my agent does",
  "url": "https://my-agent.example.com",
  "agentVersion": "1.0.0",
  "provider": { "name": "My Org" },
  "capabilities": { "a2aVersion": "1.0" },
  "authSchemes": [{ "type": "none" }],
  "skills": [
    {
      "id": "my-skill",
      "name": "My Skill",
      "description": "Describe what the skill does in detail",
      "tags": ["tag1", "tag2"],
      "examples": ["Example query that triggers this skill"]
    }
  ],
  "tags": ["category", "domain"]
}
```

See `example-agents/` for complete examples.

---

## API Reference

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/.well-known/agent.json` | — | This engine's own A2A agent card |
| `POST` | `/` | — | A2A JSON-RPC 2.0 endpoint |
| `POST` | `/register` | Key | Register an agent card |
| `GET` | `/agents` | — | List agents (paginated: `?skip=0&limit=20`) |
| `GET` | `/agents/{id}` | — | Single agent + health status |
| `GET` | `/agents/{id}/history` | — | Card version history |
| `DELETE` | `/agents/{id}` | Key | Remove an agent |
| `POST` | `/agents/{id}/click` | — | Log a click `{"query": "..."}` |
| `POST` | `/agents/{id}/review` | — | Submit review `{"reviewer_id","score","comment"}` |
| `GET` | `/agents/{id}/reviews` | — | List reviews + avg rating |
| `POST` | `/search` | — | Semantic search + reranking |
| `GET` | `/analytics` | Key | Top queries, zero-result queries, top clicked agents |
| `GET` | `/health` | — | Service health check |
| `GET` | `/docs` | — | Swagger UI |

**Rate limits** (per IP): `/search` 30/min · `/agents` 20/min · `/agents/{id}` 60/min · `/agents/{id}/click` 60/min

---

## A2A Methods Supported

| Method | Description |
|---|---|
| `message/send` | Send a task query, get matching agents back |
| `tasks/get` | Retrieve a previous task result |
| `tasks/cancel` | Cancel a task |
| `ping` | Health check |

---

## Configuration

All behaviour is controlled via environment variables:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | — | PostgreSQL connection string; unset = SQLite |
| `REGISTRY_API_KEY` | — | API key for write/admin endpoints; unset = open |
| `PORT` | `8000` | Server port |
| `HEALTH_CHECK_INTERVAL_SECONDS` | `300` | How often to ping registered agents |
| `STALE_AFTER_DAYS` | `3` | Days offline before status → `stale` |
| `DEREGISTER_AFTER_DAYS` | `7` | Days offline before auto-deletion |
| `RANK_WEIGHT_SEMANTIC` | `0.50` | Ranking: semantic similarity weight |
| `RANK_WEIGHT_CTR` | `0.20` | Ranking: click-through rate weight |
| `RANK_WEIGHT_RECENCY` | `0.15` | Ranking: recency weight |
| `RANK_WEIGHT_TASK_AFFINITY` | `0.10` | Ranking: task affinity weight |
| `RANK_WEIGHT_REPUTATION` | `0.05` | Ranking: review score weight |
| `RANK_COLD_START_BONUS` | `0.10` | Ranking: boost for new agents |
| `RANK_POOL_MULTIPLIER` | `3` | Candidate pool = top_k × this |

---

## Railway Deployment

1. Create a Railway project and link this repo
2. Add a **PostgreSQL** service
3. In your app service → **Variables** → add `DATABASE_URL` as a Variable Reference pointing to the Postgres service
4. Optionally set `REGISTRY_API_KEY`
5. Deploy — tables are created automatically on first startup

---

## Further Reading

- [`features.md`](features.md) — complete feature reference
- [`Agent_Engine_Optimization.md`](Agent_Engine_Optimization.md) — ranking system design
- [`database_choices.md`](database_choices.md) — why SQLite + PostgreSQL
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — component decisions and data flow
