# Features

A complete reference of every feature in the Agent Search Engine.

---

## Core

### A2A-native registry
The search engine is itself a valid A2A agent. It publishes its own agent card at `/.well-known/agent.json` and responds to `message/send` JSON-RPC calls — so any A2A-compatible orchestrator can discover and query it the same way it would any other agent, with no special casing.

### Semantic search
Agents are indexed by embedding their card (name, description, skills, tags) using the `BAAI/bge-small-en-v1.5` model via fastembed + ONNX runtime. Queries are embedded at search time and ranked by cosine similarity. Results reflect intent, not just keyword matches — "summarise documents" finds a PDF summariser agent even if the word "summarise" doesn't appear verbatim in the card.

### Tag filtering
`POST /search` accepts an optional `tags` array. When provided, only agents whose card or skill tags overlap with the filter are considered, regardless of semantic score. Useful for narrowing to a capability category (e.g. `["nlp", "vision"]`).

### Agent registration
`POST /register` accepts either a full A2A agent card JSON or a base URL, in which case the engine fetches `<url>/.well-known/agent.json` automatically. Optional fields (`humanReadableId`, `provider`, `agentVersion`) are auto-filled if omitted, so non-strict A2A cards register without error.

### Agent listing and lookup
- `GET /agents` — paginated list of all registered agents (default 20 per page, max 100)
- `GET /agents/{id}` — single agent card lookup

### Agent removal
`DELETE /agents/{id}` removes an agent from the registry immediately.

---

## Health Monitoring

### Periodic liveness checks
A background asyncio task runs on a configurable interval (default every 5 minutes, set via `HEALTH_CHECK_INTERVAL_SECONDS`) and pings each registered agent's URL with a GET request (5-second timeout). The result updates the agent's status.

### Status tracking
Each agent carries one of four statuses:

| Status | Meaning |
|---|---|
| `unknown` | Never checked yet (just registered) |
| `online` | Last ping succeeded (HTTP < 500) |
| `offline` | Last ping failed; not yet stale |
| `stale` | Offline for more than `STALE_AFTER_DAYS` days (default 3) |

Status, `last_checked`, and `last_seen_online` timestamps are stored on the `agents` table and returned on every agent endpoint.

### Auto-deregistration
Agents that have been offline or stale for longer than `DEREGISTER_AFTER_DAYS` (default 7 days) are automatically deleted from the registry on the next health check cycle. A log line is emitted for each auto-deregistered agent.

### Status badges in the UI
Every agent card in the web UI displays a colour-coded status badge — green (online), red (offline), yellow (stale), grey (unknown).

---

## Card Versioning

### Change detection on re-register
When an agent re-registers, the new card is diffed against the stored card field by field. Only fields whose JSON-serialised value changed are recorded.

### Version history
Changed cards are saved as snapshots in the `agent_versions` table, storing the full new card and a `{field: {old, new}}` diff object. Version numbers are monotonically increasing per agent.

### History endpoint
`GET /agents/{id}/history` returns the full version timeline for an agent, ordered oldest to newest.

### History panel in the UI
Each agent card has a "History" button that expands an inline panel showing all recorded versions with colour-coded field diffs — red for removed/old values, green for new values.

---

## Search Analytics

### Query logging
Every call to `POST /search` logs the query text, result count, and any tag filters to the `search_logs` table.

### Click tracking
`POST /agents/{id}/click` records a click event in the `agent_clicks` table. The web UI fires this automatically when a user clicks on an agent card.

### Analytics endpoint
`GET /analytics` (API key required) returns aggregated stats:
- **Top queries** — the 10 most frequently searched terms
- **Zero-result queries** — the 10 most searched terms that returned no agents (reveals gaps in the registry)
- **Top clicked agents** — the 10 agents clicked most often

### Analytics panel in the UI
A collapsible "Search Analytics" section at the bottom of the web UI. Enter the API key to load and display the three analytics tables.

---

## Security

### API key authentication
Write and admin endpoints require an `X-Api-Key` header matching the `REGISTRY_API_KEY` environment variable:
- `POST /register`
- `DELETE /agents/{id}`
- `GET /analytics`

Read and search endpoints are intentionally public — discovery should be open. If `REGISTRY_API_KEY` is not set, the guard is skipped and a warning is logged at startup (convenient for local development).

### Rate limiting
Per-IP rate limits enforced by slowapi:

| Endpoint | Limit |
|---|---|
| `POST /search` | 30 req/min |
| `POST /` (A2A) | 30 req/min |
| `GET /agents` | 20 req/min |
| `GET /agents/{id}` | 60 req/min |
| `GET /agents/{id}/history` | 30 req/min |
| `POST /agents/{id}/click` | 60 req/min |

### Pagination
`GET /agents` returns a maximum of 100 agents per page (`skip` + `limit` query params with a `total` count). Prevents full-registry enumeration in a single request.

### Input validation
`SearchRequest.query` is capped at 500 characters. All request bodies are validated by Pydantic before reaching business logic.

---

## Dual-backend storage

The app auto-selects its database backend at startup based on the `DATABASE_URL` environment variable:
- **Set** → PostgreSQL (production, Railway)
- **Not set** → SQLite (`agents.db`, local development)

The public API is identical in both modes — nothing outside `database.py` knows which backend is active. This means zero setup for local development and durable storage in production without any code changes.

---

## Web UI

A dark-themed single-page interface served at `/ui`:

- **Search bar** with tag filter input
- **Result cards** showing name, description, endpoint URL, version, provider, tags, skills, similarity score, and health status badge
- **Register form** with two tabs — paste JSON or fetch from URL — plus API key input
- **All agents list** with health badges, history panel, and click tracking
- **Analytics panel** (API key gated) with top queries, zero-result queries, and top clicked agents

---

## API

Full OpenAPI/Swagger documentation is auto-generated by FastAPI and available at `/docs`. The engine also publishes its own A2A agent card at `/.well-known/agent.json`.

### Endpoints summary

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/.well-known/agent.json` | — | This engine's own A2A agent card |
| POST | `/` | — | A2A JSON-RPC 2.0 endpoint |
| POST | `/register` | Key | Register an agent card |
| GET | `/agents` | — | List agents (paginated) |
| GET | `/agents/{id}` | — | Get a single agent + health status |
| GET | `/agents/{id}/history` | — | Agent card version history |
| DELETE | `/agents/{id}` | Key | Remove an agent |
| POST | `/agents/{id}/click` | — | Log a click event |
| POST | `/search` | — | Semantic search |
| GET | `/analytics` | Key | Search and click analytics |
| GET | `/health` | — | Service health check |

---

## Configuration

All tuneable behaviour is controlled via environment variables:

| Variable | Default | Effect |
|---|---|---|
| `DATABASE_URL` | — | If set, use PostgreSQL; otherwise SQLite |
| `REGISTRY_API_KEY` | — | API key for write/admin endpoints; unset = open |
| `HEALTH_CHECK_INTERVAL_SECONDS` | `300` | How often to ping agents (seconds) |
| `STALE_AFTER_DAYS` | `3` | Days offline before status becomes `stale` |
| `DEREGISTER_AFTER_DAYS` | `7` | Days offline before auto-deletion |
| `PORT` | `8000` | Port the server listens on |
