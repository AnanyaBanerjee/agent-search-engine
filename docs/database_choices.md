# Database Choices

This document covers every database option we evaluated for the Agent Search Engine, why we ruled each one out, and why we landed on the dual SQLite/PostgreSQL approach.

---

## What we needed

The registry has a specific workload profile that shaped every decision:

- **Write-light.** Agents register once and rarely update. Write throughput is negligible.
- **Read-heavy.** Every search reads all embeddings into RAM for cosine similarity. Many concurrent reads.
- **Two storage concerns.** Structured data (agent cards as JSON) and binary blobs (384-dim float32 embeddings). Both must live together efficiently.
- **Zero ops burden at the start.** An early-stage project should have no infrastructure to manage locally.
- **Cloud-compatible.** Production deployment on Railway must survive redeploys without data loss.

---

## Options evaluated

### SQLite

**What it is:** An embedded, file-based relational database. The database is a single `.db` file on disk — no server process, no network socket.

**Why it works well for local development:**
- No setup. Clone the repo, run the app, the database is created automatically.
- No Docker, no connection strings, no service to start or stop.
- Handles thousands of agents and hundreds of concurrent reads without issue — WAL mode allows concurrent readers with a single writer.
- The database file travels with the repo and can be committed for snapshots or shared for debugging.

**Why it doesn't work for production on Railway:**
- Railway's filesystem is ephemeral. Every redeploy starts from a clean image, wiping the SQLite file and losing all registered agents.
- Can't be shared across multiple replicas — each instance would have its own copy of the file with diverging state.

**Verdict:** Perfect for local development. Not viable for production.

---

### PostgreSQL

**What it is:** A full client-server relational database. Runs as a separate service; application connects over a network socket.

**Why it works well for production:**
- Persistent across redeploys — data lives in the managed Postgres service, not the app container.
- Handles concurrent writes safely with proper locking and MVCC.
- Railway provides a managed Postgres add-on with automatic `DATABASE_URL` injection — zero infrastructure work beyond adding the add-on.
- Supports `pgvector` extension when we need native vector indexing at scale.

**Why we didn't use it for local development:**
- Requires a running Postgres server (or Docker). Adds friction for new contributors.
- Overkill for a single-developer local environment where the dataset is small and restarts are frequent.

**Verdict:** Right choice for production. Too heavy for local development.

---

### MongoDB

**What it is:** A document-oriented NoSQL database. Stores JSON-like BSON documents.

**Why we considered it:**
- Agent cards are JSON documents — a document store is a natural fit for the data shape.
- Flexible schema would handle the variation in A2A agent card fields without migrations.

**Why we ruled it out:**
- No native support for binary vector blobs. Embeddings would need to be stored as arrays of floats or as base64 strings — neither is efficient for the bulk reads the similarity search requires.
- Adds a third dependency just for document storage when PostgreSQL handles JSON natively via the `TEXT` column + application-level parsing (which is what we do).
- Requires a running server locally, removing the zero-setup advantage.
- The A2A schema is well-defined via Pydantic models — a flexible schema is a liability here, not an asset.

**Verdict:** Rejected. Adds complexity without meaningful benefit over Postgres + JSON columns.

---

### Redis

**What it is:** An in-memory key-value store, often used for caching, session storage, and pub/sub.

**Why we considered it:**
- Extremely fast reads — agent cards could be cached in RAM for near-zero latency lookups.
- Could store embeddings as Redis vectors (RediSearch module supports vector similarity).

**Why we ruled it out:**
- Not a primary store — Redis is typically used alongside a persistent database, not instead of one. Using it as the sole store adds complexity without removing the need for a real database.
- RediSearch vector indexing requires the RediSearch module, which is not available on all Redis hosts and adds licensing considerations.
- In-RAM storage is already handled by our numpy cosine search — we load embeddings into a matrix per search. Adding Redis for this would duplicate the in-RAM layer without improving it.
- No Railway-native Postgres-equivalent simplicity for Redis at the time of design.

**Verdict:** Rejected. Redundant with the existing in-RAM numpy approach and adds infrastructure complexity.

---

### Pinecone / Qdrant / Weaviate (dedicated vector databases)

**What they are:** Purpose-built vector databases that store embeddings and perform approximate nearest-neighbour (ANN) search.

**Why we considered them:**
- Native vector search — no numpy cosine loop needed.
- Scale to millions of vectors efficiently using HNSW or IVF indexes.
- Managed hosting available (Pinecone, Weaviate Cloud).

**Why we ruled them out:**
- **Over-engineered for the current scale.** A numpy dot product over 10,000 384-dim vectors takes ~2 ms. We don't need ANN search until we exceed ~50,000 agents.
- **External dependency.** Every search would require a network call to an external service, adding latency and a new failure mode.
- **Cost.** Managed vector databases charge per vector stored and per query. At early-stage scale this is unnecessary spend.
- **Split storage.** We'd still need a relational database for agent card JSON, registered timestamps, health status, and version history. Two databases is worse than one.

**Verdict:** Rejected for now. The migration path when we need them is documented in ARCHITECTURE.md — `pgvector` on the existing Postgres instance is the most likely first step.

---

### DynamoDB / Firestore (cloud-native NoSQL)

**What they are:** Fully managed, serverless NoSQL databases from AWS and Google Cloud respectively.

**Why we considered them:**
- Zero infrastructure — no server to manage, scales automatically.
- Native JSON document storage.

**Why we ruled them out:**
- **Vendor lock-in.** The project is deployed on Railway, not AWS or GCP. Using DynamoDB would require AWS credentials, SDK, and billing separate from Railway.
- **No binary blob support.** Embeddings would need base64 encoding, inflating storage and slowing reads.
- **No SQL.** Analytics queries (GROUP BY query, COUNT clicks) are natural SQL. With a NoSQL store these require application-level aggregation or a separate analytics service.
- **Cost model.** Pay-per-read/write pricing is unpredictable for a search engine with bursts of read traffic.

**Verdict:** Rejected. Wrong deployment target and wrong data model for this workload.

---

## What we chose and why

**Dual-backend: SQLite (local) + PostgreSQL (production)**

The core insight is that local development and production have fundamentally different constraints. Rather than compromise — using Postgres locally (heavyweight) or SQLite in production (data loss) — we built a thin abstraction in `database.py` that presents an identical API regardless of which backend is active.

At startup, `database.py` checks for the `DATABASE_URL` environment variable:
- **Set** (Railway injects this automatically when a Postgres add-on is attached) → connect to PostgreSQL via `psycopg2`
- **Not set** → open `agents.db` as SQLite

Nothing else in the codebase knows or cares which backend is running. All queries are written twice — once with `%s` placeholders for psycopg2, once with `?` for sqlite3 — and selected by the same `if _USE_POSTGRES` branch pattern used everywhere.

This costs roughly 2x the SQL code in `database.py` and nothing else. The tradeoff is well worth it: contributors get zero-setup local development, and production gets durable, production-grade storage.

---

## Schema overview

All tables exist in both backends with equivalent structure:

| Table | Purpose |
|---|---|
| `agents` | Agent cards, embeddings, health status |
| `agent_versions` | Card change history (snapshot + field-level diff) |
| `search_logs` | Every search query + result count (analytics) |
| `agent_clicks` | Agent click events (analytics) |

---

## When to migrate away from this approach

| Trigger | Recommended change |
|---|---|
| >50,000 agents | Add `pgvector` to Postgres; replace numpy cosine search with native vector query |
| Multi-region deployment | Postgres read replicas or a globally distributed database (CockroachDB, PlanetScale) |
| >1M embeddings/day | Replace fastembed with a hosted embedding API (Cohere, Voyage, OpenAI) |
| Per-user access control | Add a `tenants` table and row-level security in Postgres |
