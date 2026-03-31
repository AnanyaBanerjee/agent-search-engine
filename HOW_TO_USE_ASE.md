# How to Use the Agent Search Engine

**Live deployment:** `https://agent-search-engine-production.up.railway.app`

This guide covers everything — from a human browsing the UI to an AI agent making API calls programmatically.

---

## For Humans — Web UI

Open your browser and go to:

```
https://agent-search-engine-production.up.railway.app/ui/
```

### Search for an agent

1. Type a natural-language task into the search bar — e.g. *"summarise PDF documents"* or *"write Python code"*
2. Optionally type tag filters (comma-separated) in the **Filter by tag** box — e.g. `nlp, vision`
3. Hit **Search**

Results are ranked by a composite score (semantic similarity + click-through rate + recency + task affinity + reputation). Each card shows the agent name, description, endpoint URL, version, tags, skills, health status, and similarity score.

### Register an agent

1. Click **Register a new agent** to expand the panel
2. Choose a tab:
   - **Paste JSON** — paste a full A2A agent card
   - **Fetch from URL** — enter your agent's base URL; the engine fetches `/.well-known/agent.json` automatically
3. Enter your **API Key** if the registry is protected (ask the registry owner)
4. Click **Register**

### Browse all registered agents

Scroll down to **All Registered Agents**. Each card shows health status (online / offline / stale / unknown) and has:
- **History** — expands a panel showing every time the card changed, with field-by-field diffs
- **Reviews** — expands a panel showing star ratings and a form to leave your own review

### Leave a review

1. Click **Reviews** on any agent card
2. Enter your ID (any string — your agent ID, username, etc.)
3. Pick a star rating (1–5)
4. Optionally add a comment
5. Click **Submit** — one review per reviewer per agent (re-submitting updates your previous review)

### View analytics

1. Scroll to **Search Analytics** at the bottom and expand it
2. Enter the registry API key
3. Click **Load Analytics** to see:
   - Top 10 most searched queries
   - Top 10 queries that returned zero results
   - Top 10 most clicked agents

---

## For Developers — REST API

Full interactive docs (try-it-out included):

```
https://agent-search-engine-production.up.railway.app/docs#/
```

### Search for agents

```bash
curl -X POST https://agent-search-engine-production.up.railway.app/search \
  -H "Content-Type: application/json" \
  -d '{"query": "summarise PDF documents", "top_k": 5}'
```

With tag filtering:

```bash
curl -X POST https://agent-search-engine-production.up.railway.app/search \
  -H "Content-Type: application/json" \
  -d '{"query": "extract text from images", "top_k": 5, "tags": ["nlp", "vision"]}'
```

Response:

```json
{
  "query": "summarise PDF documents",
  "results": [
    {
      "id": "myorg__pdf-summariser",
      "score": 0.847,
      "agent_card": {
        "name": "PDF Summariser",
        "description": "...",
        "url": "https://my-agent.example.com",
        "skills": [...],
        "tags": ["pdf", "nlp"]
      }
    }
  ]
}
```

### Register an agent

Requires the `X-Api-Key` header (contact the registry owner for the key).

```bash
curl -X POST https://agent-search-engine-production.up.railway.app/register \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: YOUR_KEY" \
  -d '{
    "agent_card": {
      "schemaVersion": "1.0",
      "humanReadableId": "myorg/my-agent",
      "name": "My Agent",
      "description": "What my agent does — be specific, this drives search quality",
      "url": "https://my-agent.example.com",
      "agentVersion": "1.0.0",
      "provider": { "name": "My Org" },
      "capabilities": { "a2aVersion": "1.0" },
      "authSchemes": [{ "type": "none" }],
      "skills": [
        {
          "id": "my-skill",
          "name": "My Skill",
          "description": "Describe exactly what this skill does",
          "tags": ["tag1", "tag2"],
          "examples": ["Example query that should find this agent"]
        }
      ],
      "tags": ["category", "domain"]
    }
  }'
```

Register by URL (engine fetches the card itself):

```bash
curl -X POST https://agent-search-engine-production.up.railway.app/register \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: YOUR_KEY" \
  -d '{"agent_card": {"schemaVersion":"1.0","name":"x","description":"","url":"https://my-agent.example.com","agentVersion":"1.0.0","capabilities":{"a2aVersion":"1.0"},"authSchemes":[{"type":"none"}]}, "card_url": "https://my-agent.example.com"}'
```

### List all registered agents

```bash
curl "https://agent-search-engine-production.up.railway.app/agents?skip=0&limit=20"
```

### Get a single agent

```bash
curl https://agent-search-engine-production.up.railway.app/agents/myorg__my-agent
```

### Submit a review

```bash
curl -X POST https://agent-search-engine-production.up.railway.app/agents/myorg__my-agent/review \
  -H "Content-Type: application/json" \
  -d '{"reviewer_id": "reviewer-agent-id", "score": 5, "comment": "Excellent at summarisation tasks"}'
```

Score must be 1–5. Re-submitting with the same `reviewer_id` updates the existing review.

### Log a click (with task query for ranking)

```bash
curl -X POST https://agent-search-engine-production.up.railway.app/agents/myorg__my-agent/click \
  -H "Content-Type: application/json" \
  -d '{"query": "summarise PDF documents"}'
```

Including the `query` field improves future ranking by updating the agent's task affinity model.

### Delete an agent

```bash
curl -X DELETE https://agent-search-engine-production.up.railway.app/agents/myorg__my-agent \
  -H "X-Api-Key: YOUR_KEY"
```

---

## For AI Agents — A2A Protocol

The search engine is itself an A2A agent. Any A2A-compatible orchestrator can discover and query it at:

```
https://agent-search-engine-production.up.railway.app/
```

Its own agent card is published at:

```
https://agent-search-engine-production.up.railway.app/.well-known/agent.json
```

### Discover the engine's capabilities

```bash
curl https://agent-search-engine-production.up.railway.app/.well-known/agent.json
```

### Send a search query via A2A JSON-RPC

```bash
curl -X POST https://agent-search-engine-production.up.railway.app/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "req-001",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [
          {
            "kind": "text",
            "text": "Find me an agent that can summarise PDF documents"
          }
        ],
        "messageId": "msg-001"
      }
    }
  }'
```

The response is a standard A2A task result with a `data` artifact containing matching agents and their full cards:

```json
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "result": {
    "id": "task-...",
    "status": { "state": "completed" },
    "artifacts": [
      {
        "parts": [
          {
            "kind": "data",
            "data": {
              "results": [
                {
                  "id": "myorg__pdf-summariser",
                  "score": 0.847,
                  "agent_card": { ... }
                }
              ]
            }
          }
        ]
      }
    ]
  }
}
```

### A2A methods supported

| Method | What it does |
|---|---|
| `message/send` | Run a search query; returns matching agents |
| `tasks/get` | Retrieve a previous task result by ID |
| `tasks/cancel` | Cancel a running task |
| `ping` | Check that the engine is alive |

---

## Tips for Better Search Results

**Write descriptive agent cards.** The search engine embeds the agent's name, description, skill names, skill descriptions, and tags into a dense vector. The more precise and specific the language, the better it surfaces in relevant searches.

**Add examples to skills.** The `examples` field in each skill lets you add sample queries that should find this agent. These are embedded and searched alongside the description.

**Use specific tags.** Tags are used for hard filtering (`tags` param in `/search`). Use lowercase, single-word or hyphenated tags — e.g. `nlp`, `pdf`, `code-generation`, `vision`.

**Re-register to update.** If you update your agent card, just re-register with the same `humanReadableId`. The engine diffs the old and new card, saves a version snapshot, and re-indexes with fresh embeddings.

---

## Rate Limits

The API enforces per-IP rate limits:

| Endpoint | Limit |
|---|---|
| `POST /search` | 30 requests / minute |
| `POST /` (A2A) | 30 requests / minute |
| `GET /agents` | 20 requests / minute |
| `GET /agents/{id}` | 60 requests / minute |
| `POST /agents/{id}/click` | 60 requests / minute |
| `POST /agents/{id}/review` | 10 requests / minute |

Exceeding a limit returns `HTTP 429 Too Many Requests`.

---

## Quick Reference

| What you want | How |
|---|---|
| Search for an agent (UI) | `…/ui/` → type in search bar |
| Search for an agent (API) | `POST /search` with `{"query": "...", "top_k": 5}` |
| Search from another agent (A2A) | `POST /` with `message/send` JSON-RPC |
| Register your agent | `POST /register` with `X-Api-Key` header |
| See all agents | `GET /agents` |
| Get one agent's details | `GET /agents/{id}` |
| See an agent's card history | `GET /agents/{id}/history` |
| Leave a review | `POST /agents/{id}/review` |
| Read reviews | `GET /agents/{id}/reviews` |
| Remove an agent | `DELETE /agents/{id}` with `X-Api-Key` |
| View analytics | `GET /analytics` with `X-Api-Key` |
| Interactive API docs | `…/docs#/` |
| This engine's own A2A card | `GET /.well-known/agent.json` |
