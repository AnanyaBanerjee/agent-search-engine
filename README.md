# Agent Search Engine

A2A-compatible registry and **semantic search engine** for AI agents.

Agents register their [A2A Agent Card](https://a2a-protocol.org) JSON and the engine indexes it with sentence-transformers embeddings. Any other agent (or human) can then query the registry in natural language to find the best agent for a task.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Agent Search Engine                       │
│                                                             │
│   /.well-known/agent.json  ←  own A2A agent card           │
│   POST /                   ←  A2A JSON-RPC 2.0 endpoint    │
│   POST /register           ←  submit an agent card         │
│   POST /search             ←  semantic REST search          │
│   GET  /agents             ←  list all agents               │
│   GET  /ui                 ←  web UI                        │
└─────────────────────────────────────────────────────────────┘
        ↑ agents call via A2A JSON-RPC
        ↑ humans browse via web UI
```

**Stack**
- **Backend**: FastAPI + uvicorn (Python)
- **Storage**: SQLite (agents.db)
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Search**: Cosine similarity (numpy, in-memory)
- **Protocol**: A2A v1.0 (JSON-RPC 2.0 over HTTP)

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

---

## Registering an Agent

### Via REST
```bash
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{"agent_card": '"$(cat example-agents/pdf-summariser.json)"'}'
```

### Via the Web UI
Open `http://localhost:8000/ui` → click **"Register a new agent"** → paste your agent card JSON.

### By URL (agent fetches its own card)
```bash
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{"agent_card": {"schemaVersion":"1.0","humanReadableId":"x/y","name":"x","description":"","url":"https://my-agent.com","agentVersion":"1.0.0","provider":{"name":"x"},"capabilities":{"a2aVersion":"1.0"},"authSchemes":[{"type":"none"}]}, "card_url": "https://my-agent.com"}'
```

---

## Searching (A2A — agent-to-agent)

Any A2A-compatible agent can discover agents by sending a `message/send` request:

```bash
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{ "kind": "text", "text": "Find me an agent that can summarise PDF documents" }],
        "messageId": "msg-001"
      }
    }
  }'
```

The response includes a `data` artifact with matching agents and their full A2A cards.

---

## Searching (REST)

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "summarise PDF documents", "top_k": 5}'
```

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

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/.well-known/agent.json` | This engine's own A2A agent card |
| `POST` | `/` | A2A JSON-RPC 2.0 endpoint |
| `POST` | `/register` | Register an agent card |
| `GET` | `/agents` | List all registered agents |
| `GET` | `/agents/{id}` | Get a single agent card |
| `DELETE` | `/agents/{id}` | Remove an agent |
| `POST` | `/search` | Semantic search (REST) |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI |

## A2A Methods Supported

| Method | Description |
|--------|-------------|
| `message/send` | Send a task query, get matching agents back |
| `tasks/get` | Retrieve a previous task result |
| `tasks/cancel` | Cancel a task |
| `ping` | Health check |
