"""
Agent Search Engine — FastAPI application

REST endpoints:
  GET  /.well-known/agent.json        This engine's own A2A agent card
  POST /                              A2A JSON-RPC 2.0 endpoint
  POST /register                      Register an agent card  [API key required]
  GET  /agents                        List all registered agents (paginated)
  GET  /agents/{id}                   Get a single agent card
  DELETE /agents/{id}                 Remove an agent  [API key required]
  POST /search                        Semantic search (REST convenience)
  GET  /health                        Health check
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from a2a_handler import handle_jsonrpc
from database import (
    cosine_search,
    delete_agent,
    get_agent,
    init_db,
    list_all_agents,
    make_agent_id,
    upsert_agent,
)
from models import (
    AgentCapabilities,
    AgentCard,
    AgentProvider,
    AgentResult,
    AgentSkill,
    AuthScheme,
    RegisterRequest,
    SearchRequest,
    SearchResponse,
)
from search import embed_agent_card, embed_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

# Set REGISTRY_API_KEY in your environment / Railway env vars.
# If unset, write endpoints are open (convenient for local dev).
_REGISTRY_API_KEY: str | None = os.environ.get("REGISTRY_API_KEY")


def require_api_key(x_api_key: str = Header(default=None)):
    """Dependency — rejects requests that don't carry the correct API key."""
    if _REGISTRY_API_KEY and x_api_key != _REGISTRY_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# This search engine's own A2A agent card
# ---------------------------------------------------------------------------

OWN_CARD = AgentCard(
    schemaVersion="1.0",
    humanReadableId="agent-search-engine/registry",
    name="Agent Search Engine",
    description=(
        "A registry and semantic search engine for A2A-compatible agents. "
        "Given a natural-language task description, returns the most relevant "
        "registered agents ranked by capability match."
    ),
    url="http://localhost:8000",
    agentVersion="1.0.0",
    provider=AgentProvider(name="Agent Search Engine", url="http://localhost:8000"),
    capabilities=AgentCapabilities(a2aVersion="1.0", streaming=False),
    authSchemes=[AuthScheme(type="apiKey")],
    defaultInputModes=["text/plain"],
    defaultOutputModes=["text/plain", "application/json"],
    skills=[
        AgentSkill(
            id="find_agents",
            name="Find Agents",
            description=(
                "Given a natural-language task or query, semantically searches "
                "the registry and returns the best-matching agents with their "
                "A2A endpoints and capability details."
            ),
            tags=["search", "discovery", "registry", "a2a"],
            examples=[
                "Find me an agent that can summarise PDF documents",
                "Which agents can write Python code?",
                "I need an agent for sentiment analysis",
            ],
        ),
        AgentSkill(
            id="register_agent",
            name="Register Agent",
            description="Register a new agent card in the registry via the REST API.",
            tags=["register", "registry"],
        ),
    ],
    tags=["search", "discovery", "registry", "a2a", "meta"],
    documentationUrl="http://localhost:8000/docs",
)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialising database …")
    init_db()
    logger.info("Pre-loading embedding model …")
    from search import _get_model
    _get_model()
    if _REGISTRY_API_KEY:
        logger.info("Registry write endpoints are protected by API key.")
    else:
        logger.warning("REGISTRY_API_KEY is not set — write endpoints are open.")
    logger.info("Agent Search Engine ready.")
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agent Search Engine",
    version="1.0.0",
    description="A2A-compatible registry and semantic search for AI agents.",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# ---------------------------------------------------------------------------
# A2A discovery + JSON-RPC
# ---------------------------------------------------------------------------

@app.get("/.well-known/agent.json", tags=["A2A"])
async def agent_card():
    """Return this engine's own A2A agent card."""
    return OWN_CARD.model_dump()


@app.post("/", tags=["A2A"])
@limiter.limit("30/minute")
async def a2a_jsonrpc(request: Request):
    """
    A2A JSON-RPC 2.0 endpoint.
    Other agents send message/send, tasks/get, tasks/cancel, or ping here.
    """
    try:
        raw = await request.json()
    except Exception:
        from a2a_handler import PARSE_ERROR, _err
        return JSONResponse(_err(None, PARSE_ERROR))

    def search_fn(query: str, top_k: int = 5) -> list[dict]:
        vec = embed_text(query)
        hits = cosine_search(vec, top_k=top_k)
        results = []
        for agent_id, score in hits:
            card = get_agent(agent_id)
            if card:
                results.append(
                    {"id": agent_id, "score": score, "agent_card": card.model_dump()}
                )
        return results

    response = handle_jsonrpc(raw, search_fn)
    return JSONResponse(response)


# ---------------------------------------------------------------------------
# Registry REST API
# ---------------------------------------------------------------------------

@app.post(
    "/register",
    tags=["Registry"],
    summary="Register an agent card",
    dependencies=[Depends(require_api_key)],
)
async def register_agent(req: RegisterRequest):
    """
    Submit an A2A agent card to be indexed.

    Either supply `agent_card` directly, or provide `card_url` and the engine
    will fetch the card from `<card_url>/.well-known/agent.json`.

    Requires `X-Api-Key` header when `REGISTRY_API_KEY` is configured.
    """
    card = req.agent_card

    if req.card_url:
        url = req.card_url.rstrip("/") + "/.well-known/agent.json"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                card = AgentCard.model_validate(resp.json())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to fetch agent card: {exc}")

    # Fill optional fields that agents commonly omit
    if not card.humanReadableId:
        card.humanReadableId = card.name.lower().replace(" ", "-")
    if not card.provider:
        card.provider = AgentProvider(name="Unknown")
    if not card.agentVersion and card.version:
        card.agentVersion = card.version

    agent_id = make_agent_id(card)
    embedding = embed_agent_card(card.model_dump())
    upsert_agent(agent_id, card, embedding)

    logger.info("Registered agent: %s (%s)", agent_id, card.name)
    return {"id": agent_id, "message": f"Agent '{card.name}' registered successfully."}


@app.get("/agents", tags=["Registry"], summary="List all registered agents")
@limiter.limit("20/minute")
async def list_agents(
    request: Request,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
):
    all_agents = list_all_agents()
    return {
        "total": len(all_agents),
        "skip": skip,
        "limit": limit,
        "agents": all_agents[skip : skip + limit],
    }


@app.get("/agents/{agent_id}", tags=["Registry"], summary="Get a single agent card")
@limiter.limit("60/minute")
async def get_agent_endpoint(request: Request, agent_id: str):
    card = get_agent(agent_id)
    if card is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"id": agent_id, "agent_card": card.model_dump()}


@app.delete(
    "/agents/{agent_id}",
    tags=["Registry"],
    summary="Remove an agent",
    dependencies=[Depends(require_api_key)],
)
async def remove_agent(agent_id: str):
    """Requires `X-Api-Key` header when `REGISTRY_API_KEY` is configured."""
    if not delete_agent(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"message": f"Agent '{agent_id}' removed."}


# ---------------------------------------------------------------------------
# Search REST API
# ---------------------------------------------------------------------------

@app.post("/search", tags=["Search"], response_model=SearchResponse)
@limiter.limit("30/minute")
async def search_agents(request: Request, req: SearchRequest):
    """
    Semantic search over the agent registry.
    Returns the top-k most relevant agents for the given task description.
    """
    vec = embed_text(req.query)
    hits = cosine_search(vec, top_k=req.top_k, tag_filter=req.tags or None)

    results: list[AgentResult] = []
    for agent_id, score in hits:
        card = get_agent(agent_id)
        if card:
            results.append(AgentResult(id=agent_id, score=score, agent_card=card))

    return SearchResponse(query=req.query, results=results)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
async def health():
    return {"status": "ok", "agents": len(list_all_agents())}


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    reload = os.environ.get("RAILWAY_ENVIRONMENT") is None  # reload only in local dev
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload)
