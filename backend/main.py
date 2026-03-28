"""
Agent Search Engine — FastAPI application

REST endpoints:
  GET  /.well-known/agent.json        This engine's own A2A agent card
  POST /                              A2A JSON-RPC 2.0 endpoint
  POST /register                      Register an agent card  [API key required]
  GET  /agents                        List all registered agents (paginated)
  GET  /agents/{id}                   Get a single agent card + health status
  GET  /agents/{id}/history           Agent card version history
  DELETE /agents/{id}                 Remove an agent  [API key required]
  POST /agents/{id}/click             Log an agent click (analytics)
  POST /search                        Semantic search (REST convenience)
  GET  /analytics                     Search & click analytics  [API key required]
  GET  /health                        Health check
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
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
    delete_stale_agents,
    diff_cards,
    get_agent,
    get_agent_raw,
    get_agent_versions,
    get_agent_with_health,
    get_all_agent_urls_and_ids,
    get_analytics,
    init_db,
    list_all_agents,
    list_all_agents_with_health,
    log_agent_click,
    log_search,
    make_agent_id,
    save_agent_version,
    update_agent_health,
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
# Config from environment
# ---------------------------------------------------------------------------

_REGISTRY_API_KEY: str | None = os.environ.get("REGISTRY_API_KEY")
HEALTH_CHECK_INTERVAL: int = int(os.environ.get("HEALTH_CHECK_INTERVAL_SECONDS", 300))
STALE_AFTER_DAYS: int = int(os.environ.get("STALE_AFTER_DAYS", 3))
DEREGISTER_AFTER_DAYS: int = int(os.environ.get("DEREGISTER_AFTER_DAYS", 7))

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

def require_api_key(x_api_key: str = Header(default=None)):
    if _REGISTRY_API_KEY and x_api_key != _REGISTRY_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# Health monitor background task
# ---------------------------------------------------------------------------

async def _health_monitor_loop() -> None:
    """Ping every registered agent periodically, update status, auto-deregister stale ones."""
    while True:
        await asyncio.sleep(HEALTH_CHECK_INTERVAL)
        try:
            now = datetime.now(timezone.utc)
            deregister_cutoff = (now - timedelta(days=DEREGISTER_AFTER_DAYS)).isoformat()
            stale_cutoff = (now - timedelta(days=STALE_AFTER_DAYS)).isoformat()

            deleted_ids = delete_stale_agents(deregister_cutoff)
            for aid in deleted_ids:
                logger.info("Auto-deregistered stale agent: %s", aid)

            agents = get_all_agent_urls_and_ids()
            deleted_set = set(deleted_ids)

            async with httpx.AsyncClient(timeout=5.0) as client:
                for agent in agents:
                    if agent["id"] in deleted_set or not agent["url"]:
                        continue
                    now_iso = datetime.now(timezone.utc).isoformat()
                    try:
                        resp = await client.get(agent["url"])
                        is_online = resp.status_code < 500
                    except Exception:
                        is_online = False

                    last_seen = agent["last_seen_online"]
                    if is_online:
                        new_status = "online"
                        new_last_seen = now_iso
                    else:
                        new_last_seen = last_seen
                        if last_seen and last_seen < stale_cutoff:
                            new_status = "stale"
                        else:
                            new_status = "offline"

                    update_agent_health(agent["id"], new_status, now_iso, new_last_seen)

            logger.info("Health check complete: %d agents checked, %d auto-deregistered.",
                        len(agents), len(deleted_ids))
        except Exception as exc:
            logger.error("Health monitor error: %s", exc)

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
    task = asyncio.create_task(_health_monitor_loop())
    logger.info(
        "Health monitor started (interval=%ds, stale=%dd, deregister=%dd).",
        HEALTH_CHECK_INTERVAL, STALE_AFTER_DAYS, DEREGISTER_AFTER_DAYS,
    )
    logger.info("Agent Search Engine ready.")
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


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

FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# ---------------------------------------------------------------------------
# A2A discovery + JSON-RPC
# ---------------------------------------------------------------------------

@app.get("/.well-known/agent.json", tags=["A2A"])
async def agent_card():
    return OWN_CARD.model_dump()


@app.post("/", tags=["A2A"])
@limiter.limit("30/minute")
async def a2a_jsonrpc(request: Request):
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
    Diffs against the existing card and saves a version snapshot if anything changed.
    Requires X-Api-Key header when REGISTRY_API_KEY is configured.
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

    if not card.humanReadableId:
        card.humanReadableId = card.name.lower().replace(" ", "-")
    if not card.provider:
        card.provider = AgentProvider(name="Unknown")
    if not card.agentVersion and card.version:
        card.agentVersion = card.version

    agent_id = make_agent_id(card)

    # Versioning: diff against existing card before overwriting
    try:
        old_card_dict = get_agent_raw(agent_id)
        embedding = embed_agent_card(card.model_dump())
        upsert_agent(agent_id, card, embedding)
        if old_card_dict is not None:
            diff = diff_cards(old_card_dict, card.model_dump())
            if diff:
                save_agent_version(agent_id, card.model_dump_json(), diff)
                logger.info("Saved version for agent %s (%d field(s) changed)", agent_id, len(diff))
    except Exception as exc:
        logger.error("Error during registration of %s: %s", agent_id, exc)
        raise HTTPException(status_code=500, detail="Registration failed")

    logger.info("Registered agent: %s (%s)", agent_id, card.name)
    return {"id": agent_id, "message": f"Agent '{card.name}' registered successfully."}


@app.get("/agents", tags=["Registry"], summary="List all registered agents")
@limiter.limit("20/minute")
async def list_agents(
    request: Request,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
):
    all_agents = list_all_agents_with_health()
    return {
        "total": len(all_agents),
        "skip": skip,
        "limit": limit,
        "agents": all_agents[skip : skip + limit],
    }


@app.get("/agents/{agent_id}/history", tags=["Registry"], summary="Agent card version history")
@limiter.limit("30/minute")
async def agent_history(request: Request, agent_id: str):
    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    versions = get_agent_versions(agent_id)
    return {"id": agent_id, "versions": versions}


@app.get("/agents/{agent_id}", tags=["Registry"], summary="Get a single agent card")
@limiter.limit("60/minute")
async def get_agent_endpoint(request: Request, agent_id: str):
    result = get_agent_with_health(agent_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return result


@app.delete(
    "/agents/{agent_id}",
    tags=["Registry"],
    summary="Remove an agent",
    dependencies=[Depends(require_api_key)],
)
async def remove_agent(agent_id: str):
    if not delete_agent(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"message": f"Agent '{agent_id}' removed."}


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

@app.post("/agents/{agent_id}/click", tags=["Analytics"], summary="Log an agent click")
@limiter.limit("60/minute")
async def click_agent(request: Request, agent_id: str):
    if get_agent(agent_id) is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    try:
        log_agent_click(agent_id)
    except Exception as exc:
        logger.warning("Failed to log click for %s: %s", agent_id, exc)
    return {"message": "click recorded"}


@app.get(
    "/analytics",
    tags=["Analytics"],
    summary="Search and click analytics",
    dependencies=[Depends(require_api_key)],
)
async def analytics():
    """Top queries, zero-result queries, and top clicked agents. API key required."""
    return get_analytics()


# ---------------------------------------------------------------------------
# Search REST API
# ---------------------------------------------------------------------------

@app.post("/search", tags=["Search"], response_model=SearchResponse)
@limiter.limit("30/minute")
async def search_agents(request: Request, req: SearchRequest):
    vec = embed_text(req.query)
    hits = cosine_search(vec, top_k=req.top_k, tag_filter=req.tags or None)

    results: list[AgentResult] = []
    for agent_id, score in hits:
        card = get_agent(agent_id)
        if card:
            results.append(AgentResult(id=agent_id, score=score, agent_card=card))

    try:
        log_search(req.query, len(results), req.tags or [])
    except Exception as exc:
        logger.warning("Failed to log search: %s", exc)

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
    reload = os.environ.get("RAILWAY_ENVIRONMENT") is None
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload)
