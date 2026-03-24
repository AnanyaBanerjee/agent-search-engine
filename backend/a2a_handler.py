"""
A2A JSON-RPC 2.0 handler.

The search engine itself is a valid A2A agent. It exposes a single skill:
  find_agents — given a natural-language task, return matching agents.

Other A2A agents send a message/send request; this handler parses it,
runs semantic search, and returns the results as a JSON data part.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from models import (
    A2AMessage,
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    MessagePart,
    MessageSendParams,
    TaskCancelParams,
    TaskGetParams,
)


# ---------------------------------------------------------------------------
# Errors (JSON-RPC codes)
# ---------------------------------------------------------------------------

PARSE_ERROR      = JSONRPCError(code=-32700, message="Parse error")
INVALID_REQUEST  = JSONRPCError(code=-32600, message="Invalid request")
METHOD_NOT_FOUND = JSONRPCError(code=-32601, message="Method not found")
INVALID_PARAMS   = JSONRPCError(code=-32602, message="Invalid params")
INTERNAL_ERROR   = JSONRPCError(code=-32603, message="Internal error")


def _err(req_id: Any, error: JSONRPCError) -> dict:
    return JSONRPCResponse(id=req_id, error=error).model_dump()


def _ok(req_id: Any, result: Any) -> dict:
    return JSONRPCResponse(id=req_id, result=result).model_dump()


# ---------------------------------------------------------------------------
# Task store (in-memory; good enough for short-lived queries)
# ---------------------------------------------------------------------------

_tasks: dict[str, Any] = {}


def _make_task(status: str, artifacts: list[dict] | None = None) -> dict:
    return {
        "taskId": str(uuid.uuid4()),
        "status": status,
        "artifacts": artifacts or [],
    }


# ---------------------------------------------------------------------------
# Core dispatch
# ---------------------------------------------------------------------------

def handle_jsonrpc(raw: dict, search_fn) -> dict:
    """
    Dispatch a JSON-RPC request.

    `search_fn(query: str, top_k: int) -> list[dict]`
    should return a list of agent result dicts.
    """
    # Validate envelope
    try:
        req = JSONRPCRequest.model_validate(raw)
    except Exception:
        return _err(None, INVALID_REQUEST)

    method = req.method
    params = req.params or {}
    req_id = req.id

    if method == "message/send":
        return _handle_message_send(req_id, params, search_fn)

    if method == "tasks/get":
        return _handle_tasks_get(req_id, params)

    if method == "tasks/cancel":
        return _handle_tasks_cancel(req_id, params)

    if method == "ping":
        return _ok(req_id, {"status": "ok"})

    return _err(req_id, METHOD_NOT_FOUND)


# ---------------------------------------------------------------------------
# message/send
# ---------------------------------------------------------------------------

def _handle_message_send(req_id: Any, params: dict, search_fn) -> dict:
    try:
        p = MessageSendParams.model_validate(params)
    except Exception as exc:
        return _err(req_id, JSONRPCError(code=-32602, message=str(exc)))

    msg = p.message

    # Extract the user's text query
    query = _extract_text(msg)
    if not query:
        return _err(
            req_id,
            JSONRPCError(code=-32602, message="No text content found in message parts"),
        )

    # Run semantic search
    try:
        results = search_fn(query=query, top_k=5)
    except Exception as exc:
        return _err(req_id, JSONRPCError(code=-32603, message=str(exc)))

    # Build response artifact
    artifact = {
        "artifactId": str(uuid.uuid4()),
        "name": "search_results",
        "parts": [
            {
                "kind": "text",
                "text": _format_results_text(query, results),
            },
            {
                "kind": "data",
                "data": {
                    "query": query,
                    "results": results,
                },
            },
        ],
    }

    task = _make_task("completed", [artifact])
    task["contextId"] = msg.contextId
    task["messageId"] = msg.messageId
    _tasks[task["taskId"]] = task

    return _ok(req_id, task)


def _extract_text(msg: A2AMessage) -> str:
    for part in msg.parts:
        if part.kind == "text" and part.text:
            return part.text.strip()
    return ""


def _format_results_text(query: str, results: list[dict]) -> str:
    if not results:
        return f'No agents found for: "{query}"'
    lines = [f'Found {len(results)} agent(s) for: "{query}"\n']
    for i, r in enumerate(results, 1):
        card = r.get("agent_card", {})
        lines.append(
            f"{i}. [{card.get('name', 'Unknown')}] "
            f"(score: {r.get('score', 0):.3f})\n"
            f"   {card.get('description', '')}\n"
            f"   Endpoint: {card.get('url', 'N/A')}\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# tasks/get
# ---------------------------------------------------------------------------

def _handle_tasks_get(req_id: Any, params: dict) -> dict:
    try:
        p = TaskGetParams.model_validate(params)
    except Exception as exc:
        return _err(req_id, JSONRPCError(code=-32602, message=str(exc)))

    task = _tasks.get(p.taskId)
    if task is None:
        return _err(req_id, JSONRPCError(code=-32001, message="Task not found"))
    return _ok(req_id, task)


# ---------------------------------------------------------------------------
# tasks/cancel
# ---------------------------------------------------------------------------

def _handle_tasks_cancel(req_id: Any, params: dict) -> dict:
    try:
        p = TaskCancelParams.model_validate(params)
    except Exception as exc:
        return _err(req_id, JSONRPCError(code=-32602, message=str(exc)))

    if p.taskId in _tasks:
        _tasks[p.taskId]["status"] = "cancelled"
    return _ok(req_id, {"taskId": p.taskId, "status": "cancelled"})
