"""
A2A Protocol Pydantic models for Agent Cards and JSON-RPC messages.
Based on the A2A specification v1.0 (https://a2a-protocol.org)
"""
from __future__ import annotations
from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Agent Card schema
# ---------------------------------------------------------------------------

class AgentProvider(BaseModel):
    name: str
    url: Optional[str] = None
    email: Optional[str] = None


class AgentCapabilities(BaseModel):
    a2aVersion: str = "1.0"
    streaming: bool = False
    pushNotifications: bool = False
    stateTransitionHistory: bool = False


class AuthScheme(BaseModel):
    type: Literal["none", "apiKey", "bearer", "oauth2", "openIdConnect"]
    description: Optional[str] = None
    tokenUrl: Optional[str] = None
    scopes: Optional[list[str]] = None


class AgentSkill(BaseModel):
    id: str
    name: str
    description: str
    tags: list[str] = []
    inputModes: Optional[list[str]] = None
    outputModes: Optional[list[str]] = None
    examples: list[str] = []


class AgentCard(BaseModel):
    """Full A2A Agent Card as published at /.well-known/agent.json"""
    schemaVersion: str = "1.0"
    humanReadableId: Optional[str] = None       # auto-generated from name if omitted
    name: str
    description: str
    url: str                                    # A2A JSON-RPC endpoint
    agentVersion: str = "1.0.0"
    version: Optional[str] = None              # alias some agents use instead of agentVersion
    provider: Optional[AgentProvider] = None
    capabilities: AgentCapabilities = AgentCapabilities()
    authSchemes: list[AuthScheme] = [AuthScheme(type="none")]
    defaultInputModes: list[str] = ["text/plain"]
    defaultOutputModes: list[str] = ["text/plain"]
    skills: list[AgentSkill] = []
    tags: list[str] = []
    documentationUrl: Optional[str] = None
    lastUpdated: Optional[str] = None          # ISO-8601


# ---------------------------------------------------------------------------
# A2A JSON-RPC 2.0 wire types
# ---------------------------------------------------------------------------

class MessagePart(BaseModel):
    kind: Literal["text", "data", "file"] = "text"
    text: Optional[str] = None
    data: Optional[dict[str, Any]] = None
    mimeType: Optional[str] = None


class A2AMessage(BaseModel):
    role: Literal["user", "agent"]
    parts: list[MessagePart]
    messageId: str
    contextId: Optional[str] = None
    taskId: Optional[str] = None


class MessageSendParams(BaseModel):
    message: A2AMessage


class TaskGetParams(BaseModel):
    taskId: str
    messageCount: Optional[int] = None


class TaskCancelParams(BaseModel):
    taskId: str


class JSONRPCRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[Union[str, int]] = None
    method: str
    params: Optional[dict[str, Any]] = None


class JSONRPCError(BaseModel):
    code: int
    message: str
    data: Optional[Any] = None


class JSONRPCResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[Union[str, int]] = None
    result: Optional[Any] = None
    error: Optional[JSONRPCError] = None


# ---------------------------------------------------------------------------
# API-layer request / response models
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    agent_card: AgentCard
    card_url: Optional[str] = Field(
        default=None,
        description="If set, the engine will fetch the card from this URL."
    )


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Natural-language task description")
    top_k: int = Field(default=5, ge=1, le=20)
    tags: list[str] = []


class AgentResult(BaseModel):
    id: str
    score: float
    agent_card: AgentCard


class SearchResponse(BaseModel):
    query: str
    results: list[AgentResult]


class ClickRequest(BaseModel):
    query: Optional[str] = Field(
        default=None,
        description="The search query that surfaced this agent (used to update task affinity)."
    )


class ReviewRequest(BaseModel):
    reviewer_id: str = Field(..., min_length=1, max_length=200, description="Unique identifier for the reviewer")
    score: int = Field(..., ge=1, le=5, description="Rating from 1 (worst) to 5 (best)")
    comment: str = Field(default="", max_length=2000)


class ReviewResponse(BaseModel):
    agent_id: str
    reviewer_id: str
    score: int
    comment: str
    created_at: str
