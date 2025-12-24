"""FastMCP server for engram - a portable memory graph for Claude Code interactions."""

import logging

from fastmcp import FastMCP
from fastmcp.server.auth.providers.github import GitHubProvider
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from app.config import settings

# Suppress noisy MCP streamable_http ClosedResourceError logs (known issue with stateless mode)
# See: https://github.com/modelcontextprotocol/python-sdk/issues/1658
logging.getLogger("mcp.server.streamable_http").setLevel(logging.CRITICAL)
from app.tools import (
    memory_feedback,
    memory_observe,
    memory_reflect,
    memory_reflect_status,
    memory_retrieve,
    memory_retrieve_debug,
    memory_stats,
)

# Configure GitHub OAuth
auth = GitHubProvider(
    client_id=settings.github_client_id,
    client_secret=settings.github_client_secret,
    base_url=f"http://localhost:{settings.engram_port}",
)

# Initialize FastMCP server with auth (stateless for HMR compatibility)
mcp = FastMCP("engram", auth=auth, stateless_http=True, json_response=True)


# Health check endpoint
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    """Health check endpoint for container orchestration."""
    return PlainTextResponse("OK")


# Internal API endpoints for hook scripts (unauthenticated, localhost only)
# These bypass OAuth for local automation while MCP tools require auth

from starlette.responses import JSONResponse


def _check_localhost(request: Request) -> bool:
    """Verify request is from localhost."""
    client_host = request.client.host if request.client else None
    return client_host in ("127.0.0.1", "localhost", "::1")


@mcp.custom_route("/internal/observe", methods=["POST"])
async def internal_observe(request: Request) -> JSONResponse:
    """Internal endpoint for hook scripts to store observations."""
    if not _check_localhost(request):
        return JSONResponse({"error": "Forbidden: localhost only"}, status_code=403)

    try:
        data = await request.json()
        result = await memory_observe(
            content=data.get("content", ""),
            user_id=data.get("user_id", "default"),
            observation_type=data.get("observation_type", "general"),
            importance=data.get("importance"),
            section=data.get("section"),
            metadata=data.get("metadata"),
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


def _convert_decimals(obj):
    """Recursively convert Decimal values to floats for JSON serialization."""
    from decimal import Decimal
    if isinstance(obj, dict):
        return {k: _convert_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_decimals(v) for v in obj]
    elif isinstance(obj, Decimal):
        return float(obj)
    return obj


@mcp.custom_route("/internal/retrieve", methods=["POST"])
async def internal_retrieve(request: Request) -> JSONResponse:
    """Internal endpoint for hook scripts to retrieve memories."""
    if not _check_localhost(request):
        return JSONResponse({"error": "Forbidden: localhost only"}, status_code=403)

    try:
        data = await request.json()
        result = await memory_retrieve(
            query=data.get("query", ""),
            user_id=data.get("user_id", "default"),
            limit=data.get("limit", 10),
            memory_types=data.get("memory_types"),
            min_importance=data.get("min_importance", 0.0),
            include_synthetic=data.get("include_synthetic", True),
            recency_weight=data.get("recency_weight", 0.33),
            importance_weight=data.get("importance_weight", 0.33),
            relevance_weight=data.get("relevance_weight", 0.33),
            observation_types=data.get("observation_types"),
            sections=data.get("sections"),
            created_after=data.get("created_after"),
            created_before=data.get("created_before"),
            min_helpful_count=data.get("min_helpful_count", 0),
            max_harmful_count=data.get("max_harmful_count"),
        )
        return JSONResponse(_convert_decimals(result))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/internal/stats", methods=["POST"])
async def internal_stats(request: Request) -> JSONResponse:
    """Internal endpoint for hook scripts to get stats."""
    if not _check_localhost(request):
        return JSONResponse({"error": "Forbidden: localhost only"}, status_code=403)

    try:
        data = await request.json()
        result = await memory_stats(
            user_id=data.get("user_id", "default"),
            scope=data.get("scope", "summary"),
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# Register MCP tools
@mcp.tool()
async def observe(
    content: str,
    user_id: str = "default",
    observation_type: str = "general",
    importance: float | None = None,
    section: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Ingest an observation into the memory stream.

    Observations are append-only with automatic deduplication.
    Duplicate observations (>90% cosine similarity) are rejected.

    Args:
        content: The observation content to store.
        user_id: User identifier (default: "default").
        observation_type: Type of observation for importance heuristics.
            Options: error, instruction, decision, code_change, insight,
            test_result, general, tool_output.
        importance: Override importance score (1-10). If not provided,
            determined by observation_type heuristics.
        section: ACE-style section classification.
            Options: strategies, snippets, pitfalls, context, preferences.
        metadata: Optional extra metadata as key-value pairs.

    Returns:
        dict with status ("created" or "deduplicated"), memory_id if created,
        and importance_score assigned.
    """
    return await memory_observe(
        content=content,
        user_id=user_id,
        observation_type=observation_type,
        importance=importance,
        section=section,
        metadata=metadata,
    )


@mcp.tool()
async def retrieve(
    query: str,
    user_id: str = "default",
    limit: int = 10,
    memory_types: list[str] | None = None,
    min_importance: float = 0.0,
    include_synthetic: bool = True,
    recency_weight: float = 0.33,
    importance_weight: float = 0.33,
    relevance_weight: float = 0.33,
    observation_types: list[str] | None = None,
    sections: list[str] | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    min_helpful_count: int = 0,
    max_harmful_count: int | None = None,
) -> dict:
    """Retrieve memories using three-factor scoring.

    Combines recency, importance, and semantic relevance to rank results.
    Scoring formula: score = α*recency + β*importance + γ*relevance

    Args:
        query: Search query text for semantic matching.
        user_id: User identifier (default: "default").
        limit: Maximum number of results (default: 10).
        memory_types: Filter by memory types. Options: episodic, semantic, procedural.
        min_importance: Minimum importance score filter (default: 0.0).
        include_synthetic: Include reflected/synthesized memories (default: True).
        recency_weight: Weight for recency factor α (default: 0.33).
        importance_weight: Weight for importance factor β (default: 0.33).
        relevance_weight: Weight for relevance factor γ (default: 0.33).
        observation_types: Filter by observation types.
            Options: error, instruction, decision, code_change, insight, test_result, general, tool_output.
        sections: Filter by ACE sections.
            Options: strategies, snippets, pitfalls, context, preferences.
        created_after: Only include memories created after this ISO datetime.
        created_before: Only include memories created before this ISO datetime.
        min_helpful_count: Minimum helpful feedback count (default: 0).
        max_harmful_count: Maximum harmful feedback count. If None, no limit.

    Returns:
        dict with memories list, count, and weights used.
    """
    return await memory_retrieve(
        query=query,
        user_id=user_id,
        limit=limit,
        memory_types=memory_types,
        min_importance=min_importance,
        include_synthetic=include_synthetic,
        recency_weight=recency_weight,
        importance_weight=importance_weight,
        relevance_weight=relevance_weight,
        observation_types=observation_types,
        sections=sections,
        created_after=created_after,
        created_before=created_before,
        min_helpful_count=min_helpful_count,
        max_harmful_count=max_harmful_count,
    )


@mcp.tool()
async def retrieve_debug(
    query: str,
    user_id: str = "default",
    limit: int = 10,
    memory_types: list[str] | None = None,
    min_importance: float = 0.0,
    include_synthetic: bool = True,
    recency_weight: float = 0.33,
    importance_weight: float = 0.33,
    relevance_weight: float = 0.33,
    observation_types: list[str] | None = None,
    sections: list[str] | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    min_helpful_count: int = 0,
    max_harmful_count: int | None = None,
) -> dict:
    """Debug retrieval scoring for a query.

    Returns detailed breakdown of how each memory was scored,
    including raw scores before normalization and filter effects.
    Useful for understanding why certain memories are or aren't retrieved.

    Args:
        query: Search query text for semantic matching.
        user_id: User identifier (default: "default").
        limit: Maximum number of results (default: 10).
        memory_types: Filter by memory types. Options: episodic, semantic, procedural.
        min_importance: Minimum importance score filter (default: 0.0).
        include_synthetic: Include reflected/synthesized memories (default: True).
        recency_weight: Weight for recency factor α (default: 0.33).
        importance_weight: Weight for importance factor β (default: 0.33).
        relevance_weight: Weight for relevance factor γ (default: 0.33).
        observation_types: Filter by observation types.
        sections: Filter by ACE sections.
        created_after: Only include memories created after this ISO datetime.
        created_before: Only include memories created before this ISO datetime.
        min_helpful_count: Minimum helpful feedback count (default: 0).
        max_harmful_count: Maximum harmful feedback count. If None, no limit.

    Returns:
        dict with query, embedding preview, candidate counts, weights,
        score_ranges, detailed results with raw/normalized scores,
        and examples of filtered-out memories with reasons.
    """
    return await memory_retrieve_debug(
        query=query,
        user_id=user_id,
        limit=limit,
        memory_types=memory_types,
        min_importance=min_importance,
        include_synthetic=include_synthetic,
        recency_weight=recency_weight,
        importance_weight=importance_weight,
        relevance_weight=relevance_weight,
        observation_types=observation_types,
        sections=sections,
        created_after=created_after,
        created_before=created_before,
        min_helpful_count=min_helpful_count,
        max_harmful_count=max_harmful_count,
    )


@mcp.tool()
async def feedback(
    memory_id: str,
    helpful: bool,
    reason: str | None = None,
) -> dict:
    """Provide helpful/harmful feedback on a memory.

    Feedback adjusts the effective importance of memories over time.
    Effective importance = base_importance + (helpful_count - harmful_count) * 0.5

    Args:
        memory_id: UUID of the memory to provide feedback on.
        helpful: True if the memory was helpful, False if harmful.
        reason: Optional reason for the feedback.

    Returns:
        dict with status, updated counts, and effective importance.
    """
    return await memory_feedback(
        memory_id=memory_id,
        helpful=helpful,
        reason=reason,
    )


@mcp.tool()
async def reflect(
    user_id: str = "default",
    focus: str | None = None,
    max_insights: int = 5,
    force: bool = False,
) -> dict:
    """Synthesize episodic memories into semantic insights.

    Reflection is triggered automatically based on accumulated importance,
    observation count, or time since last reflection.
    Use force=True to trigger reflection regardless of thresholds.

    Args:
        user_id: User identifier (default: "default").
        focus: Optional focus area to guide reflection.
        max_insights: Maximum number of insights to generate (default: 5).
        force: Force reflection even if triggers not met (default: False).

    Returns:
        dict with status, insights created, and memories analyzed.
    """
    return await memory_reflect(
        user_id=user_id,
        focus=focus,
        max_insights=max_insights,
        force=force,
    )


@mcp.tool()
async def reflect_status(
    job_id: str,
    user_id: str = "default",
) -> dict:
    """Get the status of a background reflection job."""
    return await memory_reflect_status(job_id=job_id, user_id=user_id)


@mcp.tool()
async def stats(
    user_id: str = "default",
    scope: str = "summary",
) -> dict:
    """Get memory system statistics and health metrics.

    Args:
        user_id: User identifier (default: "default").
        scope: Level of detail. Options: summary, detailed, reflection.

    Returns:
        dict with statistics based on scope.
    """
    return await memory_stats(
        user_id=user_id,
        scope=scope,
    )


# ASGI app for uvicorn (used by start_dev_server.sh)
app = mcp.http_app()

if __name__ == "__main__":
    mcp.run(
        transport="http",
        host=settings.engram_host,
        port=settings.engram_port,
        stateless_http=True,
    )
