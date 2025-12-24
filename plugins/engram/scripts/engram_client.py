#!/usr/bin/env python3
"""Shared HTTP client for engram MCP server.

This module provides a simple interface to call engram MCP tools via HTTP.
Used by hook scripts to observe, retrieve, and check server health.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error
from typing import Any

# Configuration
# Use internal API (unauthenticated, localhost only) for hook scripts
ENGRAM_BASE_URL = os.environ.get("ENGRAM_URL", "http://localhost:8787")
ENGRAM_TIMEOUT = int(os.environ.get("ENGRAM_TIMEOUT", "5"))


class EngramError(Exception):
    """Raised when engram server returns an error."""
    pass


class EngramUnavailable(Exception):
    """Raised when engram server is not reachable."""
    pass


def _call_internal_api(endpoint: str, data: dict[str, Any]) -> dict:
    """Call internal API endpoint via HTTP POST.

    Args:
        endpoint: API endpoint path (e.g., "/internal/observe")
        data: Request body as a dict

    Returns:
        Response as a dict

    Raises:
        EngramUnavailable: If server is not reachable
        EngramError: If server returns an error
    """
    url = f"{ENGRAM_BASE_URL}{endpoint}"
    payload = json.dumps(data).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=ENGRAM_TIMEOUT) as response:
            result = json.loads(response.read().decode("utf-8"))

            if "error" in result:
                raise EngramError(result["error"])

            return result

    except urllib.error.HTTPError as e:
        if e.code == 403:
            raise EngramError("Forbidden: Internal API is localhost only")
        raise EngramUnavailable(f"HTTP error from engram server: {e}")
    except urllib.error.URLError as e:
        raise EngramUnavailable(f"Cannot connect to engram server at {ENGRAM_BASE_URL}: {e}")
    except json.JSONDecodeError as e:
        raise EngramError(f"Invalid JSON response from server: {e}")


def check_health() -> bool:
    """Check if engram server is available.

    Returns:
        True if server is healthy

    Raises:
        EngramUnavailable: If server is not reachable (with clear message)
    """
    try:
        # Try calling stats with minimal scope to check health
        _call_internal_api("/internal/stats", {"scope": "summary"})
        return True
    except EngramUnavailable:
        raise
    except EngramError:
        # Server responded but with an error - still "available"
        return True


def observe(
    content: str,
    observation_type: str = "general",
    importance: float | None = None,
    section: str | None = None,
    metadata: dict | None = None,
    user_id: str = "default"
) -> dict:
    """Store an observation in engram.

    Args:
        content: The observation text to store
        observation_type: Type of observation (error, instruction, decision,
                         code_change, insight, test_result, general, tool_output)
        importance: Override importance score (1-10), or None for type heuristics
        section: ACE section (strategies, snippets, pitfalls, context, preferences)
        metadata: Optional extra metadata
        user_id: User identifier

    Returns:
        Dict with status, memory_id (if created), and importance_score
    """
    data = {
        "content": content,
        "observation_type": observation_type,
        "user_id": user_id
    }

    if importance is not None:
        data["importance"] = importance
    if section is not None:
        data["section"] = section
    if metadata is not None:
        data["metadata"] = metadata

    return _call_internal_api("/internal/observe", data)


def retrieve(
    query: str,
    limit: int = 10,
    memory_types: list[str] | None = None,
    min_importance: float = 0.0,
    recency_weight: float = 0.33,
    importance_weight: float = 0.33,
    relevance_weight: float = 0.33,
    user_id: str = "default"
) -> dict:
    """Retrieve memories from engram.

    Args:
        query: Semantic search query
        limit: Maximum results to return
        memory_types: Filter by types (episodic, semantic, procedural)
        min_importance: Minimum importance threshold
        recency_weight: Weight for recency factor
        importance_weight: Weight for importance factor
        relevance_weight: Weight for relevance factor
        user_id: User identifier

    Returns:
        Dict with memories list, count, and weights used
    """
    data = {
        "query": query,
        "limit": limit,
        "min_importance": min_importance,
        "recency_weight": recency_weight,
        "importance_weight": importance_weight,
        "relevance_weight": relevance_weight,
        "user_id": user_id
    }

    if memory_types is not None:
        data["memory_types"] = memory_types

    return _call_internal_api("/internal/retrieve", data)


def stats(scope: str = "summary", user_id: str = "default") -> dict:
    """Get engram statistics.

    Args:
        scope: Level of detail (summary, detailed, reflection)
        user_id: User identifier

    Returns:
        Dict with statistics based on scope
    """
    return _call_internal_api("/internal/stats", {"scope": scope, "user_id": user_id})


if __name__ == "__main__":
    # Quick test
    try:
        if check_health():
            print("[engram] Server is healthy")
            result = stats()
            print(f"[engram] Stats: {json.dumps(result, indent=2)}")
    except EngramUnavailable as e:
        print(f"[engram] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
