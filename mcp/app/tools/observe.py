"""memory_observe tool for ingesting observations."""

from uuid import UUID

from app.db import get_session
from app.services import ingestion_service


async def memory_observe(
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

    Example:
        >>> memory_observe(
        ...     content="User prefers TypeScript over JavaScript",
        ...     observation_type="instruction",
        ...     section="preferences"
        ... )
        {"status": "created", "memory_id": "...", "importance_score": 10.0}
    """
    try:
        uid = UUID(user_id)
    except ValueError:
        # Use a deterministic UUID for non-UUID user IDs
        import hashlib
        uid = UUID(hashlib.md5(user_id.encode()).hexdigest())

    async with get_session() as session:
        result = await ingestion_service.ingest(
            session=session,
            user_id=uid,
            content=content,
            observation_type=observation_type,
            importance=importance,
            section=section,
            metadata=metadata,
        )
        return result
